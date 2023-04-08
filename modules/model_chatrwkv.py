import copy
import os

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from modules.model import Model, ModelContext, load_cached_model
from modules.options import cmd_opts

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Tune these below (test True/False for all of them) to find the fastest setting:
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)

########################################################################################################
# 1. set os.environ["RWKV_CUDA_ON"] = '1' if possible, for faster preprocess of a long ctx.
# 2. Reuse the state (use deepcopy to clone it) when you are running the same ctx multiple times.
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'  # manually ?

END_OF_TEXT = 0
END_OF_LINE = 187

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150

NINF = -float('inf')


def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)

    sorted_ids = torch.argsort(probs)
    if top_p < 1:
        sorted_probs = probs[sorted_ids]

        i = len(sorted_probs) - 1
        sum = 0
        while i > 0:
            sum += sorted_probs[i].item()
            if sum > top_p:
                probs[probs < sorted_probs[i]] = 0
                break

            i -= 1

    if len(probs) > top_k > 0:
        probs[sorted_ids[:-top_k]] = 0
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    try:
        out = torch.multinomial(probs, num_samples=1)[0]
    except:
        raise "temperature太小"

    return int(out)


cached_codes = {}


class ChatRWKVContext(ModelContext):
    def __init__(self, model,
                 temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2,
                 token_ban=None, token_stop=None, token_not_repeat=None,
                 chunk_len=256, prompt_file=None):
        super().__init__()
        self.model = model

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency  # Frequency Penalty (as in GPT-3)
        self.alpha_presence = alpha_presence  # Presence Penalty (as in GPT-3)
        self.token_ban = [] if token_ban is None else token_ban  # ban the generation of some tokens
        self.token_stop = [] if token_stop is None else token_stop  # stop generation whenever you see any token here
        self.token_not_repeat = [] if token_not_repeat is None else token_not_repeat
        self.states = []
        self.chunk_len = chunk_len  # split input into chunks to save VRAM (shorter -> slower)

        if prompt_file is not None and len(prompt_file):
            self.token_stop.append(END_OF_LINE)

            global cached_codes
            if prompt_file in cached_codes:
                code = cached_codes[prompt_file]
            else:
                with open(prompt_file+".py", 'rb') as file:
                    cached_codes[prompt_file] = code = compile(file.read(), prompt_file, 'exec')

            exec(code, self.__dict__)
        else:
            self.init_prompt = None

    def clear(self):
        self.states = []

    def remove_first(self):
        self.states.pop(0)

    def remove_last(self):
        self.states.pop()

    def add_last(self):
        if self.states:
            self.states.append(copy.deepcopy(self.states[-1]))
        else:
            self.states.append(None)

    def from_json(self, history):
        self.states.clear()

        state = None
        for q, a in history:
            q = self.prepare_prompt(q)
            tokens = self.model.encode(q+a)

            state = copy.deepcopy(state)
            while True:
                out, state = self.model.forward(tokens[:self.chunk_len], state)
                tokens = tokens[self.chunk_len:]
                if len(tokens) == 0:
                    break

            self.states.append(state)

    def prepare_prompt(self, query: str):
        if self.init_prompt:
            prefix = self.init_prompt if self.states[-1] is None else ""
            query = f"{prefix}\n{self.user}{self.interface} {query}\n\n{self.bot}: "
        return query


class ChatRWKVModel(Model):
    def __init__(self):
        super().__init__()

    def load(self, model_path: str, precision: str = None, cpu: bool = False):
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))

        ########################################################################################################
        #
        # Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
        #
        # fp16 = good for GPU (!!! DOES NOT support CPU !!!)
        # fp32 = good for CPU
        # bf16 = worse accuracy, supports CPU
        # xxxi8 (example: fp16i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
        #
        # Read https://pypi.org/project/rwkv/ for Strategy Guide
        #
        ########################################################################################################
        import re
        from modules.rwkv.model import RWKV, no_device_info

        def load(model=os.path.join(model_path, "model.pth")):
            return RWKV(model=model, strategy=precision, verbose=True)

        self.model = load_cached_model(model_path, load, re.sub(r'[\\/:*?"<>[]', '_', no_device_info(precision)), lambda m: m.w, load).eval()

    def encode(self, s: str):
        return self.tokenizer.encode(s).ids

    def stream_generate(self, query, token_count=100, ctx: ChatRWKVContext = None):
        out_buffer = []
        tokens = self.encode(query)

        occurrence = {}
        state = ctx.states[-1]

        for i in range(token_count):
            while len(tokens):
                out, state = self.model.forward(tokens[:ctx.chunk_len], state)
                tokens = tokens[ctx.chunk_len:]

            for n in ctx.token_ban:
                out[n] = NINF
            for n in occurrence:
                out[n] -= (ctx.alpha_presence + occurrence[n] * ctx.alpha_frequency)

            if ctx.init_prompt:  # adjust \n probability
                if i <= 0:
                    out[END_OF_LINE] = NINF
                elif i <= CHAT_LEN_SHORT:
                    out[END_OF_LINE] += (i - CHAT_LEN_SHORT) / 10
                elif i > CHAT_LEN_LONG:
                    out[END_OF_LINE] += min(3, (i - CHAT_LEN_LONG) * 0.25)  # MUST END THE GENERATION

            # sampler
            token = sample_logits(out, temperature=ctx.temperature, top_p=ctx.top_p, top_k=ctx.top_k)
            if token in ctx.token_stop:
                break

            if token in ctx.token_not_repeat:
                out[token] = NINF

            out_buffer.append(token)
            tokens = [token]
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            # output
            tmp = self.tokenizer.decode(out_buffer)
            if '\ufffd' not in tmp:  # is valid utf-8 string?
                yield tmp
                ctx.states[-1] = state
                out_buffer.clear()
            else:
                # for stop
                yield ""

        # yield "<长度限制,清空输入框并点击发送以继续>"

    def infer(self, query: str,
              ctx,
              max_length: int, top_p: float, temperature: float):
        ctx = ctx.model_history
        ctx.temperature = temperature
        ctx.top_p = top_p

        query = ctx.prepare_prompt(query) if ctx.chat else query

        try:
            out_str = ''
            for output in self.stream_generate(query, token_count=max_length, ctx=ctx):
                print(output, end='', flush=True)

                out_str += output
                yield out_str
        except Exception as e:
            import traceback
            traceback.print_tb(e.__traceback__)
            print(f"生成失败: {repr(e)}")

        print()

    def create_context(self) -> ModelContext:
        # For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
        # https://platform.openai.com/docs/api-reference/parameter-details
        return ChatRWKVContext(model=self,
                               top_k=0,  # top_k = 0 then ignore
                               alpha_frequency=0.5,
                               alpha_presence=0.5,
                               token_ban=[],  # ban the generation of some tokens
                               token_stop=[END_OF_TEXT],  # stop generation whenever you see any token here
                               token_not_repeat=self.encode("，：？！"),
                               chunk_len=256,  # split input into chunks to save VRAM (shorter -> slower)
                               prompt_file=cmd_opts.rwkv_template)