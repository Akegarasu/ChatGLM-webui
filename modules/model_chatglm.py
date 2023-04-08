import torch
from torch.cuda import get_device_properties
from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from modules.model import Model, load_cached_model


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 20005] = 5e4
        return scores


class ChatGLMModel(Model):
    def __init__(self):
        super().__init__()

    def load(self, model_path: str, precision: str = None, cpu: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        def load():
            return AutoModel.from_pretrained(model_path, trust_remote_code=True)

        if cpu:
            if precision == "fp32":
                model = load().float()
            elif precision == "bf16":
                model = load().bfloat16()
            else:
                raise f"Unknown precision {precision}"
        else:
            if precision is None:
                total_vram_in_gb = get_device_properties(0).total_memory / 1073741824

                if total_vram_in_gb > 30:
                    precision = 'fp32'
                elif total_vram_in_gb > 13:
                    precision = 'fp16'
                elif total_vram_in_gb > 10:
                    precision = 'int8'
                else:
                    precision = 'int4'

                print(f'根据你的VRAM ({total_vram_in_gb}GiB) 选择了精度 {precision}.'
                      f' 你也可以通过参数 --precision 手动指定.')

            if precision == "fp16":
                model = load().half()
            elif precision == "int4":
                model = load_cached_model(model_path, load, "int4", lambda m: m.half().quantize(4))
            elif precision == "int8":
                model = load_cached_model(model_path, load, "int8", lambda m: m.half().quantize(8))
            elif precision == "fp32":
                model = load().float()
            else:
                raise f"Unknown precision {precision}"

            model = model.cuda()

        self.model = model.eval()

    @torch.no_grad()
    def stream_chat_copy(self, query, history, max_new_tokens, top_p, temperature, chat, logits_processor=None, **kwargs):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())

        # transformer自己都说了建议使用这个（doge
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        prompt = ""
        if not history:
            prompt = query
        elif chat:
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        else:
            for (old_query, response) in enumerate(history):
                prompt += old_query
                prompt += response
            prompt += query

        input_ids = self.tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = input_ids.to(self.model.device)

        for outputs in self.model.stream_generate(**input_ids, **gen_kwargs):
            outputs = outputs.tolist()[0][len(input_ids["input_ids"][0]):]
            response = self.tokenizer.decode(outputs)
            response = self.model.process_response(response)
            yield response

    def infer(self, query: str, ctx,
              max_length: int, top_p: float, temperature: float):
        output_pos = 0
        try:
            for output in self.stream_chat_copy(
                    query=query, history=ctx.history[0:-1],
                    max_new_tokens=max_length,
                    top_p=top_p,
                    temperature=temperature,
                    chat=ctx.chat
            ):
                print(output[output_pos:], end='', flush=True)
                output_pos = len(output)
                yield output
        except Exception as e:
            print(f"生成失败: {repr(e)}")

        print()