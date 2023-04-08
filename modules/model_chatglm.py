from torch.cuda import get_device_properties
from transformers import AutoModel, AutoTokenizer

from modules.model import Model, load_cached_model


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

    def infer(self, query: str, ctx,
              max_length: int, top_p: float, temperature: float):
        output_pos = 0
        fn = self.model.stream_chat if ctx.chat else self.model.stream_generate
        try:
            for output, _ in fn(
                    self.tokenizer, query=query, history=ctx.history[0:-1],
                    # transformer自己都说了建议使用这个（doge
                    max_new_tokens=max_length,
                    max_length=None,
                    top_p=top_p,
                    temperature=temperature
            ):
                print(output[output_pos:], end='', flush=True)
                output_pos = len(output)
                yield output
        except Exception as e:
            print(f"生成失败: {repr(e)}")

        print()
