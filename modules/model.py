from typing import Optional, List, Tuple

from torch.cuda import get_device_properties

from modules.device import torch_gc
from modules.options import cmd_opts

tokenizer = None
model = None


def prepare_model():
    global model
    if cmd_opts.cpu:
        if cmd_opts.precision == "fp32":
            model = model.float()
        elif cmd_opts.precision == "bf16":
            model = model.bfloat16()
        else:
            model = model.float()
    else:
        if cmd_opts.precision is None:
            total_vram_in_gb = get_device_properties(0).total_memory / 1e9
            print(f'GPU memory: {total_vram_in_gb:.2f} GB')

            if total_vram_in_gb > 30:
                cmd_opts.precision = 'fp32'
            elif total_vram_in_gb > 13:
                cmd_opts.precision = 'fp16'
            elif total_vram_in_gb > 10:
                cmd_opts.precision = 'int8'
            else:
                cmd_opts.precision = 'int4'

            print(f'Choosing precision {cmd_opts.precision} according to your VRAM.'
                  f' If you want to decide precision yourself,'
                  f' please add argument --precision when launching the application.')

        if cmd_opts.precision == "fp16":
            model = model.half().cuda()
        elif cmd_opts.precision == "int4":
            model = model.half().quantize(4).cuda()
        elif cmd_opts.precision == "int8":
            model = model.half().quantize(8).cuda()
        elif cmd_opts.precision == "fp32":
            model = model.float()

    model = model.eval()


def load_model():
    if cmd_opts.ui_dev:
        return

    from transformers import AutoModel, AutoTokenizer

    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(cmd_opts.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(cmd_opts.model_path, trust_remote_code=True)
    prepare_model()


def infer(query,
          history: Optional[List[Tuple]],
          max_length, top_p, temperature, use_stream_chat: bool):
    if cmd_opts.ui_dev:
        yield "hello", "hello, dev mode!"
        return

    if not model:
        raise "Model not loaded"

    if history is None:
        history = []

    output_pos = 0
    if use_stream_chat:
        try:
            for output, history in model.stream_chat(
                    tokenizer, query=query, history=history,
                    max_length=max_length,
                    top_p=top_p,
                    temperature=temperature
            ):
                print(output[output_pos:], end='', flush=True)
                output_pos = len(output)
                yield query, output
        except Exception as e:
            print(f"Generation failed: {repr(e)}")
    else:
        output, history = model.chat(
            tokenizer, query=query, history=history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature
        )

        print(output)
        yield query, output

    print()
    torch_gc()
