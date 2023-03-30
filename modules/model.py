from typing import Optional, List, Tuple

from torch.cuda import get_device_properties
from transformers import AutoModel, AutoTokenizer

from modules.device import torch_gc
from modules.options import cmd_opts

tokenizer = None
model = None


def load_model_file(name: str, func=None):
    import pickle, os

    cache_model = os.path.join(cmd_opts.model_path, f"model_{name}.bin")
    if func is not None:
        if os.path.isfile(cache_model):
            with open(cache_model, "rb") as f:
                return pickle.load(f)

    model1 = AutoModel.from_pretrained(cmd_opts.model_path, trust_remote_code=True)

    if func is not None:
        model1 = func(model1)
        with open(cache_model, "wb") as f:
            pickle.dump(model1, f)
    return model1


def load_model():
    if cmd_opts.ui_dev:
        return

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cmd_opts.model_path, trust_remote_code=True)

    global model
    if cmd_opts.cpu:
        if cmd_opts.precision == "fp32":
            model = load_model_file("fp32").float()
        elif cmd_opts.precision == "bf16":
            model = load_model_file("bf16").bfloat16()
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
            model = load_model_file("fp16").half()
        elif cmd_opts.precision == "int4":
            model = load_model_file("int4", lambda m: m.half().quantize(4))
        elif cmd_opts.precision == "int8":
            model = load_model_file("int8", lambda m: m.half().quantize(8))
        elif cmd_opts.precision == "fp32":
            model = load_model_file("fp32").float()

        model = model.cuda()

    model = model.eval()


def infer(query,
          history: Optional[List[Tuple]],
          max_length, top_p, temperature, use_stream_chat: bool):
    if cmd_opts.ui_dev:
        import time
        while True:
          yield query, "hello, dev mode %s" % time.ctime()
          time.sleep(1)

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
