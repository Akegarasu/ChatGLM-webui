from typing import Optional, List, Tuple

from transformers import AutoModel, AutoTokenizer

from modules.device import torch_gc
from modules.options import cmd_opts

tokenizer = AutoTokenizer.from_pretrained(cmd_opts.model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(cmd_opts.model_path, trust_remote_code=True)


def prepare_model():
    global model
    if cmd_opts.cpu:
        model = model.float()
    else:
        if cmd_opts.precision == "fp16":
            model = model.half().cuda()
        elif cmd_opts.precision == "int4":
            model = model.half().quantize(4).cuda()
        elif cmd_opts.precision == "int8":
            model = model.half().quantize(8).cuda()

    model = model.eval()


prepare_model()


def infer(query,
          history: Optional[List[Tuple]],
          max_length, top_p, temperature):
    if history is None:
        history = []
    output, history = model.chat(
        tokenizer, query=query, history=history,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature
    )
    print(output)
    torch_gc()
    return query, output
