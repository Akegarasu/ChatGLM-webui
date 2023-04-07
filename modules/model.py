from typing import List, Tuple, Iterable

import numpy as np
import torch

from modules.device import torch_gc
from modules.options import cmd_opts

np.set_printoptions(precision=4, suppress=True, linewidth=200)


class ModelContext:
    def clear(self):
        pass

    def remove_first(self):
        pass

    def remove_last(self):
        pass

    def add_last(self):
        pass

    def from_json(self, history: List[Tuple[str, str]]):
        pass


class Model:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self, model_path: str, precision: str = None, cpu: bool = False):
        pass

    def infer(self, query, ctx,
              max_length: int, top_p: float, temperature: float) -> Iterable:
        pass

    def create_context(self) -> ModelContext:
        return ModelContext()


model: Model = None


def load_model():
    if cmd_opts.ui_dev:
        return

    global model
    if cmd_opts.model_type == 'chatglm':
        from modules.model_chatglm import ChatGLMModel
        model = ChatGLMModel()
    elif cmd_opts.model_type == 'chatrwkv':
        from modules.model_chatrwkv import ChatRWKVModel
        model = ChatRWKVModel()
    else:
        raise f"未知的模型类型{cmd_opts.model_type}"
    model.load(cmd_opts.model_path, cmd_opts.precision, cmd_opts.cpu)


def load_cached_model(model_path: str, load, name: str = None, compressor=None, torch_load=None):
    import pickle, os

    cached_model = os.path.join(model_path, f"cache_{name}.pth")
    if compressor is not None:
        if os.path.isfile(cached_model):
            try:
                if torch_load is not None:
                    return torch_load(cached_model)
                else:
                    with open(cached_model, "rb") as f:
                        return pickle.load(f)
            except Exception as e:
                import traceback
                traceback.print_exception(type(e), e, e.__traceback__)

    model1 = load()

    if compressor is not None:
        if torch_load is not None:
            # modified torch just for me, you can ignore this param
            torch.save(compressor(model1), cached_model, _use_model_foreach=True)
        else:
            model1 = compressor(model1)
            with open(cached_model, "wb") as f:
                pickle.dump(model1, f)
    return model1


def infer(query,
          ctx,
          max_length, top_p, temperature):
    if cmd_opts.ui_dev:
        import time
        while True:
            yield query, "hello, dev mode %s" % time.ctime()
            time.sleep(1)

    global model
    if not model:
        raise "没有加载模型"

    for output in model.infer(query, ctx, max_length, top_p, temperature):
        yield output
    torch_gc()
