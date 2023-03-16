import torch
from modules import options

cpu = torch.device("cpu")
cuda_available = torch.cuda.is_available()


def get_cuda_device():
    if options.cmd_opts.device_id is not None:
        return f"cuda:{options.cmd_opts.device_id}"

    return "cuda"


def get_optimal_device():
    if cuda_available:
        return torch.device(get_cuda_device())
    return cpu


def torch_gc():
    if cuda_available:
        with torch.cuda.device(get_cuda_device()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


device = cpu if options.cmd_opts.cpu else get_optimal_device()

if not cuda_available:
    print("CUDA is not available, using cpu mode...")
