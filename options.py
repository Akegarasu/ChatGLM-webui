import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--port", type=int, default="17860")
parser.add_argument("--model-path", type=str, default="THUDM/chatglm-6b")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["fp16", "int4", "int8"], default="fp16")
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--cpu", action='store_true', help="use cpu")
parser.add_argument("--share", action='store_true', help="use gradio share")
