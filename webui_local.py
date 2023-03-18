from transformers import AutoModel, AutoTokenizer
from options import parser

history = []
# cmd_opts = parser.parse_args()

class CmdOpts:
    def __init__(self):
        self.model_path = "./model/chatglm-6b"

tokenizer = AutoTokenizer.from_pretrained(cmd_opts.model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(cmd_opts.model_path, trust_remote_code=True)

model = model.half().quantize(4).cuda()
model = model.eval()

def predict(input):
    global history
    output, history = model.chat(tokenizer, input, history)
    print("[ChatGLM]:  " + history[-1][1] )

def clear_history():
    global history
    history.clear()
    print('OK')

print('-'*60)
while True:
    c = input("[Me]:  ")
    if c == "clear":
        clear_history()
    else:
        predict(c)
