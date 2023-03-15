import gradio as gr
from transformers import AutoModel, AutoTokenizer
from options import parser

history = []
cmd_opts = parser.parse_args()

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


def predict(input):
    global history
    output, history = model.chat(tokenizer, input, history)
    print(history)
    return history


with gr.Blocks(css="#chat-box {white-space: pre-line;}") as demo:
    prompt = "输入你的内容..."
    gr.Markdown("""<h2><center>ChatGLM WebUI</center></h2>""")
    chatbot = gr.Chatbot(elem_id="chat-box")
    message = gr.Textbox(placeholder=prompt)
    state = gr.State()
    with gr.Row():
        submit = gr.Button("发送")
        clear = gr.Button("清空对话")


    def clear_history():
        global history
        history.clear()
        return gr.update(value=[])


    submit.click(predict, inputs=[message], outputs=[chatbot])
    clear.click(clear_history, outputs=[chatbot])

demo.queue().launch(
    server_port=cmd_opts.port,
    share=cmd_opts.share
)
