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


def predict(query, max_length, top_p, temperature):
    global history
    output, history = model.chat(
        tokenizer, query=query, history=history,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature
    )
    print(history)
    return history


def clear_history():
    global history
    history.clear()
    return gr.update(value=[])


def create_ui():
    with gr.Blocks() as demo:
        prompt = "输入你的内容..."
        gr.Markdown("""<h2><center>ChatGLM WebUI</center></h2>""")
        with gr.Row():
            max_length = gr.Slider(minimum=4, maximum=4096, step=4, label='Max Length', value=2048)
            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=0.7)
            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=0.95)

        chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=800)
        message = gr.Textbox(placeholder=prompt, show_label=False)

        with gr.Row():
            submit = gr.Button("发送")
            clear_input = gr.Button("清空文本框")

        with gr.Row():
            clear = gr.Button("清空对话（上下文）")

        submit.click(predict,
                     inputs=[
                         message,
                         max_length,
                         top_p,
                         temperature
                     ],
                     outputs=[
                         chatbot
                     ])
        clear.click(clear_history, outputs=[chatbot])
        clear_input.click(lambda x: "", inputs=[message], outputs=[message])

    return demo


ui = create_ui()
ui.queue().launch(
    server_name="0.0.0.0" if cmd_opts.listen else None,
    server_port=cmd_opts.port,
    share=cmd_opts.share
)
