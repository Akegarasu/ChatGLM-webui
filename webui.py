import json
import os
import time

import gradio as gr
from transformers import AutoModel, AutoTokenizer
from options import parser

history = []
readable_history = []
cmd_opts = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(cmd_opts.model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(cmd_opts.model_path, trust_remote_code=True)

_css = """
#del-btn {
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
    margin: 1.5em 0;
}
"""


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


def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(lines)


def predict(query, max_length, top_p, temperature):
    global history
    output, history = model.chat(
        tokenizer, query=query, history=history,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature
    )
    readable_history.append((query, parse_codeblock(output)))
    print(output)
    return readable_history


def save_history():
    if not os.path.exists("outputs"):
        os.mkdir("outputs")

    s = [{"q": i[0], "o": i[1]} for i in history]
    filename = f"save-{int(time.time())}.json"
    with open(os.path.join("outputs", filename), "w", encoding="utf-8") as f:
        f.write(json.dumps(s, ensure_ascii=False))


def load_history(file):
    global history, readable_history
    try:
        with open(file.name, "r", encoding='utf-8') as f:
            j = json.load(f)
            _hist = [(i["q"], i["o"]) for i in j]
            _readable_hist = [(i["q"], parse_codeblock(i["o"])) for i in j]
    except Exception as e:
        print(e)
        return readable_history
    history = _hist.copy()
    readable_history = _readable_hist.copy()
    return readable_history


def clear_history():
    history.clear()
    readable_history.clear()
    return gr.update(value=[])


def create_ui():
    with gr.Blocks(css=_css) as demo:
        prompt = "输入你的内容..."
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""<h2><center>ChatGLM WebUI</center></h2>""")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            max_length = gr.Slider(minimum=4, maximum=4096, step=4, label='Max Length', value=2048)
                            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=0.7)
                        with gr.Row():
                            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=0.95)

                        # with gr.Row():
                        #     max_rounds = gr.Slider(minimum=1, maximum=50, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear = gr.Button("清空对话（上下文）")

                        with gr.Row():
                            save_his_btn = gr.Button("保存对话")
                            load_his_btn = gr.UploadButton("读取对话", file_types=['file'], file_count='single')

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=800)
                with gr.Row():
                    message = gr.Textbox(placeholder=prompt, show_label=False, lines=2)
                    clear_input = gr.Button("🗑️", elem_id="del-btn")

                with gr.Row():
                    submit = gr.Button("发送")

        submit.click(predict, inputs=[
            message,
            max_length,
            top_p,
            temperature
        ], outputs=[
            chatbot
        ])

        clear.click(clear_history, outputs=[chatbot])
        clear_input.click(lambda x: "", inputs=[message], outputs=[message])

        save_his_btn.click(save_history)
        load_his_btn.upload(load_history, inputs=[
            load_his_btn,
        ], outputs=[
            chatbot
        ])

    return demo


ui = create_ui()
ui.queue().launch(
    server_name="0.0.0.0" if cmd_opts.listen else None,
    server_port=cmd_opts.port,
    share=cmd_opts.share
)
