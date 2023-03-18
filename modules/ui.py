import os

import gradio as gr

from modules.context import ctx
from modules.device import torch_gc
from modules.model import infer

css = "style.css"
script_path = "scripts"
_gradio_template_response_orig = gr.routes.templates.TemplateResponse


def predict(query, max_length, top_p, temperature):
    ctx.limit_round()
    _, output = infer(
        query=query,
        history=ctx.history,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature
    )
    ctx.append(query, output)
    # for clear input textbox
    return ctx.history, ""


def clear_history():
    ctx.clear()
    return gr.update(value=[])


def apply_max_round_click(max_round):
    ctx.max_rounds = max_round


def create_ui():
    reload_javascript()

    with gr.Blocks(css=css, analytics_enabled=False) as chat_interface:
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

                        with gr.Row():
                            max_rounds = gr.Slider(minimum=1, maximum=100, step=1, label="最大对话轮数（调小可以显著改善爆显存，但是会丢失上下文）", value=20)
                            apply_max_rounds = gr.Button("✔", elem_id="del-btn")

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear = gr.Button("清空对话（上下文）")

                        with gr.Row():
                            save_his_btn = gr.Button("保存对话")
                            load_his_btn = gr.UploadButton("读取对话", file_types=['file'], file_count='single')

                        with gr.Row():
                            save_md_btn = gr.Button("保存为 MarkDown")

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=800)
                with gr.Row():
                    input_message = gr.Textbox(placeholder=prompt, show_label=False, lines=2, elem_id="chat-input")
                    clear_input = gr.Button("🗑️", elem_id="del-btn")

                with gr.Row():
                    submit = gr.Button("发送", elem_id="c_generate")

        submit.click(predict, inputs=[
            input_message,
            max_length,
            top_p,
            temperature
        ], outputs=[
            chatbot,
            input_message
        ])

        clear.click(clear_history, outputs=[chatbot])
        clear_input.click(lambda x: "", inputs=[input_message], outputs=[input_message])

        save_his_btn.click(ctx.save_history)
        save_md_btn.click(ctx.save_as_md)
        load_his_btn.upload(ctx.load_history, inputs=[
            load_his_btn,
        ], outputs=[
            chatbot
        ])

        apply_max_rounds.click(apply_max_round_click, inputs=[max_rounds])

        interfaces = [
            (chat_interface, "Chat", "chat"),
        ]

    with gr.Blocks(css=css, analytics_enabled=False, title="ChatGLM") as demo:
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id="tab_" + ifid):
                    interface.render()

    return demo


def reload_javascript():
    scripts_list = [os.path.join(script_path, i) for i in os.listdir(script_path) if i.endswith(".js")]
    javascript = ""
    # with open("script.js", "r", encoding="utf8") as js_file:
    #     javascript = f'<script>{js_file.read()}</script>'

    for path in scripts_list:
        with open(path, "r", encoding="utf8") as js_file:
            javascript += f"\n<script>{js_file.read()}</script>"

    # todo: theme
    # if cmd_opts.theme is not None:
    #     javascript += f"\n<script>set_theme('{cmd_opts.theme}');</script>\n"

    def template_response(*args, **kwargs):
        res = _gradio_template_response_orig(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
