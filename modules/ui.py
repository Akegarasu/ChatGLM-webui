import os

import gradio as gr

from modules import options
from modules.context import Context
from modules.model import infer

css = "style.css"
script_path = "scripts"
_gradio_template_response_orig = gr.routes.templates.TemplateResponse


def predict(ctx, query, max_length, top_p, temperature, use_stream_chat):
    ctx.limit_round()
    ctx.limit_word()
    flag = True
    for _, output in infer(
            query=query,
            history=ctx.history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            use_stream_chat=use_stream_chat
    ):
        if flag:
            ctx.append(query, output)
            flag = False
        else:
            ctx.update_last(query, output)
        yield ctx.rh, ""
    ctx.refresh_last()
    yield ctx.rh, ""

def regenerate(ctx, max_length, top_p, temperature, use_stream_chat):
    if ctx.history and ctx.rh:
        query = ctx.history[-1][0]
        ctx.revoke()
        ctx.limit_round()
        ctx.limit_word()
        flag = True
        for _, output in infer(
                query=query,
                history=ctx.history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                use_stream_chat=use_stream_chat
        ):
            if flag:
                ctx.append(query, output)
                flag = False
            else:
                ctx.update_last(query, output)
            yield ctx.rh, ""
        ctx.refresh_last()
        yield ctx.rh, ""

def clear_history(ctx):
    ctx.clear()
    return gr.update(value=[])

def edit_history(ctx, log, idx):
    if log == '':
        return ctx.rh, {'visible': True, '__type__': 'update'},  {'value': ctx.history[idx[0]][idx[1]], '__type__': 'update'}, idx
    ctx.edit_history(log, idx[0], idx[1])
    return ctx.rh, *gr_hide()

def gr_show_and_load(ctx, evt: gr.SelectData):
    if evt.index[1] == 0:
        label = f'修改提问内容{evt.index[0]}：'
    else:
        label = f'修改回答内容{evt.index[0]}：'
    return {'visible': True, '__type__': 'update'}, {'value': ctx.history[evt.index[0]][evt.index[1]], 'label': label, '__type__': 'update'}, evt.index

def gr_hide():
    return {'visible': False, '__type__': 'update'}, {'value': '', 'label': '', '__type__': 'update'}, []

def apply_max_round_click(ctx, max_round):
    ctx.max_rounds = max_round
    return f"成功设置: 最大对话轮数 {ctx.max_rounds}"

def apply_max_words_click(ctx, max_words):
    ctx.max_words = max_words
    return f"成功设置: 最大对话字数 {ctx.max_words}"

def create_ui():
    reload_javascript()

    with gr.Blocks(css=css, analytics_enabled=False) as chat_interface:
        _ctx = Context()
        state = gr.State(_ctx)
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
                            max_rounds = gr.Slider(minimum=1, maximum=100, step=1, label="最大对话轮数", value=20)
                            apply_max_rounds = gr.Button("✔", elem_id="del-btn")

                        with gr.Row():
                            max_words = gr.Slider(minimum=4, maximum=4096, step=4, label='最大对话字数', value=2048)
                            apply_max_words = gr.Button("✔", elem_id="del-btn")

                        cmd_output = gr.Textbox(label="消息输出")
                        with gr.Row():
                            use_stream_chat = gr.Checkbox(label='使用流式输出', value=True)
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear_history_btn = gr.Button("清空对话")

                        with gr.Row():
                            sync_his_btn = gr.Button("同步对话")

                        with gr.Row():
                            save_his_btn = gr.Button("保存对话")
                            load_his_btn = gr.UploadButton("读取对话", file_types=['file'], file_count='single')

                        with gr.Row():
                            save_md_btn = gr.Button("保存为 MarkDown")

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            gr.Markdown('''说明:<br/>`Max Length` 生成文本时的长度限制<br/>`Top P` 控制输出文本中概率最高前 p 个单词的总概率<br/>`Temperature` 控制生成文本的多样性和随机性<br/>`Top P` 变小会生成更多样和不相关的文本；变大会生成更保守和相关的文本。<br/>`Temperature` 变小会生成更保守和相关的文本；变大会生成更奇特和不相关的文本。<br/>`最大对话轮数` 对话记忆轮数<br/>`最大对话字数` 对话记忆字数<br/>限制记忆可减小显存占用。<br/>点击对话可直接修改对话内容''')

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=800)
                with gr.Row(visible=False) as edit_log:
                    with gr.Column():
                        log = gr.Textbox(placeholder="输入你修改后的内容", show_label=False, lines=4, elem_id="chat-input").style(container=False)
                        with gr.Row():
                            submit_log = gr.Button('保存')
                            cancel_log = gr.Button('取消')
                log_idx = gr.State([])

                with gr.Row():
                    input_message = gr.Textbox(placeholder="输入你的内容...(按 Ctrl+Enter 发送)", show_label=False, lines=4, elem_id="chat-input").style(container=False)
                    clear_input = gr.Button("🗑️", elem_id="del-btn")

                with gr.Row():
                    submit = gr.Button("发送", elem_id="c_generate")

                with gr.Row():
                    revoke_btn = gr.Button("撤回")
                
                with gr.Row():
                    regenerate_btn = gr.Button("重新生成")

        submit.click(predict, inputs=[
            state,
            input_message,
            max_length,
            top_p,
            temperature,
            use_stream_chat
        ], outputs=[
            chatbot,
            input_message
        ])

        regenerate_btn.click(regenerate, inputs=[
            state,
            max_length,
            top_p,
            temperature,
            use_stream_chat
        ], outputs=[
            chatbot,
            input_message
        ])
        revoke_btn.click(lambda ctx: ctx.revoke(), inputs=[state], outputs=[chatbot])
        clear_history_btn.click(clear_history, inputs=[state], outputs=[chatbot])
        clear_input.click(lambda x: "", inputs=[input_message], outputs=[input_message])
        save_his_btn.click(lambda ctx: ctx.save_history(), inputs=[state], outputs=[cmd_output])
        save_md_btn.click(lambda ctx: ctx.save_as_md(), inputs=[state], outputs=[cmd_output])
        load_his_btn.upload(lambda ctx, f: ctx.load_history(f), inputs=[state, load_his_btn], outputs=[chatbot])
        sync_his_btn.click(lambda ctx: ctx.rh, inputs=[state], outputs=[chatbot])
        apply_max_rounds.click(apply_max_round_click, inputs=[state, max_rounds], outputs=[cmd_output])
        apply_max_words.click(apply_max_words_click, inputs=[state, max_words], outputs=[cmd_output])
        chatbot.select(gr_show_and_load, inputs=[state], outputs=[edit_log, log, log_idx])
        submit_log.click(edit_history, inputs=[state, log, log_idx], outputs=[chatbot, edit_log, log, log_idx])
        cancel_log.click(gr_hide, outputs=[edit_log, log, log_idx])

    with gr.Blocks(css=css, analytics_enabled=False) as settings_interface:
        with gr.Row():
            reload_ui = gr.Button("Reload UI")

        def restart_ui():
            options.need_restart = True

        reload_ui.click(restart_ui)

    interfaces = [
        (chat_interface, "Chat", "chat"),
        (settings_interface, "Settings", "settings")
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
