import os
import random

import gradio as gr

from modules import options
from modules.context import Context, global_ctx
from modules.model import infer
from modules.options import cmd_opts

css = "style.css"
script_path = "scripts"
_gradio_template_response_orig = gr.routes.templates.TemplateResponse


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def predict(ctx, sh, query, max_length, top_p, temperature):
    ctx = myctx(ctx, sh)
    ctx.infer_begin(query)

    yield ctx.rh, "❌"

    for output in infer(
            query=query,
            ctx=ctx,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature
    ):
        yield ctx.rh, gr_show()

        if ctx.infer_loop(output):
            break

    ctx.infer_end()
    yield ctx.rh, "闲"


def regenerate(ctx, sh, max_length, top_p, temperature):
    ctx = myctx(ctx, sh)
    if not ctx.rh:
        raise RuntimeWarning("没有过去的对话")

    query, _ = ctx.revoke()

    for p0, p1 in predict(ctx, sh, query, max_length, top_p, temperature):
        yield p0, p1


def clear_history(ctx, sh):
    ctx = myctx(ctx, sh)
    ctx.clear()
    return gr.update(value=[])


def apply_max_round_click(ctx, sh, max_round, chat_or_generate):
    ctx = myctx(ctx, sh)
    ctx.max_rounds = max_round
    ctx.chat = chat_or_generate
    return f"设置了最大对话轮数 {ctx.max_rounds}"


def myctx(ctx, sh: bool):
    return global_ctx if sh and cmd_opts.shared_session else ctx


api_ctx = {}


def create_ui():
    reload_javascript()

    with gr.Blocks(css=css, analytics_enabled=False) as chat_interface:
        state = gr.State(Context())

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"""<h2><center>文字转文字|{cmd_opts.model_type}</center></h2>""")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            max_length = gr.Slider(minimum=8, maximum=2048, step=4, label='最大生成', value=2048)
                        with gr.Row():
                            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=0.7)
                            temperature = gr.Slider(minimum=0.05, maximum=5.0, step=0.01, label='Temperature', value=1)

                        cmd_output = gr.Textbox(label="Command Output", elem_id="cmd_output")
                        with gr.Row():
                            shared_context = gr.Checkbox(label='共享上下文', value=False, visible=cmd_opts.shared_session)
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear_his_btn = gr.Button("清空对话")
                            sync_his_btn = gr.Button("同步对话")

                        with gr.Row():
                            save_his_btn = gr.Button("保存至文件")
                            load_his_btn = gr.UploadButton("从文件加载", file_types=['file'], file_count='single')

                        with gr.Row():
                            save_md_btn = gr.Button("保存为 MarkDown")

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=800)
                with gr.Row():
                    input_message = gr.Textbox(placeholder="输入你的内容...(按 Ctrl+Enter 发送)", show_label=False, lines=4, elem_id="chat-input").style(container=False)
                    stop_generate = gr.Button("闲", elem_id="del-btn")

                with gr.Row():
                    submit = gr.Button("发送", elem_id="c_generate")
                    regen = gr.Button("重新生成")
                    revoke_btn = gr.Button("撤回")

        submit.click(predict, inputs=[
            state,
            shared_context,
            input_message,
            max_length,
            top_p,
            temperature
        ], outputs=[chatbot, stop_generate])

        regen.click(regenerate, inputs=[
            state,
            shared_context,
            max_length,
            top_p,
            temperature
        ], outputs=[chatbot, stop_generate])

        stop_generate.click(lambda ctx, sh: myctx(ctx, sh).interrupt(), inputs=[state, shared_context], outputs=[])
        def revoke(ctx, sh):
            ctx = myctx(ctx, sh)
            ctx.revoke()
            return ctx.rh
        revoke_btn.click(revoke, inputs=[state, shared_context], outputs=[chatbot])
        clear_his_btn.click(clear_history, inputs=[state, shared_context], outputs=[chatbot])
        save_his_btn.click(lambda ctx, sh: myctx(ctx, sh).save_history(), inputs=[state, shared_context], outputs=[cmd_output])
        save_md_btn.click(lambda ctx, sh: myctx(ctx, sh).save_as_md(), inputs=[state, shared_context], outputs=[cmd_output])
        load_his_btn.upload(lambda ctx, sh, f: myctx(ctx, sh).load_history(f), inputs=[state, shared_context, load_his_btn], outputs=[chatbot])
        sync_his_btn.click(lambda ctx, sh: myctx(ctx, sh).rh, inputs=[state, shared_context], outputs=[chatbot])

        # 未经测试。
        if cmd_opts.api:
            global api_ctx

            session_id = gr.Number(visible=False)
            btn = gr.Button("", visible=False)

            def api_new_session():
                if len(api_ctx) > 10:
                    return "-1"

                while True:
                    id = int(random.random()*10000000)
                    if id not in api_ctx:
                        api_ctx[id] = Context()
                        break
                return str(id)
            btn.click(api_new_session, outputs=[cmd_output], api_name="new_session")

            def api_del_session(id):
                if id in api_ctx:
                    del api_ctx[id]
                    return "true"
                return "false"
            btn.click(api_del_session, inputs=[session_id], outputs=[cmd_output], api_name="del_session")

            def api_generate(id, query, max_length, top_p, temperature):
                if id not in api_ctx:
                    return None

                ctx = api_ctx[id]
                ctx.infer_begin(query)

                for output in infer(
                        query=query,
                        ctx=ctx,
                        max_length=max_length,
                        top_p=top_p,
                        temperature=temperature
                ):
                    if ctx.infer_loop(output):
                        break

                ctx.infer_end()
                return ctx.rh
            btn.click(api_generate, inputs=[
                session_id,
                input_message,
                max_length,
                top_p,
                temperature
            ], outputs=[chatbot], api_name="generate")

            def api_stop(id):
                if id in api_ctx:
                    api_ctx[id].interrupt_and_wait()
            btn.click(api_stop, inputs=[session_id], api_name="stop_generate")

            def api_revoke(id):
                if id not in api_ctx:
                    return None
                api_ctx[id].revoke()
                return api_ctx[id].rh
            btn.click(api_revoke, inputs=[session_id], outputs=[chatbot], api_name="revoke")

    with gr.Blocks(css=css, analytics_enabled=False) as settings_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel"):
                gr.HTML("<h1>通用</h1>")
                max_rounds = gr.Slider(minimum=1, maximum=100, step=1, label="最大对话轮数", value=20)
                # 切换后建议手动清空对话...
                chat_or_generate = gr.Checkbox(label='对话(关闭为续写)', value=True)
                apply_max_rounds = gr.Button("✔", elem_id="del-btn")

            apply_max_rounds.click(apply_max_round_click, inputs=[state, shared_context, max_rounds, chat_or_generate], outputs=[cmd_output])

            with gr.Column(variant="panel"):
                gr.HTML("<h1>ChatRWKV模型</h1>")
                alpha_freq = gr.Slider(minimum=0, maximum=1, step=0.01, label='alpha frequency', value=0.5)
                alpha_pres = gr.Slider(minimum=0, maximum=1, step=0.01, label='alpha presence', value=0.5)
                top_k = gr.Slider(minimum=0, maximum=100, step=1, label='top_k', value=0)
                apply_rwkv_cfg = gr.Button("✔", elem_id="del-btn")

                def apply_rwkv_cfg_click(ctx, sh, alpha_freq, alpha_pres, top_k):
                    if cmd_opts.model_type != "chatrwkv":
                        return "不是ChatRWKV模型"
                    ctx = myctx(ctx, sh)
                    ctx.get_model_history().alpha_frequency = alpha_freq
                    ctx.model_history.alpha_presence = alpha_pres
                    ctx.model_history.top_k = top_k
                    return f"已保存"

                apply_rwkv_cfg.click(apply_rwkv_cfg_click, inputs=[state, shared_context, alpha_freq, alpha_pres, top_k], outputs=[cmd_output])

        with gr.Column():
            reload_ui = gr.Button("重载界面")

        def restart_ui():
            options.need_restart = True

        reload_ui.click(restart_ui)

    interfaces = [
        (chat_interface, "聊天", "chat"),
        (settings_interface, "设置", "settings")
    ]

    with gr.Blocks(css=css, analytics_enabled=False, title="Wenzi2Wenzi") as demo:
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
