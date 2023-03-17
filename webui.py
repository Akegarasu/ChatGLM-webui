from modules.model import load_model
from modules.options import cmd_opts
from modules.ui import create_ui


def main():
    load_model()
    ui = create_ui()
    ui.queue().launch(
        server_name="0.0.0.0" if cmd_opts.listen else None,
        server_port=cmd_opts.port,
        share=cmd_opts.share
    )


if __name__ == "__main__":
    main()
