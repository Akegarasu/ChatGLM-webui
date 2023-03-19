import os
from modules.model import load_model
from modules.options import cmd_opts
from modules.ui import create_ui


def ensure_output_dirs():
    folders = ["outputs/save", "outputs/markdown"]

    def check_and_create(p):
        if not os.path.exists(p):
            os.makedirs(p)

    for i in folders:
        check_and_create(i)


def init():
    ensure_output_dirs()
    load_model()


def main():
    ui = create_ui()
    ui.queue(concurrency_count=5, max_size=20).launch(
        server_name="0.0.0.0" if cmd_opts.listen else None,
        server_port=cmd_opts.port,
        share=cmd_opts.share
    )


if __name__ == "__main__":
    init()
    main()
