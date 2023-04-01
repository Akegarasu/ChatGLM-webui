import os
import time
import sysconfig
from modules import options

from modules.model import load_model

from modules.options import cmd_opts
from modules.ui import create_ui

# patch PATH for cpm_kernels libcudart lookup
os.environ['PATH'] = os.environ.get("PATH", "") + os.pathsep + os.path.join(sysconfig.get_paths()["purelib"], "torch\lib")


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


def wait_on_server(ui=None):
    while 1:
        time.sleep(1)
        if options.need_restart:
            options.need_restart = False
            time.sleep(0.5)
            ui.close()
            time.sleep(0.5)
            break


def main():
    while True:
        ui = create_ui()
        ui.queue(concurrency_count=5, max_size=64).launch(
            server_name="0.0.0.0" if cmd_opts.listen else None,
            server_port=cmd_opts.port,
            share=cmd_opts.share,
            prevent_thread_lock=True
        )
        wait_on_server(ui)
        print('Restarting UI...')


if __name__ == "__main__":
    init()
    main()
