import json
import os
import time
from typing import List, Tuple

from modules.options import cmd_opts


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


STOPPED = 0
LOOP = 1
INTERRUPTED = 2


class Context:
    def __init__(self):
        self.model_history = None
        self.history = []
        self.rh = []
        self.max_rounds = 20

        self.state = STOPPED

    def interrupt_and_wait(self):
        # gradio发展神速啊
        self.interrupt()
        import time
        while self.state != STOPPED:
            time.sleep(1)
            print("等待其他线程终止")

    def infer_begin(self, query):
        if self.model_history is None:
            from modules.model import model
            self.model_history = model.create_context()

        self.interrupt_and_wait()
        self.state = LOOP

        hl = len(self.history)
        # 大概不会执行>1次
        while hl >= self.max_rounds:
            self.model_history.remove_first()
            self.history.pop(0)
            self.rh.pop(0)
            hl -= 1

        self.history.append((query, ""))
        self.rh.append((query, ""))
        self.model_history.add_last()

    def interrupt(self):
        if self.state == LOOP:
            self.state = INTERRUPTED

    def infer_loop(self, output) -> bool:
        if self.state != LOOP:
            return True
        else:
            query, _ = self.history[-1]
            self.history[-1] = (query, output)
            self.rh[-1] = (query, output)

        return False

    def infer_end(self) -> None:
        if self.rh:
            query, output = self.rh[-1]
            self.rh[-1] = (query, parse_codeblock(output))
        self.state = STOPPED

    def clear(self) -> None:
        self.interrupt_and_wait()

        if self.model_history:
            self.model_history.clear()
        self.history = []
        self.rh = []

    def revoke(self) -> Tuple[str, str]:
        self.interrupt_and_wait()

        if not self.rh:
            raise "无法撤回！"

        self.model_history.remove_last()
        self.history.pop()
        return self.rh.pop()

    def save_history(self):
        s = [{"q": i[0], "o": i[1]} for i in self.history]

        filename = f"history-{int(time.time())}.json"
        p = os.path.join("outputs", "save", filename)
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False))
        return f"保存到了 {p}"

    def save_as_md(self):
        filename = f"history-{int(time.time())}.md"
        p = os.path.join("outputs", "markdown", filename)
        output = ""
        for i in self.history:
            output += f"# 我: {i[0]}\n\nAI: {i[1]}\n\n"
        with open(p, "w", encoding="utf-8") as f:
            f.write(output)
        return f"保存到了 {p}"

    def load_history(self, file):
        try:
            with open(file.name, "r", encoding='utf-8') as f:
                j = json.load(f)
                self.history = [(i["q"], i["o"]) for i in j]
                self.model_history.from_json(self.history)
                self.rh = [(i[0], parse_codeblock(i[1])) for i in self.history]
        except Exception as e:
            print(e)

        return self.rh


global_ctx = Context() if cmd_opts.shared_session else None
