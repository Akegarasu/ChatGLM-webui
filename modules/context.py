from typing import Optional, List, Tuple
import json
import os
import time


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
LOOP_FIRST = 1
LOOP = 2
INTERRUPTED = 3


class Context:
    def __init__(self, history: Optional[List[Tuple[str, str]]] = None):
        self.history = history if history else []
        self.rh = []
        self.max_rounds = 20

        self.state = STOPPED

    def inferBegin(self):
        # gradio发展神速啊
        # self.interrupt()
        # import time
        # while self.state != STOPPED:
        #     time.sleep(1)
        #     print("等待其他线程终止")

        self.state = LOOP_FIRST

        hl = len(self.history)
        if hl == 0:
            return
        elif hl == self.max_rounds:
            self.history.pop(0)
            self.rh.pop(0)
        elif hl > self.max_rounds:
            self.history = self.history[-self.max_rounds:]
            self.rh = self.rh[-self.max_rounds:]

    def interrupt(self):
        if self.state == LOOP_FIRST or self.state == LOOP:
            self.state = INTERRUPTED

    def inferLoop(self, query, output) -> bool:
        # c: List[Tuple[str, str]]
        if self.state == INTERRUPTED:
            return True
        elif self.state == LOOP_FIRST:
            self.history.append((query, output))
            self.rh.append((query, parse_codeblock(output)))

            self.state = LOOP
        else:
            self.history[-1] = (query, output)
            self.rh[-1] = (query, output)

        return False

    def inferEnd(self) -> None:
        if self.rh:
            query, output = self.rh[-1]
            self.rh[-1] = (query, parse_codeblock(output))
        self.state = STOPPED

    def clear(self) -> None:
        self.history = []
        self.rh = []

    def revoke(self) -> List[Tuple[str, str]]:
        if self.rh:
            self.history.pop()
            self.rh.pop()
        return self.rh

    def save_history(self):
        s = [{"q": i[0], "o": i[1]} for i in self.history]
        filename = f"history-{int(time.time())}.json"
        p = os.path.join("outputs", "save", filename)
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False))
        return f"Successful saved to: {p}"

    def save_as_md(self):
        filename = f"history-{int(time.time())}.md"
        p = os.path.join("outputs", "markdown", filename)
        output = ""
        for i in self.history:
            output += f"# 我: {i[0]}\n\nChatGLM: {i[1]}\n\n"
        with open(p, "w", encoding="utf-8") as f:
            f.write(output)
        return f"Successful saved to: {p}"

    def load_history(self, file):
        try:
            with open(file.name, "r", encoding='utf-8') as f:
                j = json.load(f)
                _hist = [(i["q"], i["o"]) for i in j]
                _readable_hist = [(i["q"], parse_codeblock(i["o"])) for i in j]
        except Exception as e:
            print(e)
        self.history = _hist.copy()
        self.rh = _readable_hist.copy()
        return self.rh


ctx = Context()
