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


class Context:
    def __init__(self, history: Optional[List[Tuple[str, str]]] = None):
        if history:
            self.history = history
        else:
            self.history = []
        self.rh = []
        self.max_rounds = 20

    def append(self, query, output) -> str:
        # c: List[Tuple[str, str]]
        ok = parse_codeblock(output)
        self.history.append((query, output))
        self.rh.append((query, ok))
        return ok

    def clear(self):
        self.history = []
        self.rh = []

    def limit_round(self):
        hl = len(self.history)
        if hl == 0:
            return
        elif hl == self.max_rounds:
            self.history.pop(0)
            self.rh.pop(0)
        elif hl > self.max_rounds:
            self.history = self.history[-self.max_rounds:]
            self.rh = self.rh[-self.max_rounds:]

    # 保存为json文件，供webui读取使用
    def save_history(self):
        if not os.path.exists("chat_history"):
            os.mkdir("chat_history")

        s = [{"q": i[0], "o": i[1]} for i in self.history]
        filename = f"history-{int(time.time())}.json"
        with open(os.path.join("chat_history", filename), "w", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False))

    # 增加保存为markdown文档的功能，方便人类查看
    def save_as_md(self):
        if not os.path.exists("markdown"):
            os.mkdir("markdown")

        filename = f"history-{int(time.time())}.md"
        with open(os.path.join("markdown", filename), "w", encoding="utf-8") as f:
            for i in self.history:
                f.write(f"# 我: {i[0]}\n\n")
                f.write(f"ChatGLM: {i[1]}\n\n")

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
