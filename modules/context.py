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

    def append(self, query, output) -> str:
        # c: List[Tuple[str, str]]
        ok = parse_codeblock(output)
        self.history.append((query, output))
        self.rh.append((query, ok))
        return ok

    def clear(self):
        self.history = []
        self.rh = []

    def save_history(self):
        if not os.path.exists("outputs"):
            os.mkdir("outputs")

        s = [{"q": i[0], "o": i[1]} for i in self.history]
        filename = f"save-{int(time.time())}.json"
        with open(os.path.join("outputs", filename), "w", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False))

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
