"""Turn a Claude Code / Cowork session transcript (.jsonl) into a readable Markdown digest.

    python tools/transcript_digest.py <session.jsonl> [out.md]

The digest keeps every user message word for word, every assistant reply, and one line per tool call
(tool name + its short description or command). Tool OUTPUTS are left out because they are huge; the raw
.jsonl still has them. Claude Code keeps its own session transcripts in ~/.claude/projects/<folder>/*.jsonl,
so the same script works for future sessions too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SKIP_USER_PREFIXES = ("<system-reminder>", "<command-", "<local-command")


def _text_blocks(content):
    if isinstance(content, str):
        return [content]
    return [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]


def _tool_line(block) -> str:
    name = block.get("name", "?")
    inp = block.get("input") or {}
    hint = (inp.get("description") or inp.get("command") or inp.get("file_path") or inp.get("path")
            or inp.get("subject") or inp.get("query") or inp.get("method") or "")
    hint = " ".join(str(hint).split())
    return f"- `{name}` {hint[:160]}"


def digest(src: Path) -> str:
    out = [f"# Session digest: `{src.name}`", "",
           "User messages are verbatim. Assistant replies are complete. Tool calls are one line each, and tool "
           "outputs are omitted (see the raw .jsonl).", ""]
    tools: list[str] = []

    def flush_tools():
        if tools:
            out.append(f"<details><summary>{len(tools)} tool call(s)</summary>\n")
            out.extend(tools)
            out.append("\n</details>\n")
            tools.clear()

    for line in src.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind, ts = d.get("type"), (d.get("timestamp") or "")[:19].replace("T", " ")
        msg = d.get("message") or {}
        if kind == "user":
            content = msg.get("content")
            if isinstance(content, list) and all(isinstance(b, dict) and b.get("type") == "tool_result"
                                                 for b in content):
                continue                                   # tool outputs: skipped
            for text in _text_blocks(content):
                text = text.strip()
                if not text or text.startswith(SKIP_USER_PREFIXES):
                    continue
                flush_tools()
                title = "Summary of the earlier conversation" if d.get("isCompactSummary") else "User"
                out += [f"## {title} · {ts} UTC", "", text, ""]
        elif kind == "assistant":
            for b in msg.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    flush_tools()
                    out += [f"### Assistant · {ts} UTC", "", b["text"].strip(), ""]
                elif b.get("type") == "tool_use":
                    tools.append(_tool_line(b))
    flush_tools()
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".md")
    dst.write_text(digest(src), encoding="utf-8")
    print(f"wrote {dst} ({dst.stat().st_size // 1024} KB)")
