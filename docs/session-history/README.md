# Session history

Records of the Claude Cowork session that built this project (session link:
https://claude.ai/code/session_015f8aoq7vhnyojcUhsa9fXV). Read them when you need the exact wording of a request
or the reasoning behind a decision. For day-to-day work, `CLAUDE.md` and `docs/PROJECT_HISTORY.md` are enough.

| File | What it is |
|---|---|
| `2026-10-01_to_03_earlier_conversation_summary.md` | The summary written when the session's context was compacted. It covers everything from the first request (2026-10-01) to 2026-10-03 01:26 UTC: requests, design, files, every error and fix up to then. The raw transcript of that part is gone. |
| `2026-10-03_cowork_transcript.jsonl` | The raw transcript (JSON lines) from 2026-10-03 01:26 UTC to the handoff: every message, tool call and tool output (≈ 4 MB). **Git-ignored**: it contains personal details (name, e-mail, local paths). |
| `2026-10-03_cowork_conversation_digest.md` | The same transcript made readable: user messages verbatim, assistant replies, one line per tool call, no tool outputs. Made with `tools/transcript_digest.py`. |

How to search the raw transcript (each line is one JSON object; `type` is `user`, `assistant`, …):

```bat
python -c "import json;[print(json.loads(l)['timestamp'], json.loads(l)['type']) for l in open(r'docs/session-history/2026-10-03_cowork_transcript.jsonl', encoding='utf-8')]"
findstr /c:"live.rerun" docs\session-history\2026-10-03_cowork_transcript.jsonl
```

Claude Code saves its own sessions as `.jsonl` files too (under `%USERPROFILE%\.claude\projects\…`). Run
`python tools/transcript_digest.py <file.jsonl> docs/session-history/<date>_claude_code_digest.md` to keep a
readable record of later sessions here as well.
