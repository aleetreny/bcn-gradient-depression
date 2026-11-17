"""Pre-tool hook for AI coding assistants: blocks unsafe actions before they run, whatever the agent
decided. Stdlib only and standalone, because it runs on every tool call.

Claude Code (.claude/settings.json, committed) calls it as a PreToolUse hook: tool call JSON on stdin,
exit 2 + a reason on stderr blocks the call and shows the reason to the agent. Other assistants are
covered by the git hooks (.githooks/) and CI, which run the same checks after the fact.

    echo '{"tool_name": "Bash", "tool_input": {"command": "git push -f"}}' | python3 harness/agent_hook.py
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.policy import GATED_DIRS, edit_block_reason  # noqa: E402

# (regex on the shell command, reason). Matched anywhere in the command, so chains are covered.
COMMAND_RULES = (
    (r"\bgit\s+commit\b[^|;&]*(\s--no-verify\b|\s-[a-zA-Z]*n[a-zA-Z]*\b)", "skipping the pre-commit leak guard"),
    (r"\bgit\s+push\b[^|;&]*(\s--no-verify\b)", "skipping the pre-push check"),
    (r"\bgit\s+push\b[^|;&]*(\s-f\b|\s--force\b|\s--force-with-lease\b|\s\+\S)", "force-pushing rewrites shared history"),
    (r"\bgit\s+add\b[^|;&]*(\s-f\b|\s--force\b)", "force-adding an ignored file (that is how gated data leaks)"),
    (r"\bgit\s+config\b[^|;&]*core\.hooksPath(?!\s+\.githooks\b)", "changing the git hooks path disables the guard"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f|checkout\s+--\s+\.|restore\s+\.)", "discarding work in bulk: ask a human"),
    (r"\bgit\s+(filter-branch|filter-repo|rebase)\b", "rewriting history: ask a human"),
    (r"--allow-protected\b", "only a human may approve protected changes"),
    (r"\bharness\s+promote\b[^|;&]*--force\b", "forcing a champion past the gate: ask a human"),
    (r"\bgh\s+repo\s+(edit\b[^|;&]*--visibility|delete\b)", "changing repository visibility or deleting it: ask a human"),
    (r"\bgh\s+(gist|release)\s+create\b", "publishing files outside the repository"),
)
WRITERS = re.compile(r">|\btee\b|\b(sed|perl)\s+[^|;&]*-i|\b(cp|mv|truncate|dd|install|ln)\b|\bgit\s+(rm|mv|checkout)\b")
UPLOADERS = re.compile(r"\b(curl|wget|scp|rsync|sftp|ftp|nc|aws\s+s3|gsutil|rclone|gh\s+gist)\b")


def command_block_reason(cmd: str) -> str:
    for pattern, why in COMMAND_RULES:
        if re.search(pattern, cmd):
            return why
    if UPLOADERS.search(cmd) and any(d in cmd for d in GATED_DIRS + ("_atd.", "_smart_tagging", "_homography",
                                                                       "_possession_gt", "submissions/")):
        return "sending gated data, labels or predictions off this machine"
    if re.search(r"\brm\b[^|;&]*\s-[a-zA-Z]*r", cmd):
        try:
            targets = [t for t in shlex.split(cmd) if not t.startswith("-")]
        except ValueError:
            targets = []
        if any(t.rstrip("/") in ("data", "data_external", "local_data", "experiments", ".git", "/", "~", ".")
               for t in targets):
            return "recursive delete of data, experiments or the repository"
    if WRITERS.search(cmd):                        # the command writes: no path it names may be protected
        for token in _tokens(cmd):
            why = edit_block_reason(_rel(token))
            if why:
                return "writing {}: {}".format(token, why)
    return ""


def _tokens(cmd: str) -> list:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    out = []
    for t in parts:
        out.extend(x for x in re.split(r"[<>|;&]+", t) if x and not x.startswith("-"))
    return out


def _rel(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            return str(p)
    return os.path.normpath(path)


def decide(event: dict) -> str:
    tool, inp = event.get("tool_name", ""), event.get("tool_input") or {}
    if tool == "Bash":
        return command_block_reason(inp.get("command", ""))
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        return edit_block_reason(_rel(path)) if path else ""
    return ""


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    why = decide(event)
    if why:
        print("Blocked by the team harness: {}. See AGENTS.md. If this is really needed, stop and ask the "
              "human to do it or approve it.".format(why), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
