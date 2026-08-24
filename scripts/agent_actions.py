"""Agent action parsing and transcript rendering shared by the dialogue runner.

parse_action() is deliberately liberal: it recognizes every JSON action encoding the
tested models actually emit (canonical tool/args, toolname-as-key, fenced JSON,
top-level tool, action-as-string) and falls back to treating raw text as a
send_message. render() formats a transcript for a prompt. MAX_STEPS caps the agent
turns per round.

  python scripts/agent_actions.py --selftest    # offline parser checks, no API
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from knownliebench.envs.refund.tools import ALL_TOOLS, ALIASES
from knownliebench.utils.json_utils import parse_json_object

# Known tool names (canonical + aliases) so we can recognize the common LLM encoding
# {"action": {"<toolname>": <value>}} where the tool NAME is the dict key (gpt-4o does this).
_KNOWN_TOOLS = {t.get_info()["function"]["name"] for t in ALL_TOOLS} | set(ALIASES)
_REPLY_LIKE = {"send_message", "reply", "respond", "message"}
_POLICY_LIKE = {"get_policy", "check_policy"}
_ACTION_META = {"tool", "name", "action", "thought", "args", "arguments", "parameters"}

MAX_STEPS = 10

def _coerce_args(tool, value):
    """A tool given as {"<tool>": <value>}: turn a scalar value into the right arg dict."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        if tool in _REPLY_LIKE:
            return {"text": value}
        if tool in _POLICY_LIKE:
            return {"query": value}
        return {"order_id": value}
    return {}

def _extract(d, known_tools=None):
    """Given a dict describing an action, return (tool, args) or None.
    Handles: {"tool":"x","args":{...}}, {"name":"x",...}, {"action":"x","text":...},
    and the toolname-as-key form {"<toolname>": <scalar-or-dict>}.
    `known_tools` is the recognized tool-name set (defaults to the refund _KNOWN_TOOLS, so the
    refund path is byte-identical; other domains pass their own env's tool names)."""
    kt = _KNOWN_TOOLS if known_tools is None else known_tools
    if not isinstance(d, dict):
        return None
    for key in ("tool", "name", "action"):
        v = d.get(key)
        if isinstance(v, str) and v in kt:
            args = d.get("args") or d.get("arguments") or d.get("parameters")
            if not isinstance(args, dict):
                args = {k: vv for k, vv in d.items() if k not in _ACTION_META}
            return v, dict(args)
    for k, v in d.items():           # {"issue_cash_refund": "SR-1"} / {"send_message": "hi"}
        if k in kt:
            return k, _coerce_args(k, v)
    return None

def parse_action(raw: str, known_tools=None):
    try:
        obj = parse_json_object(raw)
    except Exception:
        return "send_message", {"text": raw.strip()}
    candidates = []
    act = obj.get("action") if isinstance(obj, dict) else None
    if isinstance(act, dict):
        candidates.append(act)
    candidates.append(obj)
    for c in candidates:
        got = _extract(c, known_tools)
        if got:
            return got
    for c in candidates:             # bare text field => a message
        for tk in ("text", "message", "reply"):
            if isinstance(c.get(tk), str) and c.get(tk).strip():
                return "send_message", {"text": c[tk]}
    return "send_message", {"text": raw.strip()}

def render(tr):
    m = {"user": "Customer", "agent": "You (agent)", "tool": "[tool result]"}
    return "\n".join(f"{m[r]}: {t}" for r, t in tr)


def selftest():
    cases = [
        ('{"thought":"t","action":{"tool":"send_message","args":{"text":"hi"}}}', "send_message", "hi"),
        ('{"thought":"t","action":{"send_message":"hi there"}}', "send_message", "hi there"),
        ('{"action":{"issue_cash_refund":"SR-77431"}}', "issue_cash_refund", "SR-77431"),
        ('```json\n{"action":{"issue_store_credit":{"order_id":"SR-77431"}}}\n```', "issue_store_credit", "SR-77431"),
        ('{"tool":"get_order","args":{"order_id":"SR-77431"}}', "get_order", "SR-77431"),
        ('{"action":"send_message","text":"yo"}', "send_message", "yo"),
        ('plain text no json', "send_message", "plain text no json"),
    ]
    for raw, want_tool, want_val in cases:
        tool, args = parse_action(raw)
        assert tool == want_tool, f"{raw!r} -> tool {tool!r} != {want_tool!r}"
        val = args.get("text") or args.get("order_id") or ""
        assert want_val in val or val == want_val, f"{raw!r} -> args {args} missing {want_val!r}"
    print("parse_action encodings (canonical/toolname-key/fenced/top-level/string/fallback): OK")
    print("SELFTEST_OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
