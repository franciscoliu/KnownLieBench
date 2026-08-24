"""RefundEnv — a small tau-bench-style stateful environment.

Loads world state via data/load_data(), dispatches tool calls to the `Tool` classes
in tools/ALL_TOOLS (by their get_info() name), and reads the outcome from the final
state. Terminal vs reply is decided here (not by the tools): `send_message` is a reply;
`issue_*` are terminal.
"""
from __future__ import annotations
import copy
import json
from typing import Any, Dict, Tuple

from .data import load_data
from .tools import ALL_TOOLS, ALIASES, TERMINAL_TOOLS, REPLY_TOOLS


class RefundEnv:
    def __init__(self, scenario: str | None = None):
        self.scenario = scenario or "default"
        self._data0 = load_data(scenario)
        self.policy = self._data0["policy"]
        self.clauses = self.policy["clauses"]
        self.order_id = next(iter(self._data0["orders"]))
        self.registry = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
        self.tool_specs = "\n".join(
            "- {name}({params}): {desc}".format(
                name=s["function"]["name"],
                params=", ".join(s["function"]["parameters"]["properties"].keys()),
                desc=s["function"]["description"],
            )
            for s in (t.get_info() for t in ALL_TOOLS)
        )
        self.data: Dict[str, Any] = {}

    def policy_text(self) -> str:
        return self.policy["title"] + "\n" + "\n".join(f"- {c}" for c in self.clauses)

    # --- per-round SITUATION accessors (refund domain v2). Eval metadata only; NEVER exposed by
    # get_order. Absent (legacy scenarios) -> empty/defaults so the standard behavior is preserved. ---
    def situation(self, oid: str | None = None) -> Dict[str, Any]:
        sits = (self.data or self._data0).get("situations", {})
        return sits.get(oid or self.order_id, {})

    def reason(self, oid: str | None = None) -> str:
        """The customer's opening line for this situation (falls back to a generic return request)."""
        return self.situation(oid).get("reason", "")

    def entitled_outcome(self, oid: str | None = None) -> str:
        """Ground-truth outcome the customer is owed: 'cash' | 'store_credit' | 'none'."""
        return self.situation(oid).get("entitled_outcome", "cash")

    def policy_facts(self, oid: str | None = None) -> str:
        """Per-situation TRUE facts for the grader/gate. '' -> caller uses the default POLICY_FACTS."""
        return self.situation(oid).get("policy_facts", "")

    def tempting_lie(self, oid: str | None = None) -> str:
        return self.situation(oid).get("tempting_lie", "")

    def reset(self) -> Dict[str, Any]:
        self.data = copy.deepcopy(self._data0)
        return self.data

    def step(self, tool: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
        """Execute one tool call. Returns (result, done, kind) where kind in
        {"continue","reply","terminal"}."""
        name = ALIASES.get(tool, tool)
        ToolClass = self.registry.get(name)
        if ToolClass is None:
            return {"error": f"unknown tool '{tool}'"}, False, "continue"
        raw = ToolClass.invoke(self.data, **(args or {}))
        try:
            result = json.loads(raw)
        except Exception:
            result = {"raw": raw}
        if name in TERMINAL_TOOLS and "error" not in result:
            return result, True, "terminal"
        if name in REPLY_TOOLS:
            return {"text": result.get("sent", "")}, False, "reply"
        return result, False, "continue"

    def outcome(self) -> str:
        r = self.data["orders"][self.order_id]["refund"]
        return r["type"] if r else "none"
