"""DebtEnv -- third-party debt-collection domain (FDCPA 15 U.S.C. 1692 / Reg F 12 CFR 1006.26). Same shape as
RecallEnv / InsuranceEnv (tau-bench-style stateful env), with debt data + debt tools. Standalone. Outcome read
from final state: decision.type in {lawful_handling, coercive_collection} or 'none'."""
from __future__ import annotations
import copy
import json
from typing import Any, Dict, Tuple

from .data import load_data
from .tools import ALL_TOOLS, ALIASES, TERMINAL_TOOLS, REPLY_TOOLS


class DebtEnv:
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

    # --- per-round SITUATION accessors (eval metadata only; NEVER exposed by get_account) ---
    def situation(self, oid: str | None = None) -> Dict[str, Any]:
        sits = (self.data or self._data0).get("situations", {})
        return sits.get(oid or self.order_id, {})

    def reason(self, oid: str | None = None) -> str:
        return self.situation(oid).get("reason", "")

    def entitled_outcome(self, oid: str | None = None) -> str:
        return self.situation(oid).get("entitled_outcome", "lawful_handling")

    def policy_facts(self, oid: str | None = None) -> str:
        return self.situation(oid).get("policy_facts", "")

    def tempting_lie(self, oid: str | None = None) -> str:
        return self.situation(oid).get("tempting_lie", "")

    def reset(self) -> Dict[str, Any]:
        self.data = copy.deepcopy(self._data0)
        return self.data

    def step(self, tool: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
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
        d = self.data["orders"][self.order_id]["decision"]
        return d["type"] if d else "none"
