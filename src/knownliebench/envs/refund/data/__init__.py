"""Load the refund env's world state from the JSON data files (tau-bench-style).

Returns a single `data` dict with keys: "orders", "users", "policy".

A SCENARIO is a {orders,users,policy}.json triple. The DEFAULT scenario lives in this folder
(byte-identical, pinned by tests). Named scenarios live in `scenarios/<name>/`. Same domain
(refund): scenarios vary the customer + item basket; the POLICY NUMBERS (10% bonus, ShopRight-only
store credit with no expiry, 5-10 day card refund) stay constant so the validated grader/tools remain correct.
"""
import json
import os
from typing import Any, Dict, List

FOLDER_PATH = os.path.dirname(__file__)
SCENARIOS_DIR = os.path.join(FOLDER_PATH, "scenarios")


def _scenario_dir(scenario: str | None) -> str:
    if not scenario or scenario == "default":
        return FOLDER_PATH
    return os.path.join(SCENARIOS_DIR, scenario)


def list_scenarios() -> List[str]:
    """All scenario names ('default' + any folder under scenarios/)."""
    names = ["default"]
    if os.path.isdir(SCENARIOS_DIR):
        names += sorted(d for d in os.listdir(SCENARIOS_DIR)
                        if os.path.isdir(os.path.join(SCENARIOS_DIR, d)))
    return names


def load_data(scenario: str | None = None) -> Dict[str, Any]:
    base = _scenario_dir(scenario)
    with open(os.path.join(base, "orders.json")) as f:
        orders = json.load(f)
    with open(os.path.join(base, "users.json")) as f:
        users = json.load(f)
    with open(os.path.join(base, "policy.json")) as f:
        policy = json.load(f)
    # Optional per-round SITUATIONS (refund domain v2): eval metadata keyed by order_id
    # (reason/entitled_outcome/policy_facts/tempting_lie). Absent in legacy scenarios -> {}.
    situations: Dict[str, Any] = {}
    sit_path = os.path.join(base, "situations.json")
    if os.path.isfile(sit_path):
        with open(sit_path) as f:
            situations = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    return {"orders": orders, "users": users, "policy": policy, "situations": situations}
