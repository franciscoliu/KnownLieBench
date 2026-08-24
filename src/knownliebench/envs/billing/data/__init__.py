"""Load the billing env's world state from the JSON data files (same loader as refund/deposit/subscription).

Returns a `data` dict with keys: "orders" (disputed-charge records), "users" (cardholders), "policy",
"situations". Policy grounded in Reg Z 12 CFR 1026.13 / FCBA 15 USC 1666 (see policy.json:source).
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
    situations: Dict[str, Any] = {}
    sit_path = os.path.join(base, "situations.json")
    if os.path.isfile(sit_path):
        with open(sit_path) as f:
            situations = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    return {"orders": orders, "users": users, "policy": policy, "situations": situations}
