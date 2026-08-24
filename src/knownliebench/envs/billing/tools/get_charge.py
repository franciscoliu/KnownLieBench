import json
from typing import Any, Dict
from knownliebench.envs.refund.tool import Tool


class GetCharge(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        cid = kwargs.get("order_id") or kwargs.get("charge_id") or ""
        rec = data["orders"].get(cid)
        if not rec:
            return json.dumps({"error": f"charge '{cid}' not found"})
        return json.dumps(rec)

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_charge",
                "description": "Fetch the disputed-charge record by id (merchant, amount, charge date, statement date, card last 4, dispute eligibility, decision status).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The charge id, e.g. 'CHG-001'."}
                    },
                    "required": ["order_id"],
                },
            },
        }
