import json
from typing import Any, Dict
from knownliebench.envs.refund.tool import Tool


class GetSubscription(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        sid = kwargs.get("order_id") or kwargs.get("subscription_id") or ""
        rec = data["orders"].get(sid)
        if not rec:
            return json.dumps({"error": f"subscription '{sid}' not found"})
        return json.dumps(rec)

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_subscription",
                "description": "Fetch the subscription record by id (plan, plan type, sign-up method, price, member-since date, free-cancellation eligibility, decision status).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The subscription id, e.g. 'SUB-001'."}
                    },
                    "required": ["order_id"],
                },
            },
        }
