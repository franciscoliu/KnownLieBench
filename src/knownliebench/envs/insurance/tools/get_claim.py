import json
from typing import Any, Dict
from knownliebench.envs.refund.tool import Tool


class GetClaim(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        cid = kwargs.get("order_id") or kwargs.get("claim_id") or ""
        rec = data["orders"].get(cid)
        if not rec:
            return json.dumps({"error": f"claim '{cid}' not found"})
        return json.dumps(rec)

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_claim",
                "description": "Fetch the claim/appeal record by id (service description, claim type, denial date if any, amount, whether the claim/appeal is fileable, decision status).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The claim id, e.g. 'CLM-001'."}
                    },
                    "required": ["order_id"],
                },
            },
        }
