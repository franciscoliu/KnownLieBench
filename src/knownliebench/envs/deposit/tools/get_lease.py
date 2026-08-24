import json
from typing import Any, Dict
from knownliebench.envs.refund.tool import Tool


class GetLease(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        lid = kwargs.get("order_id") or kwargs.get("lease_id") or ""
        rec = data["orders"].get(lid)
        if not rec:
            return json.dumps({"error": f"lease '{lid}' not found"})
        return json.dumps(rec)

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_lease",
                "description": "Fetch the lease / move-out record by id (unit, lease type, deposit amount, monthly rent, move-out date, deposit-return eligibility, deposit decision status).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The lease id, e.g. 'D-001'."}
                    },
                    "required": ["order_id"],
                },
            },
        }
