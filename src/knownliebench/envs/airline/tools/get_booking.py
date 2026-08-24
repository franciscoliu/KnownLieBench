import json
from typing import Any, Dict
from knownliebench.envs.refund.tool import Tool


class GetBooking(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        bid = kwargs.get("order_id") or kwargs.get("booking_id") or ""
        b = data["orders"].get(bid)
        if not b:
            return json.dumps({"error": f"booking '{bid}' not found"})
        return json.dumps(b)

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_booking",
                "description": "Fetch the booking record by id (flight, fare, payment method, fare type, refund eligibility, refund/voucher status).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The booking id, e.g. 'SK-48210'."}
                    },
                    "required": ["order_id"],
                },
            },
        }
