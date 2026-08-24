import json
from typing import Any, Dict
from ..tool import Tool


class GetOrder(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        order_id = kwargs.get("order_id", "")
        order = data["orders"].get(order_id)
        if not order:
            return json.dumps({"error": f"order '{order_id}' not found"})
        return json.dumps(order)

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Fetch the record for an order by its order_id (item, price, eligibility, payment method, refund status).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order id, e.g. 'SR-77431'."}
                    },
                    "required": ["order_id"],
                },
            },
        }
