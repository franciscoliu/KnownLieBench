import json
from typing import Any, Dict
from ..tool import Tool


class IssueCashRefund(Tool):
    """Refund to the customer's original payment method (TERMINAL). Mutates the order."""

    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        order_id = kwargs.get("order_id", "")
        order = data["orders"].get(order_id)
        if not order:
            return json.dumps({"error": f"order '{order_id}' not found"})
        if order.get("refund") is not None:
            return json.dumps({"error": "a refund has already been issued", "refund": order["refund"]})
        if not order.get("eligible_for_return", False):
            return json.dumps({"error": "order is not eligible for return"})
        order["refund"] = {
            "type": "cash",
            "amount": order["price"],
            "to": order["payment_method"],
            "eta_business_days": "5-10",
            "status": "processing",
        }
        return json.dumps({"confirmation": "RF-CASH", "refund": order["refund"]})

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "issue_cash_refund",
                "description": "Issue a refund to the customer's original payment method. Posts in 5-10 business days. Terminal action.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string", "description": "The order id."}},
                    "required": ["order_id"],
                },
            },
        }
