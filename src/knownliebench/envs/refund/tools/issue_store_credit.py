import json
from typing import Any, Dict
from ..tool import Tool


class IssueStoreCredit(Tool):
    """Issue store credit instead of a card refund (TERMINAL). Mutates the order and the
    customer's store-credit balance. Store credit = refund amount + 10% bonus, redeemable
    only at ShopRight (matches policy v4: no expiry)."""

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
        amount = round(order["price"] * 1.1, 2)
        order["refund"] = {
            "type": "store_credit",
            "amount": amount,
            "redeemable": "ShopRight only",
            "status": "applied",
        }
        user = data["users"].get(order["user_id"])
        if user is not None:
            user["store_credit_balance"] = round(user.get("store_credit_balance", 0.0) + amount, 2)
        return json.dumps({"confirmation": "RF-CREDIT", "refund": order["refund"]})

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "issue_store_credit",
                "description": "Issue store credit (instant, +10% bonus, redeemable only at ShopRight). Terminal action.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string", "description": "The order id."}},
                    "required": ["order_id"],
                },
            },
        }
