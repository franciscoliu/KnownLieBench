import json
from typing import Any, Dict
from ..tool import Tool


class SendMessage(Tool):
    """Send a chat message to the customer. Does not mutate state; the env routes the
    text to the customer simulator (this is the 'respond' path)."""

    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        text = str(kwargs.get("text") or kwargs.get("message") or "").strip()
        return json.dumps({"sent": text})

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a chat message to the customer and await their reply.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The message to send to the customer."}
                    },
                    "required": ["text"],
                },
            },
        }
