import json
from typing import Any, Dict
from ..tool import Tool


class GetPolicy(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).lower()
        clauses = data["policy"]["clauses"]
        words = [w for w in query.replace("?", " ").split() if len(w) > 3]
        hits = [c for c in clauses if any(w in c.lower() for w in words)]
        return json.dumps({"clauses": hits or clauses})

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_policy",
                "description": "Return the returns/refund policy clauses relevant to a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What you want the policy on, e.g. 'refund method'."}
                    },
                    "required": ["query"],
                },
            },
        }
