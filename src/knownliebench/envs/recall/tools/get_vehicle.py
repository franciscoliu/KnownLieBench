import json
from typing import Any, Dict
from knownliebench.envs.refund.tool import Tool


class GetVehicle(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs: Any) -> str:
        vid = kwargs.get("order_id") or kwargs.get("vehicle_id") or kwargs.get("vin") or ""
        rec = data["orders"].get(vid)
        if not rec:
            return json.dumps({"error": f"vehicle '{vid}' not found"})
        return json.dumps(rec)

    @staticmethod
    def get_info() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_vehicle",
                "description": "Fetch the vehicle/recall record by id (vehicle description, model year, whether a free recall remedy is owed for this request, decision status).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The vehicle id, e.g. 'VEH-001'."}
                    },
                    "required": ["order_id"],
                },
            },
        }
