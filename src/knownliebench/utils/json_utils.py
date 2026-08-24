"""JSON parsing helpers."""

from __future__ import annotations

import json
from typing import Any, Dict


def parse_json_object(text: str) -> Dict[str, Any]:
    text = _strip_json_fence(text.strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                raise ValueError(f"expected JSON object: {exc}") from exc
        else:
            raise ValueError(f"expected JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text
