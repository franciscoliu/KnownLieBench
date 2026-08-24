"""JSON and JSONL file I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                if line_number == 1:
                    return _read_jsonl_fallback(path, text, exc)
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def _read_jsonl_fallback(
    path: Path, text: str, original_error: json.JSONDecodeError
) -> List[Dict[str, Any]]:
    """Accept a single JSON object/list when legacy data was pretty-printed."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        rows = _read_concatenated_json_objects(text)
        if rows:
            return rows
        raise ValueError(f"{path}:1: invalid JSON: {original_error}") from exc
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return list(payload)
    raise ValueError(f"{path} must contain JSONL rows or a JSON object/list of objects")


def _read_concatenated_json_objects(text: str) -> List[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    rows: List[Dict[str, Any]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            payload, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        rows.append(payload)
        index = end
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
