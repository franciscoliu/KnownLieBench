"""Tool abstract base class (matches tau-bench's tau_bench/envs/tool.py).

Each concrete tool subclasses Tool and implements two static methods:
  invoke(data, **kwargs) -> str   # reads/writes the env `data` dict, returns a JSON string
  get_info() -> dict              # the OpenAI function-calling schema
"""
from __future__ import annotations
import abc
from typing import Any


class Tool(abc.ABC):
    @staticmethod
    def invoke(*args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict:
        raise NotImplementedError
