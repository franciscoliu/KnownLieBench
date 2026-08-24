"""Shared model-client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    name: str
    provider: str
    model: Optional[str] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model_path: Optional[str] = None
    base_url: Optional[str] = None
    run_real_api: bool = False
    raw: Optional[Dict[str, Any]] = None


class BaseModelClient(ABC):
    """Unified interface used by runners and tests."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        # per-instance token accounting (additive, non-fatal). Runners read this after a run to
        # write a usage_summary. Every provider's generate() calls _add_usage exactly once per call.
        self.usage: Dict[str, int] = {"prompt": 0, "completion": 0, "calls": 0, "errors": 0}

    def _add_usage(self, prompt_tokens: Any = None, completion_tokens: Any = None, ok: bool = True) -> None:
        """Record ONE model call. Guarded so a missing/None usage never breaks a run."""
        self.usage["calls"] += 1
        if not ok:
            self.usage["errors"] += 1
            return
        try:
            self.usage["prompt"] += int(prompt_tokens or 0)
            self.usage["completion"] += int(completion_tokens or 0)
        except Exception:
            pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError
