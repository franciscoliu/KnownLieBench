"""Mock model backend for dry-runs and tests."""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseModelClient, ModelConfig


class MockModelClient(BaseModelClient):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config or ModelConfig(name="mock", provider="mock", model="mock"))

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        return f"[mock:{self.config.name}] {prompt[:120]}"
