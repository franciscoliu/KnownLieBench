"""OpenAI model backend (provider: openai).

Talks to the standard OpenAI Responses API with the key read from OPENAI_API_KEY. The
client is inert unless run_real_api=True, so dry-runs and tests never consume credits.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import BaseModelClient, ModelConfig


class OpenAIModelClient(BaseModelClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        if not self.config.run_real_api:
            raise RuntimeError("OpenAIModelClient refuses to call APIs unless run_real_api=True")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use OpenAIModelClient") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIModelClient")

        model = self.config.model or "gpt-4o-mini"
        temp = temperature if temperature is not None else self.config.temperature
        max_t = max_tokens if max_tokens is not None else self.config.max_tokens

        client = OpenAI(api_key=api_key)
        input_payload = prompt if system is None else f"{system}\n\n{prompt}"
        response = client.responses.create(
            model=model,
            input=input_payload,
            temperature=temp,
            max_output_tokens=max_t,
            **kwargs,
        )
        u = getattr(response, "usage", None)  # Responses API: input_tokens / output_tokens
        self._add_usage(getattr(u, "input_tokens", None), getattr(u, "output_tokens", None))
        return response.output_text
