"""Generic OpenAI-compatible backend (provider: openai_compatible).

Points at ANY OpenAI-compatible chat-completions endpoint: a provider's native
compatibility endpoint, OpenRouter, a self-hosted LiteLLM gateway, or a local
``vllm serve``. The endpoint comes from the model config's ``base_url`` or the
OPENAI_COMPAT_BASE_URL env var; the key comes from OPENAI_COMPAT_API_KEY.

The client is deliberately inert unless run_real_api=True. This keeps dry-runs
and tests from consuming credits even when a key is present.

Robustness (hardened on large overnight batches, including reasoning models):
  * Transient HTTP (429/5xx) and network/timeout errors are retried with bounded
    exponential backoff -- protective for long overnight batches, not just fail-fast.
  * The response is parsed defensively: a missing `choices`/`message`/`content`
    yields "" instead of a KeyError/None that would crash the round.
  * Reasoning models can spend the whole token budget on hidden `reasoning_content`,
    returning an EMPTY answer with finish_reason == "length". That case is retried
    ONCE with a larger max_tokens so the round is not silently lost.
  * Some models reject the `temperature` param; on that 400 it is dropped and the call
    retried once.
`content` (the actual answer) is returned; `reasoning_content` (hidden chain-of-thought)
is never used as the reply.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional, Tuple

from .base import BaseModelClient, ModelConfig

_TRANSIENT_HTTP = {500, 502, 503, 504}      # 429 handled separately (bigger budget + Retry-After)


class OpenAICompatModelClient(BaseModelClient):
    MAX_RETRIES = 3            # 5xx / network retries, in addition to the first attempt
    MAX_429_RETRIES = 8        # rate-limit retries -- some hosted models throttle hard
    RATE_BACKOFF_CAP = 30      # seconds; 429 backoff grows to this (vs 8 for other transient)
    TRUNCATE_CAP = 8000        # max_tokens ceiling when retrying a reasoning-truncated response
    TIMEOUT_S = 300

    @staticmethod
    def _retry_after(exc: urllib.error.HTTPError) -> Optional[float]:
        """Honor a numeric Retry-After header (seconds) if the server sent one."""
        try:
            v = exc.headers.get("Retry-After") if exc.headers else None
            return float(v) if v is not None and str(v).strip().replace(".", "", 1).isdigit() else None
        except Exception:
            return None

    def _post(self, payload: dict, api_key: str) -> dict:
        """One HTTP round-trip with bounded backoff. 429 (rate limit) gets a larger retry budget with
        Retry-After support so heavily-throttled models still get through. Returns parsed JSON."""
        payload = dict(payload)  # local copy: we may drop unsupported params (e.g. temperature) and retry
        body = json.dumps(payload).encode("utf-8")
        url = self.config.base_url or os.getenv("OPENAI_COMPAT_BASE_URL")
        if not url:
            raise RuntimeError(
                "openai_compatible models need an endpoint: set base_url in the model config "
                "or export OPENAI_COMPAT_BASE_URL (a full /v1/chat/completions URL)"
            )
        dropped_temp = False
        tries_429 = tries_other = 0
        while True:
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.TIMEOUT_S) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body_txt = exc.read().decode("utf-8", errors="replace")
                # Some models reject `temperature` (e.g. claude-sonnet-5). Drop it and retry once.
                if (exc.code == 400 and "temperature" in body_txt.lower()
                        and "temperature" in payload and not dropped_temp):
                    payload.pop("temperature", None)
                    body = json.dumps(payload).encode("utf-8")
                    dropped_temp = True
                    continue
                if exc.code == 429 and tries_429 < self.MAX_429_RETRIES:
                    delay = self._retry_after(exc)
                    if delay is None:
                        delay = min(2 ** tries_429, self.RATE_BACKOFF_CAP)
                    tries_429 += 1
                    time.sleep(delay)
                    continue
                if exc.code in _TRANSIENT_HTTP and tries_other < self.MAX_RETRIES:
                    tries_other += 1
                    time.sleep(min(2 ** tries_other, 8))
                    continue
                raise RuntimeError(
                    f"API request failed with HTTP {exc.code}: {body_txt or exc.reason}. "
                    "Check that OPENAI_COMPAT_API_KEY and the endpoint URL are valid for this model."
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:  # network hiccup / read timeout
                if tries_other < self.MAX_RETRIES:
                    tries_other += 1
                    time.sleep(min(2 ** tries_other, 8))
                    continue
                raise RuntimeError(f"API request failed (network/timeout): {exc}") from exc

    @staticmethod
    def _parse(data: dict) -> Tuple[str, Optional[str], dict]:
        """(content, finish_reason, usage) -- defensive against missing keys / null content."""
        choices = data.get("choices") or []
        first = choices[0] if choices else {}
        msg = first.get("message") or {}
        return (msg.get("content") or ""), first.get("finish_reason"), (data.get("usage") or {})

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        if not self.config.run_real_api:
            raise RuntimeError("OpenAICompatModelClient refuses to call APIs unless run_real_api=True")
        api_key = os.getenv("OPENAI_COMPAT_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_COMPAT_API_KEY is required for OpenAICompatModelClient")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        max_t = max_tokens if max_tokens is not None else self.config.max_tokens
        payload = {
            "model": self.config.model or self.config.model_id,
            "stream": False,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_t,
        }
        payload.update(kwargs)

        content, finish, usage = self._parse(self._post(payload, api_key))
        self._add_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        # Reasoning model burned the whole budget on hidden reasoning_content -> empty answer, finish=="length".
        # Retry once with a larger cap so the round is not lost.
        if (not content) and finish == "length" and (max_t or 0) < self.TRUNCATE_CAP:
            payload["max_tokens"] = self.TRUNCATE_CAP
            content, finish, usage = self._parse(self._post(payload, api_key))
            self._add_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        return content or ""
