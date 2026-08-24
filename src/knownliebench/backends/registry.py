"""Model registry and config selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .base import BaseModelClient, ModelConfig
from .load_decompose import DecomposeGraderClient
from .load_ensemble import EnsembleGraderClient
from .load_local_hf import LocalHFModelClient
from .load_mock import MockModelClient
from .load_openai import OpenAIModelClient
from .load_openai_compat import OpenAICompatModelClient
from .load_scripted import ScriptedModelClient
from .load_verify import VerifyGraderClient


PROVIDERS = {
    "mock": MockModelClient,
    "scripted": ScriptedModelClient,
    "openai": OpenAIModelClient,
    "openai_compatible": OpenAICompatModelClient,
    "local_hf": LocalHFModelClient,
    "ensemble": EnsembleGraderClient,   # self-consistency wrapper over base_grader (config.raw)
    "decompose": DecomposeGraderClient,  # per-claim isolation wrapper over base_grader (config.raw)
    "verify": VerifyGraderClient,        # grade-then-verify precision booster over base_grader (config.raw)
}


def _to_model_config(name: str, raw: Dict[str, Any], run_real_api: bool = False) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider=str(raw.get("provider", "mock")),
        model=raw.get("model"),
        model_id=raw.get("model_id"),
        temperature=raw.get("temperature"),
        max_tokens=raw.get("max_tokens"),
        model_path=raw.get("model_path"),
        base_url=raw.get("base_url"),
        run_real_api=bool(raw.get("run_real_api", run_real_api)),
        raw=dict(raw),
    )


def load_model_config(
    config_path: Path | str,
    model_name: str,
    section: Optional[str] = None,
    run_real_api: bool = False,
) -> ModelConfig:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sections = [section] if section else [
        "sender_models",
        "environment_models",
        "mock_models",
        "local_hf_models",
    ]
    available = []
    for section_name in sections:
        entries = payload.get(section_name, {})
        if model_name in entries:
            return _to_model_config(model_name, entries[model_name], run_real_api=run_real_api)
        available.extend(entries)
    raise KeyError(f"model config not found: {model_name!r}; available: {sorted(available)}")


def build_model_client(
    model_config: ModelConfig | Dict[str, Any],
    run_real_api: Optional[bool] = None,
) -> BaseModelClient:
    if isinstance(model_config, dict):
        name = str(model_config.get("name", model_config.get("model", "anonymous")))
        config = _to_model_config(name, model_config, run_real_api=bool(run_real_api))
    else:
        config = model_config
        if run_real_api is not None:
            config.run_real_api = run_real_api
    provider = config.provider
    if provider not in PROVIDERS:
        raise ValueError(f"unknown model provider: {provider}")
    return PROVIDERS[provider](config)
