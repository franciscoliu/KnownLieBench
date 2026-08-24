"""Model backends: one client class per provider, resolved through the registry."""

from .base import BaseModelClient, ModelConfig
from .registry import build_model_client, load_model_config

__all__ = ["BaseModelClient", "ModelConfig", "build_model_client", "load_model_config"]
