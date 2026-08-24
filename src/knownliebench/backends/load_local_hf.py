"""Local Hugging Face backend (provider: local_hf): run an open-weight agent on your GPU.

`model_path` accepts either a Hugging Face hub id (e.g. ``meta-llama/Llama-3.1-8B-Instruct``,
downloaded to your HF cache on first use) or a local snapshot directory. The model and
tokenizer are loaded once, lazily, and cached on the instance. ``generate()`` returns only
the newly generated assistant text, matching the API backends, so the runner, customer
simulator, and judge are unchanged.

Config (configs/models.yaml, section ``local_hf_models``):
    local_llama31_8b:
      provider: local_hf
      model_path: meta-llama/Llama-3.1-8B-Instruct
      temperature: 0.7
      max_tokens: 2048
      # optional: dtype (bfloat16), device_map (auto), enable_thinking (false),
      #           trust_remote_code (false), local_files_only (auto)

``local_files_only`` defaults to True when model_path is an existing local directory and
False for hub ids. ``trust_remote_code`` defaults to False; enable it only for model
families whose code you trust. Requires the ``[local]`` extra: pip install -e ".[local]".
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import BaseModelClient, ModelConfig


class LocalHFModelClient(BaseModelClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._path = config.model_path or config.model_id
        if not self._path:
            raise ValueError("local_hf models require model_path (a local HF snapshot dir) or model_id")
        raw = config.raw or {}
        self._dtype = str(raw.get("dtype", "bfloat16"))
        self._device_map = raw.get("device_map", "auto")
        self._enable_thinking = bool(raw.get("enable_thinking", False))
        self._trust_remote_code = bool(raw.get("trust_remote_code", False))
        lfo = raw.get("local_files_only")
        self._local_files_only = bool(lfo) if lfo is not None else os.path.isdir(str(self._path))
        if config.model is None:
            config.model = str(self._path)   # checkpoint provenance for run_meta / usage_summary
        self._model = None
        self._tok = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "local_hf models need the [local] extra: pip install -e \".[local]\""
            ) from exc

        dtype = getattr(torch, self._dtype, torch.bfloat16)
        self._tok = AutoTokenizer.from_pretrained(
            self._path, local_files_only=self._local_files_only, trust_remote_code=self._trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._path,
            dtype=dtype,
            device_map=self._device_map,
            local_files_only=self._local_files_only,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.eval()

    def _input_device(self):
        """Where to put input ids. For a single-GPU load or accelerate-sharded load, layer 0 (the
        embeddings) lives on cuda:0; sending inputs there is correct in both cases."""
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        self._load()
        import torch

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Render the model's own chat template; disable "thinking" for clean agentic output unless asked.
        try:
            text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self._enable_thinking
            )
        except TypeError:  # template doesn't accept enable_thinking
            text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        enc = self._tok(text, return_tensors="pt")
        dev = self._input_device()
        enc = {k: v.to(dev) for k, v in enc.items()}
        n_prompt = int(enc["input_ids"].shape[1])

        temp = temperature if temperature is not None else (self.config.temperature or 0.0)
        max_new = int(max_tokens or self.config.max_tokens or 1024)
        gen_kwargs = dict(
            max_new_tokens=max_new,
            pad_token_id=(self._tok.pad_token_id if self._tok.pad_token_id is not None else self._tok.eos_token_id),
        )
        if temp and float(temp) > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temp), top_p=float(kwargs.get("top_p", 0.95)))
        else:
            gen_kwargs.update(do_sample=False)

        try:
            with torch.no_grad():
                out = self._model.generate(**enc, **gen_kwargs)
            new_ids = out[0][n_prompt:]
            answer = self._tok.decode(new_ids, skip_special_tokens=True)
            self._add_usage(n_prompt, int(new_ids.shape[0]))
        except Exception:
            self._add_usage(n_prompt, 0, ok=False)
            raise

        # Reasoning models may still emit a <think>...</think> block; keep only the final answer.
        if "</think>" in answer:
            answer = answer.rsplit("</think>", 1)[-1]
        return answer.strip()
