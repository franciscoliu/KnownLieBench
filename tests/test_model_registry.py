import pytest

from knownliebench.backends.base import ModelConfig
from knownliebench.backends.load_local_hf import LocalHFModelClient
from knownliebench.backends.load_mock import MockModelClient
from knownliebench.backends.load_openai import OpenAIModelClient
from knownliebench.backends.load_openai_compat import OpenAICompatModelClient
from knownliebench.backends.load_scripted import ScriptedModelClient
from knownliebench.backends.registry import build_model_client, load_model_config


def test_mock_model_loads_from_config():
    config = load_model_config("configs/models.yaml", "mock")
    client = build_model_client(config)
    assert isinstance(client, MockModelClient)
    assert client.generate("hello").startswith("[mock:mock]")


def test_openai_refuses_without_run_real_api(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = load_model_config("configs/models.yaml", "openai_agent_gpt4o_mini")
    client = OpenAIModelClient(config)
    with pytest.raises(RuntimeError, match="run_real_api=True"):
        client.generate("hello")


def test_openai_compat_refuses_without_run_real_api(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "present_but_must_not_be_used")
    config = load_model_config("configs/models.yaml", "compat_agent")
    client = build_model_client(config)
    assert isinstance(client, OpenAICompatModelClient)
    with pytest.raises(RuntimeError, match="run_real_api=True"):
        client.generate("hello")


def test_local_hf_resolves_and_records_provenance():
    config = load_model_config("configs/models.yaml", "local_llama31_8b")
    client = build_model_client(config)
    assert isinstance(client, LocalHFModelClient)
    assert config.model == "meta-llama/Llama-3.1-8B-Instruct"   # provenance for run_meta
    assert client._local_files_only is False                     # hub id, not a local dir


def test_local_hf_without_torch_raises_helpful_error(monkeypatch):
    torch_missing = False
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_missing = True
    if not torch_missing:
        pytest.skip("torch installed; the [local] extra error path is not reachable")
    client = LocalHFModelClient(ModelConfig(name="x", provider="local_hf", model_path="not/a-real-model"))
    with pytest.raises(RuntimeError, match="local"):
        client.generate("hello")


def test_scripted_roles_resolve():
    for name, role in (("scripted_agent_deceptive", "agent_deceptive"),
                       ("scripted_agent_honest", "agent_honest"),
                       ("scripted_customer", "customer"),
                       ("scripted_grader", "grader")):
        client = build_model_client(load_model_config("configs/models.yaml", name))
        assert isinstance(client, ScriptedModelClient)
        assert client._role == role
    gate = build_model_client(load_model_config("configs/models.yaml", "scripted_agent_deceptive"))
    assert '"card_refund_available":true' in gate.generate("... internal knowledge check ...")


def test_usage_accumulator_is_additive_and_guarded():
    m = MockModelClient()
    assert m.usage == {"prompt": 0, "completion": 0, "calls": 0, "errors": 0}
    m._add_usage(100, 20)
    m._add_usage(50, 10)
    m._add_usage(None, None, ok=False)          # an errored call: counted, no tokens
    assert m.usage == {"prompt": 150, "completion": 30, "calls": 3, "errors": 1}
    m._add_usage("bad", "values")               # unparseable usage -> guarded, still counts the call
    assert m.usage["calls"] == 4 and m.usage["prompt"] == 150
