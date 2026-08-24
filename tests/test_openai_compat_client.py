"""Robustness tests for OpenAICompatModelClient: transient backoff, defensive parsing, truncation retry.

urlopen is monkeypatched -> NO real API calls. Covers the failure modes that matter for hosted
reasoning models: empty/null content, 429/5xx retries, and the reasoning-truncation
(finish_reason=="length" + empty content) one-shot bump.
"""
import io
import json
import urllib.error

import pytest

import knownliebench.backends.load_openai_compat as pp
from knownliebench.backends.load_openai_compat import OpenAICompatModelClient
from knownliebench.backends.registry import load_model_config


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, body=b'{"error":"x"}'):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body))


def _seq_urlopen(items, captured):
    """Fake urlopen: yields each item in order; a dict -> response, an Exception -> raised."""
    it = iter(items)

    def fake(request, timeout=None):
        captured.append(json.loads(request.data.decode()))
        nxt = next(it)
        if isinstance(nxt, Exception):
            raise nxt
        return _Resp(nxt)

    return fake


def _msg(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7}}


def _client():
    cfg = load_model_config("configs/models.yaml", "compat_agent", run_real_api=True)
    return OpenAICompatModelClient(cfg)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://compat.test/v1/chat/completions")
    monkeypatch.setattr(pp.time, "sleep", lambda *_a, **_k: None)  # no real backoff sleeps


def test_happy_path_returns_content_and_counts_usage(monkeypatch):
    cap = []
    monkeypatch.setattr(pp.urllib.request, "urlopen", _seq_urlopen([_msg("hi there")], cap))
    c = _client()
    assert c.generate("q") == "hi there"
    assert c.usage["calls"] == 1 and c.usage["prompt"] == 5 and c.usage["completion"] == 7
    assert len(cap) == 1


def test_null_content_returns_empty_string(monkeypatch):
    cap = []
    monkeypatch.setattr(pp.urllib.request, "urlopen", _seq_urlopen([_msg(None)], cap))
    assert _client().generate("q") == ""


def test_missing_choices_returns_empty_string(monkeypatch):
    cap = []
    monkeypatch.setattr(pp.urllib.request, "urlopen", _seq_urlopen([{"usage": {}}], cap))
    assert _client().generate("q") == ""


def test_transient_503_then_success(monkeypatch):
    cap = []
    monkeypatch.setattr(pp.urllib.request, "urlopen",
                        _seq_urlopen([_http_error(503), _msg("recovered")], cap))
    assert _client().generate("q") == "recovered"
    assert len(cap) == 2  # retried once


def test_non_transient_400_raises_no_retry(monkeypatch):
    cap = []
    monkeypatch.setattr(pp.urllib.request, "urlopen",
                        _seq_urlopen([_http_error(400, b'{"error":"bad request"}')], cap))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        _client().generate("q")
    assert len(cap) == 1  # 400 is not retried


def test_truncation_retry_bumps_max_tokens(monkeypatch):
    cap = []
    monkeypatch.setattr(pp.urllib.request, "urlopen",
                        _seq_urlopen([_msg("", finish="length"), _msg("full answer")], cap))
    c = _client()
    assert c.generate("q", max_tokens=4000) == "full answer"
    assert len(cap) == 2
    assert cap[0]["max_tokens"] == 4000
    assert cap[1]["max_tokens"] == OpenAICompatModelClient.TRUNCATE_CAP
    assert c.usage["calls"] == 2  # both round-trips counted


def test_rate_limit_429_gets_bigger_retry_budget(monkeypatch):
    """429 (rate limit) retries up to MAX_429_RETRIES -- 5 in a row (past the 3 for 5xx) then succeeds."""
    cap = []
    seq = [_http_error(429)] * 5 + [_msg("through at last")]
    monkeypatch.setattr(pp.urllib.request, "urlopen", _seq_urlopen(seq, cap))
    c = _client()
    assert c.generate("q") == "through at last"
    assert len(cap) == 6  # 5 rate-limit retries + success; would have failed at MAX_RETRIES=3


def test_rate_limit_429_exhausts_then_raises(monkeypatch):
    """Sustained 429 past the budget raises (so the round errors and fail-fast can eventually trip)."""
    cap = []
    seq = [_http_error(429)] * (pp.OpenAICompatModelClient.MAX_429_RETRIES + 1)
    monkeypatch.setattr(pp.urllib.request, "urlopen", _seq_urlopen(seq, cap))
    with pytest.raises(RuntimeError, match="HTTP 429"):
        _client().generate("q")


def test_temperature_rejected_400_drops_and_retries(monkeypatch):
    """A model that rejects `temperature` (400) -> param dropped, call retried once, succeeds."""
    cap = []
    err = _http_error(400, b'{"error":{"message":"`temperature` is deprecated for this model."}}')
    monkeypatch.setattr(pp.urllib.request, "urlopen", _seq_urlopen([err, _msg("ok no temp")], cap))
    c = _client()
    assert c.generate("q", temperature=0.7) == "ok no temp"
    assert len(cap) == 2
    assert "temperature" in cap[0]          # first attempt sent temperature
    assert "temperature" not in cap[1]      # retry dropped it


def test_empty_content_finish_stop_does_not_retry(monkeypatch):
    """Empty answer that FINISHED normally (not truncated) is returned as '' -- no wasteful retry."""
    cap = []
    monkeypatch.setattr(pp.urllib.request, "urlopen", _seq_urlopen([_msg("", finish="stop")], cap))
    assert _client().generate("q") == ""
    assert len(cap) == 1
