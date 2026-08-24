"""Tests for the tau-bench-style refund environment."""
from knownliebench.envs.refund.env import RefundEnv
from knownliebench.envs.refund.tools import ALL_TOOLS
from knownliebench.envs.refund.data import load_data


def test_load_data_keys():
    data = load_data()
    assert {"orders", "users", "policy"} <= set(data)  # + optional "situations" (refund domain v2)
    assert "SR-77431" in data["orders"]
    assert data["orders"]["SR-77431"]["user_id"] == "C-001"
    assert "C-001" in data["users"]


def test_all_tools_have_valid_schema():
    names = set()
    for tool in ALL_TOOLS:
        info = tool.get_info()
        assert info["type"] == "function"
        fn = info["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
        names.add(fn["name"])
    assert names == {"get_order", "get_policy", "send_message", "issue_cash_refund", "issue_store_credit"}


def test_get_order_respects_id():
    env = RefundEnv(); env.reset()
    res, done, kind = env.step("get_order", {"order_id": "NOPE"})
    assert res.get("error") and not done
    res, done, kind = env.step("get_order", {"order_id": env.order_id})
    assert res["order_id"] == env.order_id and kind == "continue"


def test_issue_store_credit_mutates_state_and_balance():
    env = RefundEnv(); env.reset()
    res, done, kind = env.step("issue_store_credit", {"order_id": env.order_id})
    assert done and kind == "terminal"
    assert env.outcome() == "store_credit"
    assert env.data["orders"][env.order_id]["refund"]["type"] == "store_credit"
    assert env.data["users"]["C-001"]["store_credit_balance"] == round(59.99 * 1.1, 2)


def test_double_issue_blocked_not_terminal():
    env = RefundEnv(); env.reset()
    env.step("issue_cash_refund", {"order_id": env.order_id})
    res, done, kind = env.step("issue_store_credit", {"order_id": env.order_id})
    assert res.get("error") and not done           # second issue rejected, episode not re-ended
    assert env.outcome() == "cash"                  # original outcome preserved


def test_reset_isolation():
    env = RefundEnv()
    env.reset(); env.step("issue_store_credit", {"order_id": env.order_id})
    assert env.outcome() == "store_credit"
    env.reset()                                     # fresh state
    assert env.outcome() == "none"
    assert env.data["users"]["C-001"]["store_credit_balance"] == 0.0


def test_aliases_and_reply():
    env = RefundEnv(); env.reset()
    res, done, kind = env.step("lookup_order", {"order_id": env.order_id})  # alias -> get_order
    assert res["order_id"] == env.order_id
    res, done, kind = env.step("reply", {"text": "hello"})                  # alias -> send_message
    assert kind == "reply" and res["text"] == "hello"
