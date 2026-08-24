"""Refund-env tools (tau-bench-style: one class per file, exported via ALL_TOOLS).

Terminal vs reply is decided by the env, not the tool: `send_message` is the reply
path; `issue_cash_refund`/`issue_store_credit` are terminal. The episode outcome is
read from the final state, not from a tool return value.
"""
from .get_order import GetOrder
from .get_policy import GetPolicy
from .send_message import SendMessage
from .issue_cash_refund import IssueCashRefund
from .issue_store_credit import IssueStoreCredit

ALL_TOOLS = [GetOrder, GetPolicy, SendMessage, IssueCashRefund, IssueStoreCredit]

# tool-name aliases accepted from model output -> canonical get_info() name
ALIASES = {
    "lookup_order": "get_order",
    "check_policy": "get_policy",
    "reply": "send_message",
    "respond": "send_message",
    "message": "send_message",
}

TERMINAL_TOOLS = {"issue_cash_refund", "issue_store_credit"}
REPLY_TOOLS = {"send_message"}
