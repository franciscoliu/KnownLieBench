"""Airline-env tools. get_policy + send_message are reused VERBATIM from the refund env (domain-neutral);
get_booking / issue_cash_refund / issue_voucher are airline-specific. Terminal vs reply is decided by the
env. The episode outcome is read from final state: refund.type in {refund_original, voucher} or 'none'."""
from knownliebench.envs.refund.tools.get_policy import GetPolicy
from knownliebench.envs.refund.tools.send_message import SendMessage
from .get_booking import GetBooking
from .issue_cash_refund import IssueCashRefund
from .issue_voucher import IssueVoucher

ALL_TOOLS = [GetBooking, GetPolicy, SendMessage, IssueCashRefund, IssueVoucher]

ALIASES = {
    "lookup_booking": "get_booking",
    "get_order": "get_booking",
    "lookup_order": "get_booking",
    "check_policy": "get_policy",
    "reply": "send_message",
    "respond": "send_message",
    "message": "send_message",
}

TERMINAL_TOOLS = {"issue_cash_refund", "issue_voucher"}
REPLY_TOOLS = {"send_message"}
