"""Billing-env tools. get_policy + send_message reused VERBATIM from the refund env (domain-neutral);
get_charge / open_billing_dispute / deny_billing_dispute are billing-specific. The episode outcome is read
from final state: decision.type in {dispute, deny} or 'none'."""
from knownliebench.envs.refund.tools.get_policy import GetPolicy
from knownliebench.envs.refund.tools.send_message import SendMessage
from .get_charge import GetCharge
from .open_billing_dispute import OpenBillingDispute
from .deny_billing_dispute import DenyBillingDispute

ALL_TOOLS = [GetCharge, GetPolicy, SendMessage, OpenBillingDispute, DenyBillingDispute]

ALIASES = {
    "lookup_charge": "get_charge",
    "get_order": "get_charge",
    "lookup_order": "get_charge",
    "get_transaction": "get_charge",
    "check_policy": "get_policy",
    "reply": "send_message",
    "respond": "send_message",
    "message": "send_message",
}

TERMINAL_TOOLS = {"open_billing_dispute", "deny_billing_dispute"}
REPLY_TOOLS = {"send_message"}
