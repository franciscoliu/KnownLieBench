"""Deposit-env tools. get_policy + send_message are reused VERBATIM from the refund env (domain-neutral);
get_lease / return_deposit / withhold_deposit are deposit-specific. Terminal vs reply is decided by the env.
The episode outcome is read from final state: decision.type in {deposit_return, withhold} or 'none'."""
from knownliebench.envs.refund.tools.get_policy import GetPolicy
from knownliebench.envs.refund.tools.send_message import SendMessage
from .get_lease import GetLease
from .return_deposit import ReturnDeposit
from .withhold_deposit import WithholdDeposit

ALL_TOOLS = [GetLease, GetPolicy, SendMessage, ReturnDeposit, WithholdDeposit]

ALIASES = {
    "lookup_lease": "get_lease",
    "get_order": "get_lease",
    "lookup_order": "get_lease",
    "get_booking": "get_lease",
    "check_policy": "get_policy",
    "reply": "send_message",
    "respond": "send_message",
    "message": "send_message",
}

TERMINAL_TOOLS = {"return_deposit", "withhold_deposit"}
REPLY_TOOLS = {"send_message"}
