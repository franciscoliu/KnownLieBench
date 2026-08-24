"""Subscription-env tools. get_policy + send_message reused VERBATIM from the refund env (domain-neutral);
get_subscription / cancel_subscription / retain_subscription are subscription-specific. The episode outcome
is read from final state: decision.type in {cancel, retain} or 'none'."""
from knownliebench.envs.refund.tools.get_policy import GetPolicy
from knownliebench.envs.refund.tools.send_message import SendMessage
from .get_subscription import GetSubscription
from .cancel_subscription import CancelSubscription
from .retain_subscription import RetainSubscription

ALL_TOOLS = [GetSubscription, GetPolicy, SendMessage, CancelSubscription, RetainSubscription]

ALIASES = {
    "lookup_subscription": "get_subscription",
    "get_order": "get_subscription",
    "lookup_order": "get_subscription",
    "get_account": "get_subscription",
    "check_policy": "get_policy",
    "reply": "send_message",
    "respond": "send_message",
    "message": "send_message",
}

TERMINAL_TOOLS = {"cancel_subscription", "retain_subscription"}
REPLY_TOOLS = {"send_message"}
