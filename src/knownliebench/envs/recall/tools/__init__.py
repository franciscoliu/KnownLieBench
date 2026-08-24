"""Recall-env tools. get_policy + send_message reused VERBATIM from the refund env (domain-neutral);
get_vehicle / provide_recall_remedy / deny_recall are recall-specific. The episode outcome is read from final
state: decision.type in {free_remedy, denied} or 'none'."""
from knownliebench.envs.refund.tools.get_policy import GetPolicy
from knownliebench.envs.refund.tools.send_message import SendMessage
from .get_vehicle import GetVehicle
from .provide_recall_remedy import ProvideRecallRemedy
from .deny_recall import DenyRecall

ALL_TOOLS = [GetVehicle, GetPolicy, SendMessage, ProvideRecallRemedy, DenyRecall]

ALIASES = {
    "lookup_vehicle": "get_vehicle",
    "get_order": "get_vehicle",
    "lookup_order": "get_vehicle",
    "get_recall": "get_vehicle",
    "check_recall": "get_vehicle",
    "check_policy": "get_policy",
    "reply": "send_message",
    "respond": "send_message",
    "message": "send_message",
}

TERMINAL_TOOLS = {"provide_recall_remedy", "deny_recall"}
REPLY_TOOLS = {"send_message"}
