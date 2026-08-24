"""Insurance-env tools. get_policy + send_message reused VERBATIM from the refund env (domain-neutral);
get_claim / file_claim / refuse_claim are insurance-specific. The episode outcome is read from final state:
decision.type in {filed, refused} or 'none'."""
from knownliebench.envs.refund.tools.get_policy import GetPolicy
from knownliebench.envs.refund.tools.send_message import SendMessage
from .get_claim import GetClaim
from .file_claim import FileClaim
from .refuse_claim import RefuseClaim

ALL_TOOLS = [GetClaim, GetPolicy, SendMessage, FileClaim, RefuseClaim]

ALIASES = {
    "lookup_claim": "get_claim",
    "get_order": "get_claim",
    "lookup_order": "get_claim",
    "get_appeal": "get_claim",
    "check_policy": "get_policy",
    "reply": "send_message",
    "respond": "send_message",
    "message": "send_message",
}

TERMINAL_TOOLS = {"file_claim", "refuse_claim"}
REPLY_TOOLS = {"send_message"}
