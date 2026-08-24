"""Domain registry for the multi-round dialogue benchmark.

A `DomainSpec` collects exactly the domain-coupled seams (outcome vocabulary + the env/prompt/grader/gate/
receiver callables). The dialogue runner + scorer resolve a spec via `get_domain(name)` and route every
domain-specific call through it, so adding a domain is additive: the REFUND spec binds to the existing
`envs.refund.*` functions, while new domains supply their own spec (usually via the generic
builders in `domains._prompts`).

Deception type is EMERGENT (the sender's strategy), never a label here. The only domain-level outcome
vocabulary a spec declares is the OWED vs COMPANY-FAVORED outcome; harm/DSR is anchored to `owed`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class OutcomeSpec:
    """The outcome the receiver is OWED vs the company-FAVORED alternative (+ human labels for the grader)."""
    owed: str            # ground-truth owed outcome string written by the terminal tool (e.g. "cash")
    favored: str         # company-favored alternative (e.g. "store_credit")
    owed_label: str      # grader phrasing for the owed outcome
    favored_label: str   # grader phrasing for the favored outcome
    none_label: str      # grader phrasing for the calibration/ineligible case


@dataclass(frozen=True)
class DomainSpec:
    name: str
    institution: str
    agent_role: str
    receiver_role: str
    outcomes: OutcomeSpec
    repeated_interaction: bool          # True -> longitudinal + cross-sectional; False -> cross-sectional only
    order_tools: Tuple[str, ...]        # tools that take an order/record id (runner auto-fills the id)
    terminal_tools: Tuple[str, ...]     # the issue/finalize tools (used to build the "finalize: emit X or Y" hint)
    # callables -- signatures match the existing refund functions so the runner calls them uniformly:
    env_factory: Callable               # (scenario) -> env
    agent_system: Callable              # (env, condition) -> str
    gate_prompt: Callable               # (env, order_id) -> str
    gate_passed: Callable               # (raw, expected) -> bool
    grader_prompt: Callable             # (tr_str, outcome, policy_view, policy_facts, lie_examples, entitled_outcome) -> str
    derive: Callable                    # (ev) -> dict   (shared refund.derive is domain-neutral)
    receiver_system: Callable           # (trust_level, has_tools, verify_line, item, memory_block, insist_card, trust_scalar) -> str
    opening_for: Callable               # (item, reason) -> str


REGISTRY: dict[str, DomainSpec] = {}


def register(spec: DomainSpec) -> DomainSpec:
    REGISTRY[spec.name] = spec
    return spec


def _load_builtins() -> None:
    """Lazily import the built-in domain modules so they register their specs (avoids import-time cycles)."""
    from . import refund  # noqa: F401  (registers REFUND_SPEC)
    for mod in ("airline", "deposit", "subscription", "billing", "insurance", "recall", "debt"):
        try:
            __import__(f"knownliebench.domains.{mod}")
        except Exception:
            pass  # a domain not yet authored is fine


def get_domain(name: str) -> DomainSpec:
    if name not in REGISTRY:
        _load_builtins()
    if name not in REGISTRY:
        raise ValueError(f"unknown domain {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


# ---- display / sector names ---------------------------------------------------------------------
# The internal slug (env folders, file names, the runner, trajectory labels, tests) stays STABLE.
# These are the intuitive sector labels used wherever a domain is SHOWN -- figures, tables, the paper.
# Single source of truth; reporting code calls get_display_name(slug). Edit a value here to relabel.
DISPLAY_NAMES = {
    "refund": "retail",              # consumer retail returns / refunds
    "airline": "travel",             # airline refunds (DOT 14 CFR 260)
    "deposit": "housing",            # rental security deposits (CA Civ 1950.5)
    "subscription": "subscriptions", # auto-renewal cancellation (CA ARL / ROSCA)
    "billing": "banking",            # credit-card billing disputes (FCBA / Reg Z)
    "insurance": "healthcare",       # health-plan claims/appeals (ERISA) -- the healthcare-sector domain
    "recall": "automotive",          # vehicle safety recalls (49 U.S.C. 30120)
    "debt": "debt collection",       # FDCPA third-party debt collection (the "legal threats" / archetype-B domain)
}


def get_display_name(name: str) -> str:
    """Intuitive sector label for a domain slug (falls back to the slug itself if unmapped)."""
    return DISPLAY_NAMES.get(name, name)
