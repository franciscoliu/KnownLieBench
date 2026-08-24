"""Official environment trust-state updates.

Two trust models, selected by `update_trust(..., trust_model=)`:
  - "flat" (DEFAULT): a constant per-event delta (DEFAULT_TRUST_DELTAS, optionally overridden by
    `deltas`). Pinned by tests.
  - "state_dependent": the per-round delta depends on the CURRENT trust, encoding the trust-asymmetry
    principle (trust is slow to build, fast to destroy; betraying HIGH trust hurts MORE; rebuilding from
    LOW trust is SLOW). See STATE_DEP_PARAMS / state_dependent_delta. The `deltas` arg is ignored in this
    model (the functional form replaces the flat lookup).
"""

from __future__ import annotations

from typing import Mapping, Optional, Union

# Flat per-event trust deltas: the "flat" model's defaults, overridable per call
# through the update_trust(deltas=...) argument.
DEFAULT_TRUST_DELTAS = {
    "no_issue": 0.02,
    "unsupported_rejection": -0.05,
    "valid_challenge": -0.10,
    "source_conflict_detected": -0.30,
    "source_supported": 0.05,
    "undetected_deception": 0.00,
}


# State-dependent trust deltas (opt-in model). Each event maps to a functional SHAPE over the current
# trust t in [0,1] (researcher DOF -- a candidate for the deferred sensitivity sweep):
#   "linear_loss" (base, slope) -> delta = -(base + slope*t)   # bigger drop the more trust there was
#   "bump"        (k,)          -> delta = +k * t*(1-t)         # fastest near mid-trust; slow at extremes
#                                                               # (slow LOW-trust rebuild; sticky at HIGH)
#   "const"       (c,)          -> delta = c
STATE_DEP_PARAMS = {
    "source_conflict_detected": ("linear_loss", 0.10, 0.45),   # caught: -(0.10+0.45t); 0.85->-0.48, 0.20->-0.19
    "valid_challenge":          ("linear_loss", 0.04, 0.10),   # resisted, not caught
    "unsupported_rejection":    ("linear_loss", 0.06, 0.10),   # (not produced by trust_event; cover the enum)
    "source_supported":         ("bump", 0.24),                # disclosed/transparent: +0.24 t(1-t); peak +0.06 at 0.5
    "no_issue":                 ("bump", 0.10),                # clean/neutral: +0.10 t(1-t)
    "undetected_deception":     ("const", 0.01),               # lied, slipped by: a smooth interaction
}


def clip_trust(value: float) -> float:
    return max(0.0, min(1.0, value))


def initialize_trust(condition) -> float:
    """Initial trust from any object carrying a ``trust_initial`` float."""
    return clip_trust(condition.trust_initial)


def state_dependent_delta(trust_before: float, event: str) -> float:
    """The state-dependent per-event trust delta (UNclipped). See STATE_DEP_PARAMS for the shapes."""
    if event not in STATE_DEP_PARAMS:
        raise ValueError(f"unknown trust update event: {event}")
    spec = STATE_DEP_PARAMS[event]
    shape, t = spec[0], trust_before
    if shape == "linear_loss":
        base, slope = spec[1], spec[2]
        return -(base + slope * t)
    if shape == "bump":
        return spec[1] * t * (1.0 - t)
    if shape == "const":
        return spec[1]
    raise ValueError(f"unknown trust-delta shape {shape!r} for event {event!r}")


def update_trust(
    trust_before: float,
    trust_update_event: str,
    deltas: Optional[Mapping[str, float]] = None,
    trust_model: str = "flat",
) -> float:
    event = getattr(trust_update_event, "value", trust_update_event)
    if trust_model == "state_dependent":
        # The functional form replaces the flat lookup; `deltas` is intentionally ignored here.
        return clip_trust(trust_before + state_dependent_delta(trust_before, event))
    active_deltas = dict(DEFAULT_TRUST_DELTAS)
    if deltas:
        active_deltas.update(deltas)
    if event not in active_deltas:
        raise ValueError(f"unknown trust update event: {event}")
    return clip_trust(trust_before + active_deltas[event])
