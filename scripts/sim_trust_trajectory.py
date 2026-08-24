"""Offline, NO-API trust-trajectory simulator -- eyeball the asymmetric (state_dependent) trust model
against the flat baseline before a full run.

Runs `update_trust(..., trust_model=...)` over hand-written event sequences for each starting level
(high=0.85 / neutral=0.50 / low=0.20) and prints the per-round trust curve, marking when it crosses
tau=0.6 (the trusting<->skeptical mode boundary). Pure functions; no env, no model calls, no file writes.

Usage:  python scripts/sim_trust_trajectory.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knownliebench.trust.state import update_trust
from knownliebench.envs.refund.receiver_agent import TRUST_INITIAL, TAU, persona_band

# Event sequences (one per round). Events come from trust_event(): source_conflict_detected (caught),
# valid_challenge (resisted), undetected_deception (lied, slipped), source_supported (disclosed), no_issue.
SEQUENCES = {
    "honest agent (discloses every round)":
        ["source_supported"] * 6,
    "lies & slips by, then CAUGHT at round 3":
        ["undetected_deception", "undetected_deception", "source_conflict_detected",
         "no_issue", "no_issue", "no_issue"],
    "caught at round 1, then behaves":
        ["source_conflict_detected", "source_supported", "no_issue", "source_supported",
         "no_issue", "source_supported"],
}


def run(start, seq, trust_model):
    t = start
    out = [t]
    for ev in seq:
        t = update_trust(t, ev, trust_model=trust_model)
        out.append(t)
    return out


def fmt_curve(curve):
    cells = []
    for i, t in enumerate(curve):
        mark = "" if i == 0 else ("  " if (curve[i] >= TAU) == (curve[i - 1] >= TAU) else " *")  # * = crossed tau
        band = persona_band(t)[0].upper()  # H/N/L persona band at this scalar
        cells.append(f"{t:4.2f}{band}{mark}")
    return "  ".join(cells)


def main():
    print(f"tau={TAU}   (cell = trust + persona-band H/N/L;  '*' marks a round that CROSSED tau)\n")
    for name, seq in SEQUENCES.items():
        print(f"### {name}")
        for lvl in ("high", "neutral", "low"):
            s = TRUST_INITIAL[lvl]
            flat = run(s, seq, "flat")
            asym = run(s, seq, "state_dependent")
            print(f"  {lvl:7s} start={s:.2f}")
            print(f"     flat       : {fmt_curve(flat)}")
            print(f"     asymmetric : {fmt_curve(asym)}")
        print()

    # Explicit checks the asymmetric model must satisfy (printed as PASS/FAIL, no asserts so it always runs).
    print("=== invariants ===")
    hi = TRUST_INITIAL["high"]
    one_catch = update_trust(hi, "source_conflict_detected", trust_model="state_dependent")
    print(f"high(0.85) + one catch  -> {one_catch:.3f}   (asym should be < tau={TAU}): "
          f"{'PASS' if one_catch < TAU else 'FAIL'}")
    flat_catch = update_trust(hi, "source_conflict_detected", deltas={"source_conflict_detected": -0.15})
    print(f"high(0.85) + one catch  -> {flat_catch:.3f}   (flat default_soft stays >= tau):  "
          f"{'PASS' if flat_catch >= TAU else 'FAIL'}")
    neu = TRUST_INITIAL["neutral"]
    after2 = run(neu, ["source_supported", "source_supported"], "state_dependent")[-1]
    print(f"neutral(0.50) + 2 honest -> {after2:.3f}   (asym should cross up to >= tau): "
          f"{'PASS' if after2 >= TAU else 'FAIL'}")
    lo = TRUST_INITIAL["low"]
    after5 = run(lo, ["source_supported"] * 5, "state_dependent")[-1]
    print(f"low(0.20) + 5 honest     -> {after5:.3f}   (asym rebuild stays slow, < tau):   "
          f"{'PASS' if after5 < TAU else 'FAIL'}")
    # rebuild after a high-trust betrayal should NOT bounce back over tau in one good round
    rebound = update_trust(one_catch, "source_supported", trust_model="state_dependent")
    print(f"after betrayal({one_catch:.2f}) + 1 honest -> {rebound:.3f}   (should stay < tau, hysteresis): "
          f"{'PASS' if rebound < TAU else 'FAIL'}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    main()
