"""Re-grade saved trajectories for ANY domain under a different grader config.

Uses the domain's own spec.grader_prompt + env_factory, re-renders the saved transcript with
the same "You (agent):"/"Customer:" labels the grader's attribution rule references, re-runs
the grader, recomputes derive(), and reports old->new lie counts plus, per row, whether the
flagged quote appears in an AGENT line vs only a CUSTOMER line (a false-positive detector).
Writes <infile>_regraded.jsonl next to the input.

Usage:  python scripts/regrade_domain.py <domain> <trajectories.jsonl> [grader_cfg]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0].strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        os.environ.setdefault(k.strip(), v)

from knownliebench.domains import get_domain
from knownliebench.envs.refund.grader import derive
from knownliebench.utils.json_utils import parse_json_object
from knownliebench.backends.registry import load_model_config, build_model_client

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("domain", help="refund | airline | deposit | recall | subscription | billing | insurance | debt")
_ap.add_argument("infile", help="a trajectories_*.jsonl file written by run_benchmark.py")
_ap.add_argument("grader_cfg", nargs="?", default="openai_grader_gpt51")
_args = _ap.parse_args()
DOMAIN, INFILE, GRADER_CFG = _args.domain, _args.infile, _args.grader_cfg

_REQUIRED_ENV = {"openai": ("OPENAI_API_KEY",), "openai_compatible": ("OPENAI_COMPAT_API_KEY",)}


def render(tr):
    m = {"user": "Customer", "agent": "You (agent)", "tool": "[tool result]"}
    return "\n".join(f"{m.get(r, r)}: {t}" for r, t in tr)


def _role_text(tr, role):
    return "\n".join(str(t) for r, t in tr if r == role).lower()


def main():
    spec = get_domain(DOMAIN)
    env = spec.env_factory(); env.reset()
    _gcfg = load_model_config(ROOT / "configs" / "models.yaml", GRADER_CFG, run_real_api=True)
    for _var in _REQUIRED_ENV.get(_gcfg.provider, ()):
        if not os.getenv(_var):
            raise SystemExit(f"grader config {GRADER_CFG!r} needs {_var} set (see .env.example)")
    g = build_model_client(_gcfg)
    rows = [json.loads(l) for l in open(INFILE) if l.strip() and "run_meta" not in l]
    sc = [r for r in rows if r.get("gate_passed")]
    old_lie = sum(1 for r in sc if r["derived"].get("lied"))
    new = []
    for r in sc:
        oid = r["order_id"]
        try:
            ev = parse_json_object(g.generate(
                spec.grader_prompt(render(r["transcript"]), r.get("outcome", "none"), r.get("policy_view", ""),
                                   policy_facts=env.policy_facts(oid) or None,
                                   lie_examples=env.tempting_lie(oid), entitled_outcome=env.entitled_outcome(oid)),
                system="Return only JSON.", temperature=0.0))
        except Exception as e:
            print("  grade err:", str(e)[:120])
            new.append({**r, "regrade_error": str(e)[:200]})
            continue
        new.append({**r, "facts": ev, "derived": derive(ev)})
    n_err = sum(1 for r in new if r.get("regrade_error"))
    new_lie = sum(1 for r in new if not r.get("regrade_error") and r["derived"].get("lied"))
    cust_only = 0
    for r in new:
        if r.get("regrade_error") or not r["derived"].get("lied"):
            continue
        q = (r["facts"].get("false_claim_quote") or "").strip().lower()[:45]
        if q and q not in _role_text(r["transcript"], "agent") and q in _role_text(r["transcript"], "user"):
            cust_only += 1
    print(f"=== RE-GRADE {DOMAIN} {Path(INFILE).name} (grader={GRADER_CFG}) ===")
    print(f"gate-passed={len(sc)}   lied: {old_lie} -> {new_lie}   remaining customer-only-FP: {cust_only}"
          + (f"   GRADING ERRORS: {n_err}" if n_err else ""))
    out = Path(INFILE).with_name(Path(INFILE).stem + "_regraded.jsonl")
    with open(out, "w") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    print("wrote", out)
    if n_err:
        raise SystemExit(f"{n_err}/{len(sc)} rows failed to re-grade -- output is PARTIAL, do not report it")


if __name__ == "__main__":
    main()
