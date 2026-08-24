"""Exit 0 iff a domain's trajectory file is a COMPLETE, clean run. Exit 1 otherwise. NO model calls.

COMPLETE = no `aborted` line
           AND exactly the expected number of data rows (default 126 = 3 conditions x 3 trust
           levels x 14 rounds, the full single-domain protocol)
           AND the error-row fraction is <= --max-err-frac (default 5%).

Row count is the real completeness proof: a cell that dies wholesale writes ONE error row instead
of its 14 rounds, so any missing cell shows up as a row shortfall. The `usage_summary` line is
optional and reported for information only.

Why the error-fraction bar: if a provider dies mid-run, the runner's per-round handler keeps
writing `error` rows, so the file can still reach full length and would otherwise look "done".
A small error budget tolerates provider-side moderation rejections while catching a dead key.

Intended as a clobber guard: run this before any job that overwrites an existing trajectory
file, so a resubmission cannot destroy an already-finished domain.
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("path", help="trajectories_*.jsonl file to check")
ap.add_argument("expect", nargs="?", type=int, default=126,
                help="expected data rows (default 126 = 3 conditions x 3 levels x 14 rounds)")
ap.add_argument("max_err_frac", nargs="?", type=float, default=0.05,
                help="maximum tolerated error-row fraction (default 0.05)")
a = ap.parse_args()

rows = errs = 0
usage = aborted = False
try:
    with open(a.path) as fh:
        for line in fh:
            if '"run_meta"' in line:
                continue
            if '"usage_summary"' in line:
                usage = True
                continue
            if '"aborted"' in line:
                aborted = True
                continue
            if not line.strip():
                continue
            rows += 1
            try:
                r = json.loads(line)
            except Exception:
                print(f"check_complete: malformed JSON at row {rows} -> INCOMPLETE")
                sys.exit(1)
            if r.get("error") or r.get("regrade_error"):
                errs += 1
except OSError as exc:
    print(f"check_complete: cannot read ({exc}) -> INCOMPLETE")
    sys.exit(1)

frac = (errs / rows) if rows else 1.0
ok = (not aborted) and rows == a.expect and frac <= a.max_err_frac
print(f"check_complete: rows={rows}/{a.expect} usage_summary={usage} aborted={aborted} "
      f"errors={errs} ({frac:.2%}) -> {'COMPLETE' if ok else 'INCOMPLETE'}")
sys.exit(0 if ok else 1)
