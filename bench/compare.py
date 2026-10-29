#!/usr/bin/env python3
"""Run the same scenarios on several ONNX policies and write one comparison table.

    python bench/compare.py --role headstand \
        --onnx policies/headspin.onnx policies/headspin_robust.onnx \
        --scenarios headspin headspin_slow headspin_fast -n 10 --delay train

Every policy sees exactly the same trials (same seeds, same conditions), so the
differences in the table come from the policies. Writes compare.md and
compare.csv to --out (default results/compare/<time>).
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from battery import DEFAULT_POLICIES, RESULTS, run_trial, sample_conditions, wilson  # noqa: E402
from duckbench.scenarios import build_scenarios  # noqa: E402
from duckbench.sim import Policy  # noqa: E402

# Metrics averaged into the table, per scenario (the rest stay in compare.csv).
KEY_METRICS = {
    "headstand": ("entry_s", "on_head_fraction"),
    "headstand_hold": ("on_head_fraction", "head_drift_m"),
    "headstand_long": ("on_head_fraction",),
    "headspin": ("right2_ratio", "left2_ratio", "right3_ratio", "left3_ratio", "stop_deg_s",
                 "right3_drift_mm_s"),
    "headspin_from_stand": ("right2_ratio", "left2_ratio", "right3_ratio", "left3_ratio", "stop_deg_s"),
}


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", required=True, help="the role the policies under test play (e.g. headstand)")
    ap.add_argument("--onnx", nargs="+", type=Path, required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("-n", "--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", choices=("nominal", "train"), default="train")
    ap.add_argument("--delay", default="none",
                    help="command delay: none (infer_policy rehearsal), train (3-6 physics steps), or N steps")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scns = build_scenarios()
    for s in args.scenarios:
        if s not in scns:
            raise SystemExit(f"unknown scenario {s}; choose from {sorted(scns)}")
    others = {r: Policy(p) for r, p in DEFAULT_POLICIES.items() if p.is_file()}
    out = args.out or RESULTS / "compare" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    conds = [sample_conditions(args.dr, np.random.default_rng(args.seed + i + 1_000_003), 0.0, (0.5, 3.0),
                               args.delay)
             for i in range(args.trials)]
    table, all_rows = [], []
    for path in args.onnx:
        policies = dict(others)
        policies[args.role] = Policy(path)
        line = {"policy": path.name}
        for name in args.scenarios:
            scn = scns[name]
            rows = []
            for i, cond in enumerate(conds):
                _, res = run_trial(scn, policies, cond, args.seed + i)
                rows.append({"policy": path.name, "scenario": name, "trial": i, **cond, **res})
            k = sum(r["success"] for r in rows)
            lo, hi = wilson(k, len(rows))
            line[name] = f"{k}/{len(rows)} ({100 * lo:.0f}-{100 * hi:.0f}%)"
            for key in KEY_METRICS.get(name, ()):
                v = mean_of(rows, key)
                line[f"{name}.{key}"] = None if v is None else round(v, 3)
            all_rows += rows
            print(f"{path.name:32s} {name:18s} {line[name]}", flush=True)
        table.append(line)

    cols = ["policy"] + [c for c in table[0] if c != "policy"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for line in table:
        md.append("| " + " | ".join("" if line.get(c) is None else str(line[c]) for c in cols) + " |")
    header = (f"# Policy comparison ({datetime.now():%Y-%m-%d %H:%M})\n\n"
              f"role={args.role}, {args.trials} trials per cell, dr={args.dr}, delay={args.delay}, seed={args.seed}. "
              "Success column: successes/trials (95% Wilson CI).\n"
              "Metrics are means over trials (ratio = achieved / commanded speed).\n\n")
    (out / "compare.md").write_text(header + "\n".join(md) + "\n", encoding="utf-8")
    keys = sorted({k for r in all_rows for k in r})
    with open(out / "compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out / 'compare.md'}")
    print(header + "\n".join(md))


if __name__ == "__main__":
    main()
