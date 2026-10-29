#!/usr/bin/env python3
"""Headspin rate vs command delay: figure for the report.

    python bench/plot_delay_sweep.py results/delay_sweep_headspin_v1line \
        job2_HeadSpin_12998.onnx job3_HeadSpinRobust_14997.onnx

Reads <dir>/delay_<N>/compare.csv (compare.py --scenarios headspin --delay N, N = 0..6
physics steps of 5 ms) and writes <dir>/delay_sweep.png and delay_sweep.csv:
  top    achieved / commanded spin rate, mean over the four spin segments
         (+2, -2, +3, -3 rad/s); whiskers: the slowest and fastest segment mean
  bottom trials passing the headspin judge (every segment >= 0.7x, stops, stays on the head)
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONT = "/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc"
SEGS = ("right2", "left2", "right3", "left3")     # compare.py labels: right = +wz (counter-clockwise from above)
POLICIES = [   # (file name in compare.csv, legend label); categorical slots 1, 2 in this order
    ("job1_HeadSpin_15997.onnx", "改善前（遅れ15〜30msで学習）"),
    ("job1_HeadSpinRobust_17996.onnx", "改善後（遅れ0〜30msで学習）"),
]
COLORS = ["#2a78d6", "#eb6834"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
STEP_MS = 5


def load(root):
    out = []   # rows: delay_ms, policy, mean, lo, hi, passed, trials
    dirs = [p for p in root.glob("delay_*") if p.is_dir() and p.name.split("_")[1].isdigit()]
    for d in sorted(dirs, key=lambda p: int(p.name.split("_")[1])):
        rows = list(csv.DictReader(open(d / "compare.csv", encoding="utf-8")))
        for name, _ in POLICIES:
            rs = [r for r in rows if r["policy"] == name and r["scenario"] == "headspin"]
            if not rs:
                continue
            seg_means = [sum(float(r[f"{s}_ratio"]) for r in rs) / len(rs) for s in SEGS]
            out.append({"delay_ms": int(d.name.split("_")[1]) * STEP_MS, "policy": name,
                        "mean": sum(seg_means) / len(seg_means), "lo": min(seg_means), "hi": max(seg_means),
                        "passed": sum(r["success"] == "True" for r in rs), "trials": len(rs)})
    return out


def main():
    root = Path(sys.argv[1])
    if len(sys.argv) == 4:     # the two ONNX file names to compare (before, after); labels stay
        POLICIES[:] = [(sys.argv[2], POLICIES[0][1]), (sys.argv[3], POLICIES[1][1])]
    data = load(root)
    with open(root / "delay_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)

    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=FONT).get_name(), "font.size": 11,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
    for a in (ax, bx):
        a.axvspan(15, 30, color=GRID, alpha=0.6, lw=0, zorder=0)
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
    top = max(1.65, max(r["hi"] for r in data) + 0.25)    # room for the band label above the whiskers
    ax.text(22.5, top - 0.03, "改善前の学習時の遅れ", ha="center", va="top", color=INK2, fontsize=10)
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.7, color=MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
    ax.text(0.3, 1.02, "指令どおり（1.0倍）", color=INK2, fontsize=9, va="bottom")
    ax.text(33.8, 0.68, "合格の下限（0.7倍）", color=MUTED, fontsize=9, va="top", ha="right")

    width = len(POLICIES)
    for k, ((name, label), col) in enumerate(zip(POLICIES, COLORS)):
        rows = [r for r in data if r["policy"] == name]
        x = [r["delay_ms"] + (k - (width - 1) / 2) * 0.5 for r in rows]   # nudge so whiskers don't overlap
        m = [r["mean"] for r in rows]
        ax.vlines(x, [r["lo"] for r in rows], [r["hi"] for r in rows], color=col, lw=1.2, alpha=0.6, zorder=2)
        ax.plot(x, m, color=col, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, zorder=3, label=label)
        nudge = 9 if m[-1] >= max(r["mean"] for r in data if r["delay_ms"] == rows[-1]["delay_ms"]) else -9
        ax.annotate(label.split("（")[0], (x[-1], m[-1]), xytext=(9, nudge), textcoords="offset points",
                    color=INK, fontsize=10, va="center")
        bx.bar([r["delay_ms"] + (k - (width - 1) / 2) * 1.8 for r in rows], [r["passed"] for r in rows],
               width=1.6, color=col, label=label, zorder=2)
    trials = data[0]["trials"]
    ax.set_ylim(0, top)
    ax.set_ylabel("回転の速さ（指令の何倍か）")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)
    ax.set_title("指令の遅れとヘッドスピンの速さ（このPCの物理エンジン、毎秒2・3radの左右4指令）",
                 loc="left", color=INK, fontsize=12)
    bx.set_ylim(0, trials)
    bx.set_yticks(range(0, trials + 1))
    bx.set_ylabel(f"合格（{trials}回中）")
    bx.set_xlabel("モーター指令の遅れ（ms）")
    bx.set_xticks(sorted({r["delay_ms"] for r in data}))
    bx.set_xlim(-2.5, 34)
    fig.savefig(root / "delay_sweep.png", dpi=160, bbox_inches="tight")
    for r in data:
        print(f"{r['delay_ms']:3d} ms  {r['policy']:34s} mean {r['mean']:.2f}  "
              f"range {r['lo']:.2f}-{r['hi']:.2f}  pass {r['passed']}/{r['trials']}")
    print(f"wrote {root / 'delay_sweep.png'}")


if __name__ == "__main__":
    main()
