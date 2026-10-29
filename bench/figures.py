#!/usr/bin/env python3
"""Report figures, rebuilt from the result files (results/...). One PNG per figure.

    python bench/figures.py [NAME ...]          (needs matplotlib; imageio for routine_strip, headstand_strip)

Writes results/figures/<name>.png. Without names, every figure is drawn. Each figure
reads only saved results; nothing is simulated here.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import figstyle as fs
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"
OUT = RES / "figures"


def rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=fs.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.png")


# ---- 1. headspin on command: command vs measured --------------------------------------

def tracking():
    tr = rows(RES / "headspin" / "spin_tracking_14997_w45.csv")
    t = [float(r["t"]) for r in tr]
    cmd = [float(r["command_rad_s"]) for r in tr]
    meas = [float(r["measured_rad_s"]) for r in tr]
    k0 = 25                                   # the measured rate needs 0.5 s of history
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fs.grid_y(ax)
    ax.axhline(0, color=fs.MUTED, lw=0.8)
    ax.step(t, cmd, where="post", color=fs.SERIES[1], lw=2, ls=(0, (4, 3)), label="指令")
    ax.plot(t[k0:], meas[k0:], color=fs.SERIES[0], lw=2, label="実際の回転（0.5秒平均）")
    ax.set_xlabel("時間（秒）")
    ax.set_ylabel("回転の速さ（rad/s、＋は左回り）")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-6, 6)
    ax.set_yticks([-5, -4, 0, 4, 5])
    ax.set_title("指令どおりの速さで回る：毎秒4・5radの指令に対して0.96〜1.10倍")
    ax.legend(loc="upper right", ncols=2)
    fig.text(0.99, -0.07, "このPCの物理エンジン（MuJoCo＋モーターモデル）、指令の遅れ20ms",
             ha="right", va="top", fontsize=9, color=fs.INK2)
    save(fig, "tracking")


# ---- 2. routine: stage-by-stage success, with and without the command delay ----------

STAGES = [("walk", "歩く"), ("stop", "止まる"), ("recover", "後転から立つ"), ("headstand", "逆立ち"),
          ("spin1", "左回り 毎秒4rad"), ("spin2", "右回り 毎秒4rad"), ("spin_stop", "回転を止める"),
          ("release", "倒れる"), ("standup", "起き上がる")]


def routine():
    runs = [("学習と同じ遅れ（15〜30ms）", RES / "routine" / "spin_14997_w4_delay_train" / "summary.json"),
            ("遅れなし", RES / "routine" / "spin_14997_w4_delay_none" / "summary.json")]
    data = [(label, json.load(open(p, encoding="utf-8"))) for label, p in runs]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    fs.grid_x(ax)
    n_st = len(STAGES)
    h = 0.38
    for k, ((label, s), col) in enumerate(zip(data, fs.SERIES)):
        ys = [i + (k - 0.5) * (h + 0.04) for i in range(n_st)]
        vals = [s["passed"].get(name, 0) for name, _ in STAGES]
        ax.barh(ys, vals, height=h, color=col, label=f"{label}：最後まで {s['completed']}/{s['trials']}")
        for y, v in zip(ys, vals):
            ax.text(v + 0.6, y, f"{v}", va="center", fontsize=9, color=fs.INK2)
    ax.set_yticks(range(n_st))
    ax.set_yticklabels([lab for _, lab in STAGES])
    ax.invert_yaxis()
    trials = data[0][1]["trials"]
    ax.set_xlim(0, trials * 1.12)
    ax.set_xlabel(f"その段まで成功した回数（{trials}回中）")
    ax.set_title("連続技の段階ごとの成功：遅れがあれば全段50/50、遅れなしは回転で止まる")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncols=2, fontsize=10)
    save(fig, "routine")


# ---- 3. backroll: clean rolls before / after the latency-robust training ---------------

def backroll():
    src = {("改善前", "none"): RES / "delay_20261002" / "backroll_delay_none" / "trials.csv",
           ("改善前", "train"): RES / "delay_20261002" / "backroll_delay_train" / "trials.csv",
           ("改善後", "none"): RES / "backroll_robust" / "br_robust_12998_delay_none" / "trials.csv",
           ("改善後", "train"): RES / "backroll_robust" / "br_robust_12998_delay_train" / "trials.csv"}
    groups = [("none", "遅れなし"), ("train", "学習と同じ遅れ（15〜30ms）")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fs.grid_y(ax)
    w = 0.36
    for k, (pol, col) in enumerate(zip(("改善前", "改善後"), fs.SERIES)):
        xs, vals, lo, hi = [], [], [], []
        for g, (key, _) in enumerate(groups):
            tr = rows(src[(pol, key)])
            n, c = len(tr), sum(r["clean"] == "True" for r in tr)
            a, b = fs.wilson(c, n)
            xs.append(g + (k - 0.5) * (w + 0.04))
            vals.append(100 * c / n)
            lo.append(100 * (c / n - a))
            hi.append(100 * (b - c / n))
            ax.text(xs[-1], 100 * b + 3, f"{c}/{n}", ha="center", va="bottom", fontsize=10, color=fs.INK)
        label = "改善前（遅れ15〜30msで学習）" if pol == "改善前" else "改善後（遅れ0〜30msで学習）"
        ax.bar(xs, vals, width=w, color=col, label=label)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=fs.INK2, elinewidth=1, capsize=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([lab for _, lab in groups])
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("きれいに転がれた割合（%）")
    ax.set_title("後転：遅れの幅を広げて学習し直すと、遅れなしでも50/50")
    ax.legend(loc="upper left", fontsize=9, ncols=2, bbox_to_anchor=(0, -0.13))
    save(fig, "backroll")


# ---- 4. backroll -> stand: how long the roll runs before the stand policy takes over ---

def handoff():
    # the 10/02 sweep (original backroll) predates the delay option: the bench had no delay then
    series = [("改善前の後転・遅れなし", RES / "backroll_20261002" / "handoff", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]),
              ("改善後の後転・遅れなし", RES / "handoff_robust" / "delay_none", [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]),
              ("改善後の後転・学習と同じ遅れ", RES / "handoff_robust" / "delay_train", [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0])]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fs.grid_y(ax)
    for k, ((label, root, hs), col) in enumerate(zip(series, fs.SERIES)):
        xs, ys, lo, hi = [], [], [], []
        for h in hs:
            p = root / f"h{h}" / "summary.json"
            if not p.is_file():
                continue
            s = json.load(open(p, encoding="utf-8"))
            r = s["successes"] / s["trials"]
            a, b = fs.wilson(s["successes"], s["trials"])
            xs.append(h + (k - 1) * 0.025)          # nudge so the whiskers don't overlap
            ys.append(100 * r)
            lo.append(100 * (r - a))
            hi.append(100 * (b - r))
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, lw=2, marker="o", ms=7, mec=fs.SURFACE, mew=1.5,
                    elinewidth=1, capsize=3, label=label)
    ax.set_xlabel("後転を始めてから「立つ」に切り替えるまでの時間（秒）")
    ax.set_ylabel("立ち上がれた割合（%）")
    ax.set_ylim(0, 108)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    ax.set_title("つなぎ目の間合い：後転から「立つ」への切り替え（各20回）")
    ax.legend(loc="lower right")
    save(fig, "handoff")


# ---- 5. headspin training: v1 headstand -> headspin -> latency-robust ---------------------

def _scalars(run, tags):
    """{tag: {iteration: value}} for one run of results/training/headspin_curves.csv
    (the training logs' episode rewards, exported per iteration)."""
    out = {tag: {} for tag in tags}
    for r in rows(RES / "training" / "headspin_curves.csv"):
        if r["run"] == run:
            for tag in tags:
                if r[tag] != "":
                    out[tag][int(r["iteration"])] = float(r[tag])
    return out


def _smooth(series, window=50):
    steps = sorted(series)
    out = []
    for i, s in enumerate(steps):
        w = [series[x] for x in steps[max(0, i - window + 1):i + 1]]
        out.append((s, sum(w) / len(w)))
    return out


def training():
    tags = ["headspin_track", "headstand_hold"]
    runs = [("headspin", "ヘッドスピンの学習\n（遅れ15〜30ms）"),
            ("headspin_robust", "遅れに強い版\n（遅れ0〜30ms）")]
    data = [(_scalars(run, tags), label) for run, label in runs]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, gridspec_kw={"hspace": 0.12})
    names = {"headspin_track": "回転の追従の報酬", "headstand_hold": "逆立ちを保つ報酬"}
    for ax, tag in zip(axes, tags):
        fs.grid_y(ax)
        for sc, _ in data:
            # right after a (re)start no 12 s episode has finished yet, so the logged episode
            # rewards are of short episodes: skip the first SKIP iterations of each run
            first = min(sc[tag])
            pts = _smooth({k: v for k, v in sc[tag].items() if k >= first + SKIP})
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=fs.SERIES[0], lw=2)
        ax.set_ylabel(names[tag])
    starts = [min(sc[tags[0]]) for sc, _ in data]
    ends = [max(sc[tags[0]]) for sc, _ in data]
    for ax in axes:
        ax.axvline(starts[1], color=fs.MUTED, lw=1, ls=(0, (3, 3)))
    for (s, e), (_, label) in zip(zip(starts, ends), data):
        axes[0].text((s + e) / 2, 0.02, label, ha="center", va="bottom", fontsize=9.5, color=fs.INK2,
                     transform=axes[0].get_xaxis_transform())
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("学習の反復回数（逆立ち 9,999回の続きから）")
    axes[0].set_title("ヘッドスピンの学習の推移（50反復の移動平均、2048体を並列に学習）")
    fig.text(0.99, -0.03, f"各学習の開始直後 {SKIP} 反復は、試行が一巡しておらず記録の値が小さく出るため除いた",
             ha="right", va="top", fontsize=9, color=fs.INK2)
    save(fig, "training")


SKIP = 60


# ---- 5b. the same headspin task from the headstand policy vs from random weights ------------

def init_compare():
    tags = ["headspin_track", "headstand_hold"]
    runs = [("headspin", "逆立ちの方策から（+3,000反復）"), ("headspin_scratch", "ランダムな重みから（8,000反復）")]
    data = [(_scalars(run, tags), label) for run, label in runs]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, gridspec_kw={"hspace": 0.12})
    names = {"headspin_track": "回転の追従の報酬", "headstand_hold": "逆立ちを保つ報酬"}
    for ax, tag in zip(axes, tags):
        fs.grid_y(ax)
        for (sc, label), col in zip(data, fs.SERIES):
            first = min(sc[tag])
            pts = _smooth({k - first: v for k, v in sc[tag].items() if k >= first + SKIP})
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, label=label)
        ax.set_ylabel(names[tag])
    axes[0].set_ylim(bottom=min(0, axes[0].get_ylim()[0]))
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=9.5)
    axes[1].set_xlabel("ヘッドスピンの学習を始めてからの反復回数")
    axes[0].set_title("初期化の比較：同じタスク（報酬・指令）で、学習の始め方を変える（50反復の移動平均）")
    fig.text(0.99, -0.03, f"各学習の開始直後 {SKIP} 反復は、試行が一巡しておらず記録の値が小さく出るため除いた",
             ha="right", va="top", fontsize=9, color=fs.INK2)
    save(fig, "init_compare")


# ---- 6. system overview: train -> ONNX policies -> evaluate (and back) ---------------------

def system():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, title, lines, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=(fs.SERIES[0] + "1f") if accent else fs.SURFACE,
                                    ec=fs.SERIES[0] if accent else fs.MUTED, lw=1.6 if accent else 1.1))
        ax.text(x + 1.6, y + h - 2.2, title, fontsize=11.5, color=fs.INK, va="top", fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(x + 1.6, y + h - 7.0 - 3.6 * i, line, fontsize=9.5, color=fs.INK2, va="top")

    def arrow(x0, y0, x1, y1, text=None, tx=0, ty=0, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                     color=fs.MUTED, connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, fontsize=9, color=fs.INK2, ha="center")

    y0, h = 13, 25
    box(0.5, y0, 30, h, "学習（RTX 4060 Laptop）",
        ["MuJoCo Warp で2048体を並列", "強化学習（PPO、mjlab）", "技ごとに報酬を設計",
         "指令の遅れ0〜30msをばらつかせる", "1反復約3秒、1晩2,000〜3,000反復"])
    box(33, y0, 30, h, "方策（ONNX、技ごとに1つ）",
        ["ヘッドスピン・後転（今回学習）", "歩く・立つ・起き上がる（公開版）", "入力：全技共通の61個の数",
         "　姿勢・関節・前回の出力・指令", "出力：14個のモーターの目標角度"], accent=True)
    box(65.5, y0, 30, h, "評価（このPCのCPU）",
        ["MuJoCo＋モーターモデル（BAM）", "遅れ・電圧・関節にばらつき", "方策を切り替えて連続技",
         "段ごとの成功率を50回ずつ", "回転の速さを指令と比べる"])
    mid = y0 + h / 2
    arrow(30.8, mid, 32.7, mid)
    arrow(63.3, mid, 65.2, mid)
    # back from the evaluation to the training: an elbow under the boxes, words below it
    ax.plot([80.5, 80.5, 15.5], [y0 - 0.4, 7.5, 7.5], color=fs.MUTED, lw=1.2)
    arrow(15.5, 7.5, 15.5, y0 - 0.4)
    ax.text(48, 5.6, "見つけた弱点を学習に戻す（例：遅れの幅を広げて学習し直す）", fontsize=9.5,
            color=fs.INK2, ha="center", va="top")
    ax.text(0.5, 43.5, "システムの構成：別の物理エンジンで確かめ、弱点を学習に戻す", fontsize=12, color=fs.INK, va="top")
    save(fig, "system")


# ---- 7. the routine as a strip of frames ------------------------------------------------

STRIP = [(1.5, "① 歩く"), (5.4, "② 後転"), (7.6, "③ 立つ"), (10.5, "④ 逆立ち"),
         (13.5, "⑤ 左回り 毎秒4rad"), (17.5, "⑥ 右回り 毎秒4rad"), (22.2, "⑦ 止まって倒れる"), (27.0, "⑧ 起き上がる")]


def routine_strip():
    import imageio.v2 as imageio
    import numpy as np
    video = RES / "routine" / "spin_14997_w4_video720" / "routine_trial000_OK.mp4"
    reader = imageio.get_reader(str(video))
    fps = reader.get_meta_data()["fps"]
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.6), gridspec_kw={"wspace": 0.04, "hspace": 0.22})
    for ax, (t, label) in zip(axes.flat, STRIP):
        frame = np.asarray(reader.get_data(int(round(t * fps))))
        h, w = frame.shape[:2]
        ax.imshow(frame[int(0.19 * h):int(0.89 * h), int(0.27 * w):int(0.73 * w)])   # robot area, overlay cut
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, fontsize=10, loc="center", color=fs.INK)
    reader.close()
    fig.suptitle("連続技（等速、約28秒）：方策を切り替えながら8つの段をこなす", x=0.125, ha="left",
                 fontsize=12, color=fs.INK, y=0.99)
    save(fig, "routine_strip")


# ---- 8. delay sweep (drawn by plot_delay_sweep.py) ----------------------------------------

def delay_sweep():
    src = RES / "delay_sweep_headspin_v1line" / "delay_sweep.png"
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, OUT / "delay_sweep.png")
    print(f"copied {src} -> {OUT / 'delay_sweep.png'}")


# ---- 9. the headstand entry as a strip of frames ----------------------------------------

HS_STRIP = [(0.0, "① 立った姿勢"), (0.16, "② 前に倒れる"), (0.32, "③ 頭頂が着く"),
            (0.64, "④ 足を上げる"), (2.0, "⑤ 頭頂だけで立つ"), (11.5, "⑥ 保ち続ける")]


def headstand_strip():
    # videos are not kept in the repository: render this one first with
    # python bench/battery.py headstand -n 20 --videos 1 --policy headstand=policies/headstand.onnx \
    #     --out results/headstand_20261001/headstand
    import imageio.v2 as imageio
    import numpy as np
    video = RES / "headstand_20261001" / "headstand" / "headstand_trial000_OK.mp4"
    reader = imageio.get_reader(str(video))
    fps = reader.get_meta_data()["fps"]
    fig, axes = plt.subplots(1, len(HS_STRIP), figsize=(10, 2.5), gridspec_kw={"wspace": 0.04})
    for ax, (t, label) in zip(axes, HS_STRIP):
        frame = np.asarray(reader.get_data(int(round(t * fps))))
        h, w = frame.shape[:2]
        ax.imshow(frame[int(0.23 * h):int(0.75 * h), int(0.36 * w):int(0.70 * w)])   # the robot only
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, fontsize=10, loc="center", color=fs.INK)
        ax.set_xlabel(f"{t:g}秒", fontsize=9, color=fs.INK2)
    reader.close()
    save(fig, "headstand_strip")


FIGURES = {"tracking": tracking, "routine": routine, "backroll": backroll, "handoff": handoff,
           "training": training, "init_compare": init_compare, "system": system, "routine_strip": routine_strip,
           "delay_sweep": delay_sweep, "headstand_strip": headstand_strip}

if __name__ == "__main__":
    fs.apply()
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
