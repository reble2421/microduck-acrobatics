#!/usr/bin/env python3
"""Assemble the final video (under 3 minutes, real time throughout) from the clips and figures.

    python bench/make_video.py OUT.mp4 [--github URL]

Every source is 1280x720 / 25 fps (clips) or a PNG (figures). Each scene is scaled into
the upper 1120x630 area and a subtitle band runs underneath, so the subtitles never
cover the readouts drawn in the clips. The storyboard is SCENES below.
"""

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FONT_B = "/mnt/c/Windows/Fonts/BIZ-UDGothicB.ttc"   # BIZ UDGothic (SIL Open Font License)
FONT_R = "/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc"
W, H, FPS = 1280, 720, 25
AREA = (80, 6, 1120, 630)                 # x, y, w, h of the scene
BAND_Y = AREA[1] + AREA[3] + 6            # subtitle band starts here
BG = (20, 22, 28)
INK, INK2 = (255, 255, 255), (200, 200, 192)

CLIPS = RES / "clips"
# Storyboard: the routine first; then only the moves that are NOT in the official
# Microduck policies (backroll, headstand entry, headspin), each with what was done to get it.
SCENES = [
    dict(kind="clip", src=RES / "routine" / "spin_14997_w4_video720" / "routine_trial000_OK.mp4",
         start=0.0, end=28.5, subs=[
             (0.0, "Microduck の連続技（強化学習・シミュレーション・すべて等速）"),
             (3.0, "歩く・止まる：公式の方策"),
             (4.5, "★ 後転：公式にない動き（今回学習）"),
             (8.5, "★ 逆立ち → ヘッドスピン：公式にない動き（今回学習）"),
             (19.5, "★ 指令で回転を止める"),
             (21.5, "倒れてから起き上がる：公式の方策")]),
    dict(kind="figure", src=RES / "figures" / "routine.png", seconds=6.0, subs=[
        (0.0, "この流れを50回試して、50回とも最後まで成功（モーター指令の遅れ15〜30ms）。")]),
    dict(kind="card", seconds=2.5, title="ポイント①　ヘッドスピン", sub="指令した速さで回り、指令で止まる"),
    dict(kind="clip", src=CLIPS / "headspin_solo.mp4", start=0.0, end=16.5, subs=[
        (0.0, "工夫：報酬は「指令した回転の速さ」との差で採点。"),
        (5.5, "工夫：重心が動くと報酬を払わない（その場で回る）。止まれの指令では回転に罰。"),
        (11.5, "左右どちらにも毎秒4rad（約1.5秒で1回転）。")]),
    dict(kind="clip", src=RES / "headspin" / "spin_tracking_14997_w45.mp4", start=0.0, end=31.5, subs=[
        (0.0, "点線が指令、実線が実際の回転。指令を切り替えても、ついていく。"),
        (8.0, "毎秒4〜5radの指令に対して、実際は指令の0.96〜1.10倍。"),
        (14.0, "止まれの指令で止まり、続けて逆回り（右回り）の指令。"),
        (18.5, "逆回りでも指令についていく（同じく指令の0.96〜1.10倍の範囲）。"),
        (28.5, "最後に止まれの指令で止まる。")]),
    dict(kind="clip", src=CLIPS / "learning_progress.mp4", start=0.0, end=8.5, subs=[
        (0.0, "逆立ちから学習を続けると、2,000反復で回り始める（同じ指令で比較）。")]),
    dict(kind="card", seconds=2.5, title="ポイント②　後転を「遅れに強く」",
         sub="別の物理エンジンで弱点を見つけ、学習し直す"),
    dict(kind="clip", src=CLIPS / "backroll_compare.mp4", start=0.0, end=7.0, subs=[
        (0.0, "工夫：学習とは別の物理エンジンで、条件を変えて50回ずつ試験。"),
        (3.5, "指令の遅れがないと乱れる弱点を見つけ、遅れの幅を0〜30msに広げて学習し直した。")]),
    dict(kind="figure", src=RES / "figures" / "backroll.png", seconds=6.0, subs=[
        (0.0, "遅れなしで、きれいに転がれた回数：17/50 → 50/50。")]),
    dict(kind="card", seconds=10.0, title="今後：実機で動かす", sub="シミュレーションで確かめた手順を、そのまま実機へ",
         lines=["実機のサーボの指令の遅れを測り、学習の条件をそれに合わせる",
                "まず後転と起き上がりから実機で確かめ、次にヘッドスピンへ",
                "頭の接地面の摩擦や床の違いにも強くなるよう、条件を広げて学習する"]),
    dict(kind="summary", seconds=12.0),
]


def font(path, size):
    return ImageFont.truetype(path, size)


def wrap(draw, text, fnt, width):
    """Character-level wrap for Japanese text."""
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=fnt) > width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + ([cur] if cur else [])


def subtitle(canvas, text):
    if not text:
        return
    dr = ImageDraw.Draw(canvas)
    f = font(FONT_B, 30)
    lines = wrap(dr, text, f, W - 120)
    y = BAND_Y + (H - BAND_Y - 40 * len(lines)) / 2
    for line in lines:
        dr.text((W / 2, y + 20), line, font=f, fill=INK, anchor="mm")
        y += 40


def place(canvas, img, light=False):
    x, y, w, h = AREA
    if light:                                  # figures sit on their own light panel
        canvas.paste(Image.new("RGB", (w, h), (252, 252, 251)), (x, y))
    s = min(w / img.width, h / img.height)
    im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    canvas.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))


def sub_at(subs, t):
    cur = ""
    for t0, text in subs or []:
        if t >= t0:
            cur = text
    return cur


def title_card(bg):
    path, t = bg
    r = imageio.get_reader(str(path))
    frame = Image.fromarray(np.asarray(r.get_data(int(t * FPS)))).convert("RGB")
    r.close()
    frame = frame.crop((160, 100, 1120, 630))           # the robot only: no title bar, no readouts
    s = max(W / frame.width, H / frame.height)
    frame = frame.resize((int(frame.width * s) + 1, int(frame.height * s) + 1), Image.LANCZOS)
    frame = frame.crop(((frame.width - W) // 2, (frame.height - H) // 2, (frame.width - W) // 2 + W,
                        (frame.height - H) // 2 + H))
    canvas = ImageEnhance.Brightness(frame).enhance(0.38)
    dr = ImageDraw.Draw(canvas)
    dr.text((W / 2, 250), "Microduck にヘッドスピンと後転を教える", font=font(FONT_B, 54), fill=INK, anchor="mm")
    dr.text((W / 2, 330), "強化学習でつくる、指令どおりに回るアクロバット", font=font(FONT_R, 34), fill=INK, anchor="mm")
    dr.text((W / 2, 430), "Physical AI 応用講座 最終課題　／　映像はすべてシミュレーション・等速", font=font(FONT_R, 26),
            fill=INK2, anchor="mm")
    return canvas


def section_card(title, sub, lines=()):
    """A divider between the parts of the video; with lines, a short list under it."""
    canvas = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(canvas)
    y0 = 300 if not lines else 150
    dr.rectangle([90, y0, 98, y0 + 120], fill=(57, 135, 229))
    dr.text((130, y0), title, font=font(FONT_B, 56), fill=INK)
    dr.text((132, y0 + 86), sub, font=font(FONT_R, 32), fill=INK2)
    y = y0 + 190
    for line in lines:
        dr.text((140, y), "●", font=font(FONT_B, 26), fill=(57, 135, 229))
        dr.text((180, y - 2), line, font=font(FONT_B, 30), fill=INK)
        y += 64
    return canvas


def summary_card(github):
    canvas = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(canvas)
    dr.text((90, 70), "まとめ", font=font(FONT_B, 48), fill=INK)
    items = ["★ ヘッドスピン：指令どおりの速さ（0.96〜1.10倍）で回り、指令で止まる",
             "★ 後転：遅れに強く学習し直し、遅れなしでも 17/50 → 50/50",
             "連続技：公式の「歩く・起き上がる」とつないで、50回中50回成功",
             "今後：実機のサーボに条件を合わせて、実機で動かす"]
    y = 170
    for it in items:
        dr.text((110, y), "●", font=font(FONT_B, 28), fill=(57, 135, 229))
        dr.text((150, y), it, font=font(FONT_B, 30), fill=INK)
        y += 70
    small = font(FONT_R, 22)
    dr.text((90, 500), f"コード：{github}", font=small, fill=INK2)
    dr.text((90, 540), "ロボット：Microduck（Pollen Robotics、オープンソース）　学習：mjlab・MuJoCo Warp・PPO",
            font=small, fill=INK2)
    dr.text((90, 575), "評価：MuJoCo＋BAMモーターモデル　フォント：BIZ UDゴシック（SIL Open Font License）",
            font=small, fill=INK2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--github", default="github.com/＜アカウント名＞/microduck-acrobatics")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with imageio.get_writer(str(args.out), fps=FPS, macro_block_size=8, quality=8) as wr:
        for sc in SCENES:
            if sc["kind"] in ("title", "summary", "card"):
                if sc["kind"] == "title":
                    img = title_card(sc["bg"])
                elif sc["kind"] == "card":
                    img = section_card(sc["title"], sc["sub"], sc.get("lines", ()))
                else:
                    img = summary_card(args.github)
                frame = np.asarray(img)
                for _ in range(int(sc["seconds"] * FPS)):
                    wr.append_data(frame)
                total += sc["seconds"]
            elif sc["kind"] == "figure":
                fig = Image.open(sc["src"]).convert("RGB")
                base = Image.new("RGB", (W, H), BG)
                place(base, fig, light=True)
                cache = {}
                for k in range(int(sc["seconds"] * FPS)):
                    text = sub_at(sc.get("subs"), k / FPS)
                    if text not in cache:
                        c = base.copy()
                        subtitle(c, text)
                        cache[text] = np.asarray(c)
                    wr.append_data(cache[text])
                total += sc["seconds"]
            else:
                r = imageio.get_reader(str(sc["src"]))
                k0, k1 = int(sc["start"] * FPS), int(sc["end"] * FPS)
                n = 0
                for k, src in enumerate(r):            # sequential read: seeking an mp4 is slow
                    if k < k0:
                        continue
                    if k >= k1:
                        break
                    c = Image.new("RGB", (W, H), BG)
                    place(c, Image.fromarray(np.asarray(src)).convert("RGB"))
                    subtitle(c, sub_at(sc.get("subs"), (k - k0) / FPS))
                    wr.append_data(np.asarray(c))
                    n += 1
                r.close()
                total += n / FPS
            print(f"  {sc['kind']:8s} {str(sc.get('src', '')).split('/')[-1]:40s} -> {total:6.1f} s", flush=True)
    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {total:.1f} s ({int(total // 60)}:{total % 60:04.1f}), {size:.1f} MB")


if __name__ == "__main__":
    main()
