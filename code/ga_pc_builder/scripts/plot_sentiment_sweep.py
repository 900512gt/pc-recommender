#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
畫口碑權重掃描的取捨圖（讀 sentiment_sweep.py 的輸出）。

上下兩張共用橫軸的小圖：上＝整機口碑、下＝顯卡跑分相對口碑 0% 的變化。
兩個量尺度不同，不畫雙 Y 軸。配色用 dataviz 規範驗證過的類別色（藍、橘、綠、黃），
在簡報底色 #EEECEC 上對比偏低，所以每條線末端直接標預算，不只靠圖例辨識。

執行（在 code/ga_pc_builder/ 底下）：
  python scripts/plot_sentiment_sweep.py
  python scripts/plot_sentiment_sweep.py --bg white     # 報告用的白底版本
輸出：results/v2_nocp/sentiment_sweep_<用途>.png
"""
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent

# 莫蘭迪學術風：降低飽和度。有變化的三條線用深海藍、赤陶橘、松石綠當主角；
# 3 萬口碑權重怎麼調都不變，用岩石灰退到背景當基準線。
# 驗證結果：色盲分辨與對比皆通過；深海藍彩度 0.099 略低於 0.10 門檻（刻意的低飽和），
# 每條線末端都有直接標示預算，不只靠顏色辨認。
SERIES = [(80000, "8 萬", "#2B5C8F"), (60000, "6 萬", "#D95F43"),
          (40000, "4 萬", "#2A9D8F"), (30000, "3 萬", "#8C8C8C")]
TEXT, MUTED, GRID = "#1a1a19", "#52514e", "#c9c7c2"
WIN_BAND, TRADE_BAND = "#d9e3ee", "#f2ddd3"   # 區段底色（極淡，標文字，不靠顏色辨識）
DEFAULT_SHARE = 1 / 3                          # 預設等權：效能、口碑、CP 值各 1/3
WIN_END = 36                                   # 雙贏區上界（35% 與 37% 兩個實測點之間）


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[int(r["budget"])][float(r["sent_share"])].append(r)
    out = {}
    for budget, by_share in agg.items():
        shares = sorted(by_share)
        mean = lambda key: [sum(float(r[key]) for r in by_share[s]) / len(by_share[s]) for s in shares]
        gpu = mean("bench_GPU")
        out[budget] = {"share": [s * 100 for s in shares], "sent": mean("sentiment"),
                       "gpu_pct": [(g / gpu[0] - 1) * 100 for g in gpu], "cpu": mean("bench_CPU")}
    return out


def main(args):
    data = load(HERE / "results" / "v2_nocp" / f"sentiment_sweep_{args.usage}.csv")
    plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12})
    bg = "none" if args.bg == "transparent" else "white"
    fig.patch.set_facecolor(bg)

    for ax in (ax1, ax2):
        ax.set_facecolor("none")
        # 6 萬在 35% 還是 RTX5070、37% 換成 RX9070，兩區的分界畫在中間 36%
        ax.axvspan(20, WIN_END, color=WIN_BAND, alpha=0.55, lw=0, zorder=0)
        ax.axvspan(WIN_END, 97, color=TRADE_BAND, alpha=0.45, lw=0, zorder=0)
        ax.axvline(DEFAULT_SHARE * 100, color=TEXT, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=11, length=0)
        ax.set_xlim(-2, 97)

    # 上：整機口碑
    # 6 萬與 8 萬在口碑佔比低時完全重合：6 萬用粗線大圓點墊底，8 萬細線小圓點疊在上面，兩者都看得到
    style = {80000: (2.0, 5.5, 5), 60000: (4.2, 10, 4), 40000: (2.2, 6.5, 3), 30000: (2.2, 6.5, 3)}
    ends = []
    for budget, label, color in SERIES:
        d = data[budget]
        lw, ms, z = style[budget]
        ax1.plot(d["share"], d["sent"], color=color, lw=lw, marker="o", ms=ms,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=z, label=label)
        ends.append((d["sent"][-1], label))
    # 末端直接標預算；6 萬、8 萬終點相同，上下錯開
    used = []
    for y, label in sorted(ends, reverse=True):
        yy = y
        while any(abs(yy - u) < 0.03 for u in used):
            yy -= 0.032
        used.append(yy)
        ax1.annotate(label, (95, y), xytext=(98.5, yy), textcoords="data", va="center",
                     fontsize=12, color=TEXT, annotation_clip=False)
    ax1.set_ylabel("整機口碑（0~1）", color=MUTED, fontsize=12)
    ax1.set_ylim(0.38, 0.86)
    ax1.text(28, 0.85, "雙贏區", ha="center", va="top", fontsize=12, color=TEXT, fontweight="bold")
    ax1.text(66.5, 0.85, "取捨區", ha="center", va="top", fontsize=12, color=TEXT, fontweight="bold")
    ax1.annotate("4 萬：CPU 換 R7 7700，跑分 +34%", (80, 0.673), xytext=(44.5, 0.615), fontsize=11, color=TEXT,
                 arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax1.annotate("3 萬：怎麼調都不變", (95, 0.475), xytext=(70, 0.5), fontsize=11, color=TEXT, va="bottom")
    ax1.text(DEFAULT_SHARE * 100 + 0.8, 0.395, "預設等權 33%", fontsize=11, color=TEXT, va="bottom")

    # 下：顯卡跑分變化（相對口碑 0%）
    for budget, label, color in SERIES:
        if budget not in (80000, 60000):
            continue   # 3 萬、4 萬顯卡沒換，畫出來只會疊在 0% 上蓋住其他線
        d = data[budget]
        lw, ms, z = style[budget]
        ax2.plot(d["share"], d["gpu_pct"], color=color, lw=lw, marker="o", ms=ms,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=z)
    ax2.axhline(0, color=MUTED, lw=1, zorder=2)
    ax2.set_ylabel("顯卡跑分變化（%）", color=MUTED, fontsize=12)
    ax2.set_ylim(-17, 17)
    ax2.annotate("8 萬：升級 RTX5070Ti，+13%", (40, 12.8), xytext=(43, 14.5), fontsize=11, color=TEXT,
                 arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax2.annotate("6 萬：37% 起換成口碑最好的 RX9070，−12%", (60, -11.7), xytext=(55, -6.2), fontsize=11, color=TEXT,
                 arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax2.text(1, 2.2, "3 萬、4 萬：顯卡不變", fontsize=11, color=MUTED)
    ax2.set_xlabel("口碑佔比（口碑在「效能＋口碑＋CP 值」中的比重，%）", color=MUTED, fontsize=12)
    ax2.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])

    leg = ax1.legend(loc="upper left", ncol=4, frameon=False, fontsize=11, handlelength=1.6,
                     bbox_to_anchor=(0, 1.13))
    for t in leg.get_texts():
        t.set_color(TEXT)

    out = HERE / "results" / "v2_nocp" / f"sentiment_sweep_{args.usage}{'' if args.bg == 'transparent' else '_white'}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=bg, transparent=args.bg == "transparent")
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", default="遊戲")
    ap.add_argument("--bg", choices=["transparent", "white"], default="transparent")
    main(ap.parse_args())
