#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可指定面向的資料覆蓋率：每個面向有多少比例的型號達到 10 則正負評價。

與 SentimentScorer 開放指定面向的規則相同（data/sentiment.py 的 _aspect_bounds）：
只算 GA 資料庫裡有賣的型號，過半型號達 MIN_MENTIONS 則才開放指定。
只畫整機口碑會看的三個類別（CPU、顯卡、主機板）。

執行（在 code/ga_pc_builder/ 底下）：
  python scripts/plot_aspect_coverage.py
  python scripts/plot_aspect_coverage.py --bg white
輸出：results/v2_nocp/aspect_coverage.png
"""
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from config import DB_PATH, MATCHED_FILES  # noqa: E402
from data.sentiment import SentimentScorer, MIN_MENTIONS  # noqa: E402

TEXT, MUTED, GRID = "#1a1a19", "#52514e", "#c9c7c2"
PASS, FAIL = "#2B5C8F", "#c9c7c2"
CATS = [("CPU", "CPU"), ("GPU", "顯卡"), ("主機板", "主機板")]


def coverage():
    stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        sc = SentimentScorer(MATCHED_FILES, DB_PATH)
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    db_models = sc._db_models(DB_PATH)
    pool = [k for k in sc.raw_aspects if not db_models or k in db_models]
    n_models = defaultdict(int)
    for key in pool:
        n_models[key[0]] += 1
    rows = []
    for cat, label in CATS:
        aspects = {a for key in pool if key[0] == cat for a in sc.raw_aspects[key]}
        items = []
        for a in aspects:
            ok = sum(sc.sample_counts[key][a] >= MIN_MENTIONS for key in pool if key[0] == cat)
            items.append((ok / n_models[cat], a, ok, n_models[cat], a in sc.selectable.get(cat, [])))
        rows.append((label, sorted(items, reverse=True)))
    return rows


def main(args):
    rows = coverage()
    plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    n = sum(len(items) for _, items in rows) + len(rows) - 1
    fig, ax = plt.subplots(figsize=(9, 0.5 * n + 0.8))
    bg = "none" if args.bg == "transparent" else "white"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor("none")

    y, ticks, labels = 0, [], []
    for label, items in rows:
        for share, aspect, ok, total, selectable in items:
            ax.barh(y, share * 100, height=0.66, color=PASS if selectable else FAIL, zorder=2)
            # 數字統一放在圖右側一欄，不會跟門檻虛線交疊
            ax.text(103, y, f"{share:.0%}（{ok}/{total}）", va="center", fontsize=15,
                    color=TEXT if selectable else MUTED, fontweight="bold" if selectable else "normal")
            ticks.append(y)
            labels.append(f"{label}　{aspect}")
            y += 1
        y += 1   # 類別之間空一行
    ax.axvline(50, color=TEXT, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.text(50.8, -0.9, "過半門檻 50%", fontsize=14, color=TEXT, va="bottom")
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(f"評價達 {MIN_MENTIONS} 則的型號比例", color=MUTED, fontsize=14)
    ax.grid(axis="x", color=GRID, lw=0.8, alpha=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=15, length=0)

    out = HERE / "results" / "v2_nocp" / f"aspect_coverage{'' if args.bg == 'transparent' else '_white'}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=bg, transparent=args.bg == "transparent")
    print(out)
    for label, items in rows:
        print(label, [(a, f"{s:.0%}", sel) for s, a, _, _, sel in items])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", choices=["transparent", "white"], default="transparent")
    main(ap.parse_args())
