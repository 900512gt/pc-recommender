#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整體口碑 vs 跑分、CP 值：口碑放進 fitness 是不是重複計算。

2×2 小圖：列＝顯卡、CPU（兩者跑分尺度不同，分開畫）；欄＝跑分、CP 值（跑分 ÷ 價格）。
每個點是一個型號，縱軸是 GA 實際使用的型號口碑（各面向類別內正規化後的平均，SentimentScorer.get）。
所有面向都不到 10 則評價的型號，分數等於類別平均，畫成空心點；標題的 Spearman ρ 用全部型號。

執行（在 code/ga_pc_builder/ 底下）：
  python scripts/plot_sentiment_vs_spec.py
  python scripts/plot_sentiment_vs_spec.py --bg white
輸出：results/v2_nocp/sentiment_vs_spec.png
"""
import os
import sys
import json
import argparse
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from config import DB_PATH, MATCHED_FILES  # noqa: E402
from data.sentiment import SentimentScorer, MIN_MENTIONS  # noqa: E402
from plot_perf_vs_sentiment import spearman  # noqa: E402

TEXT, MUTED, GRID = "#1a1a19", "#52514e", "#c9c7c2"
ROWS = [("GPU", "顯卡", "#2B5C8F"), ("CPU", "CPU", "#D95F43")]


def load():
    stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        sc = SentimentScorer(MATCHED_FILES, DB_PATH)
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    db = json.load(open(DB_PATH, encoding="utf-8"))
    data = {}
    for cat, _, _ in ROWS:
        by_model = {}
        for item in db[cat]:
            if item.get("benchmark") and item.get("price"):
                by_model.setdefault(item["ptt_model"], []).append((float(item["benchmark"]), float(item["price"])))
        pts = []
        for model, vals in by_model.items():
            score = sc.get(cat, model)
            if score is None:
                continue
            bench = st.median(v[0] for v in vals)
            price = st.median(v[1] for v in vals)
            counts = sc.sample_counts.get((cat, model), {})
            enough = any(n >= MIN_MENTIONS for n in counts.values())
            pts.append({"model": model, "bench": bench, "cp": bench / price, "score": score, "enough": enough})
        data[cat] = pts
    return data


def rho(pts, key, only_enough=False):
    p = [x for x in pts if x["enough"]] if only_enough else pts
    return spearman([x[key] for x in p], [x["score"] for x in p]), len(p)


def main(args):
    data = load()
    plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), gridspec_kw={"hspace": 0.55, "wspace": 0.18})
    bg = "none" if args.bg == "transparent" else "white"
    fig.patch.set_facecolor(bg)

    for r, (cat, label, color) in enumerate(ROWS):
        pts = data[cat]
        for c, (key, xlabel) in enumerate([("bench", "PassMark 跑分"), ("cp", "CP 值（跑分 ÷ 價格）")]):
            ax = axes[r][c]
            ax.set_facecolor("none")
            full = [x for x in pts if x["enough"]]
            thin = [x for x in pts if not x["enough"]]
            ax.scatter([x[key] for x in full], [x["score"] for x in full], s=70, color=color,
                       edgecolor="white", linewidth=1.3, zorder=3)
            ax.scatter([x[key] for x in thin], [x["score"] for x in thin], s=60, facecolor="none",
                       edgecolor=MUTED, linewidth=1.2, zorder=3)
            value, n = rho(pts, key)
            ax.set_title(f"{label}口碑 vs {xlabel.split('（')[0]}　ρ = {value:.2f}", loc="left",
                         color=TEXT, fontsize=15, fontweight="bold", pad=10)
            ax.set_xlabel(xlabel, color=MUTED, fontsize=13)
            if c == 0:
                ax.set_ylabel("整體口碑（0~1）", color=MUTED, fontsize=13)
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
            ax.set_ylim(0.15, 0.9)
            ax.grid(color=GRID, lw=0.8, alpha=0.7)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=12, length=0)
            print(f"{label} vs {key}: 全部 ρ={value:.2f}（n={n}）；"
                  f"只看評價足夠 ρ={rho(pts, key, True)[0]:.2f}（n={rho(pts, key, True)[1]}）")

    fig.text(0.125, 0.005, f"實心：有面向達 {MIN_MENTIONS} 則評價　空心：評價不足，分數為類別平均",
             color=MUTED, fontsize=12)
    out = HERE / "results" / "v2_nocp" / f"sentiment_vs_spec{'' if args.bg == 'transparent' else '_white'}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=bg, transparent=args.bg == "transparent")
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", choices=["transparent", "white"], default="transparent")
    main(ap.parse_args())
