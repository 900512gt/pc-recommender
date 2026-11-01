#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
簡報「推薦結果」兩張圖：實際跑一次 GA，畫出綜合口碑與指定面向的推薦結果。

  result_overall：6 萬遊戲的預設推薦，各零件的口碑長條。
                  只有 CPU、顯卡、主機板計入整機口碑（深色），其他零件淺色。
  result_aspect ：指定「顯卡 VRAM」前後的顯卡比較。左＝六個面向的口碑（啞鈴圖），
                  右＝跑分。兩個量尺度不同，分成兩張小圖，不畫雙 Y 軸。
配色沿用 plot_sentiment_sweep.py 驗證過的深海藍、赤陶橘；每個點與長條都直接標數值，
不只靠顏色辨識。

執行（在 code/ga_pc_builder/ 底下）：
  PYTHONHASHSEED=0 python scripts/plot_result_cases.py
  PYTHONHASHSEED=0 python scripts/plot_result_cases.py --bg white     # 報告用的白底版本
輸出：results/v2_nocp/result_overall.png、result_aspect.png
"""
import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from config import DB_PATH, MATCHED_FILES  # noqa: E402
from data.catalog import PartCatalog  # noqa: E402
from data.sentiment import SentimentScorer  # noqa: E402
from advisor.compatibility import CompatibilityChecker  # noqa: E402
from core.ga_engine import GARecommender  # noqa: E402

BLUE, ORANGE = "#2B5C8F", "#D95F43"
TEXT, MUTED, GRID, FADED = "#1a1a19", "#52514e", "#c9c7c2", "#cfcdc8"
BUDGET, USAGE, PREFS = 60000, "遊戲", {"GPU": ["VRAM"]}
LABELS = {"GPU": "顯卡", "風冷": "散熱", "水冷": "散熱"}
ORDER = ["CPU", "GPU", "主機板", "記憶體", "SSD", "電源", "風冷", "水冷", "機殼"]


def run_ga(catalog, scorer, checker, prefs):
    random.seed(0)
    np.random.seed(0)
    ga = GARecommender(catalog, scorer, checker, usage=USAGE, budget=BUDGET,
                       pop_size=300, generations=300, aspect_prefs=prefs)
    return ga, ga.run(verbose=False)[0]


def short(part):
    return part.specs.get("ptt_model") or part.short_name


def plot_overall(ax, ga, build, scorer):
    rows = []
    for cat in ORDER:
        part = build.parts.get(cat)
        if part is None:
            continue
        counted = cat in ga._sent_cats
        rows.append((LABELS.get(cat, cat), short(part), part.price, scorer.get(cat, part.short_name), counted))
    # 計入整機口碑的排在上面
    rows.sort(key=lambda r: not r[4])
    y = np.arange(len(rows))[::-1]
    for yi, (cat, model, price, score, counted) in zip(y, rows):
        ax.barh(yi, score, height=0.56, color=BLUE if counted else FADED, zorder=2)
        # 數值寫在長條內側，避免與 0.5 虛線重疊
        ax.text(score - 0.015, yi, f"{score:.2f}", va="center", ha="right", fontsize=12,
                color="white" if counted else TEXT, fontweight="bold" if counted else "normal")
        ax.text(-0.02, yi, f"{cat}  {model}", va="center", ha="right", fontsize=12,
                color=TEXT if counted else MUTED)
        ax.text(1.0, yi, f"${price:,}", va="center", ha="right", fontsize=11, color=MUTED)
    ax.axvline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(0.51, y[0] + 0.62, "類別平均 0.5", ha="left", fontsize=11, color=MUTED)
    counted = [r[3] for r in rows if r[4]]
    ax.text(1.0, y[-1] - 0.95, f"整機口碑＝深色三項平均＝{np.mean(counted):.2f}", ha="right",
            fontsize=12, color=TEXT, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(y[-1] - 1.3, y[0] + 1.0)
    ax.axis("off")


def plot_aspect(ax_l, ax_r, scorer, gpu_a, gpu_b, picked):
    aspects = list(scorer.get_aspects("GPU", gpu_a.short_name))
    a = scorer.get_aspects("GPU", gpu_a.short_name)
    b = scorer.get_aspects("GPU", gpu_b.short_name)
    name_a, name_b = short(gpu_a), short(gpu_b)
    y = np.arange(len(aspects))[::-1]
    for yi, asp in zip(y, aspects):
        hl = asp in picked
        if hl:
            ax_l.fill_between([-0.02, 1.4], yi - 0.42, yi + 0.42, color="#e6e4df", zorder=0)
        ax_l.plot([a[asp], b[asp]], [yi, yi], color=GRID, lw=2, zorder=1)
        ax_l.scatter(a[asp], yi, s=90, color=ORANGE, edgecolor="white", lw=1.5, zorder=3)
        ax_l.scatter(b[asp], yi, s=90, color=BLUE, edgecolor="white", lw=1.5, zorder=3)
        ax_l.text(-0.04, yi, asp + ("（指定）" if hl else ""), ha="right", va="center", fontsize=12,
                  color=TEXT, fontweight="bold" if hl else "normal")
    # 指定的面向在同一列右側標正負評則數
    for asp in picked:
        yi = y[aspects.index(asp)]
        pa, na = scorer.raw_counts[("GPU", scorer._resolve("GPU", gpu_a.short_name)[1])][asp]
        pb, nb = scorer.raw_counts[("GPU", scorer._resolve("GPU", gpu_b.short_name)[1])][asp]
        ax_l.text(1.07, yi, f"正／負評\n{pa}／{na} → {pb}／{nb}", ha="left", va="center", fontsize=11, color=TEXT)
    ax_l.set_xlim(-0.02, 1.4)
    ax_l.set_ylim(y[-1] - 0.9, y[0] + 1.1)
    ax_l.set_title("顯卡各面向口碑（類別內正規化）", fontsize=13, color=TEXT, loc="left", pad=10)
    ax_l.set_yticks([])
    ax_l.set_xticks([0, 0.5, 1])
    ax_l.tick_params(colors=MUTED, labelsize=11, length=0)
    ax_l.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax_l.spines["bottom"].set_bounds(0, 1)
    for s in ax_l.spines.values():
        s.set_visible(False)
    ax_l.scatter([], [], s=90, color=ORANGE, label=f"綜合口碑：{name_a}")
    ax_l.scatter([], [], s=90, color=BLUE, label=f"指定 VRAM：{name_b}")
    ax_l.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.07), ncol=2, frameon=False, fontsize=12,
                handletextpad=0.3, columnspacing=1.6)

    bench = [float(gpu_a.specs.get("benchmark") or 0), float(gpu_b.specs.get("benchmark") or 0)]
    ax_r.bar([0, 1], bench, width=0.55, color=[ORANGE, BLUE], zorder=2)
    for x, v in enumerate(bench):
        ax_r.text(x, v + 500, f"{v:,.0f}", ha="center", va="bottom", fontsize=12, color=TEXT)
    ax_r.text(1, bench[1] / 2, f"{bench[1] / bench[0] - 1:+.0%}".replace("-", "−"), ha="center", va="center",
              fontsize=14, color="white", fontweight="bold")
    ax_r.set_xticks([0, 1], [name_a, name_b], fontsize=12, color=TEXT)
    ax_r.set_yticks([])
    ax_r.set_ylim(0, max(bench) * 1.15)
    ax_r.set_title("顯卡跑分（PassMark）", fontsize=13, color=TEXT, loc="left", pad=10)
    ax_r.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax_r.spines[s].set_visible(False)
    ax_r.spines["bottom"].set_color(GRID)


def main(args):
    stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        catalog, scorer, checker = PartCatalog(DB_PATH), SentimentScorer(MATCHED_FILES, DB_PATH), CompatibilityChecker()
        ga0, base = run_ga(catalog, scorer, checker, None)
        _, picked = run_ga(catalog, scorer, checker, PREFS)
    finally:
        sys.stdout.close()
        sys.stdout = stdout

    plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    bg = "none" if args.bg == "transparent" else "white"
    out_dir = HERE / "results" / "v2_nocp"
    suffix = "" if args.bg == "transparent" else "_white"

    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.patch.set_facecolor(bg)
    plot_overall(ax, ga0, base, scorer)
    fig.savefig(out_dir / f"result_overall{suffix}.png", dpi=220, bbox_inches="tight",
                facecolor=bg, transparent=args.bg == "transparent")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.12})
    fig.patch.set_facecolor(bg)
    for ax in (ax_l, ax_r):
        ax.set_facecolor(bg)
    plot_aspect(ax_l, ax_r, scorer, base.parts["GPU"], picked.parts["GPU"], PREFS["GPU"])
    fig.savefig(out_dir / f"result_aspect{suffix}.png", dpi=220, bbox_inches="tight",
                facecolor=bg, transparent=args.bg == "transparent")

    print(f"綜合口碑：{base.total_price:,}　指定面向：{picked.total_price:,}")
    for cat in ORDER:
        if cat in base.parts or cat in picked.parts:
            a, b = base.parts.get(cat), picked.parts.get(cat)
            mark = "" if a and b and a.name == b.name else "  ← 變動"
            print(f"  {cat}\t{short(a) if a else '-'}\t{short(b) if b else '-'}{mark}")
    print(f"輸出：{out_dir}/result_overall{suffix}.png、result_aspect{suffix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", choices=["transparent", "white"], default="transparent")
    main(ap.parse_args())
