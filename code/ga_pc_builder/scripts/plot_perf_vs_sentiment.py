#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
效能口碑 vs PassMark 跑分的散布圖：評論是否帶有規格以外的資訊。

左右兩張小圖（顯卡、CPU），兩者跑分尺度不同所以分開畫。每個點是一個型號：
橫軸是資料庫裡該型號各商品的跑分中位數，縱軸是口碑檔裡的「效能」面向分數
（已往類別平均修正，未做類別內正規化）。只放效能面向正負評價達 10 則的型號。
標題附 Spearman 排名相關係數。

執行（在 code/ga_pc_builder/ 底下）：
  python scripts/plot_perf_vs_sentiment.py
  python scripts/plot_perf_vs_sentiment.py --bg white
輸出：results/v2_nocp/perf_vs_sentiment.png
"""
import json
import argparse
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
TEXT, MUTED, GRID = "#1a1a19", "#52514e", "#c9c7c2"
PANELS = [("GPU", "GPU", "顯卡", "#2B5C8F"), ("CPU", "CPU", "CPU", "#D95F43")]
# 想在圖上點名的型號（說明「跑分高不等於效能口碑好」的例子）
# 值是標籤相對點的位移（points）與對齊方式，逐一指定避免標籤互相重疊
CALLOUTS = {"GPU": {"RTX5080": (-10, -40, "right"), "RTX5070Ti": (-10, -18, "right"), "RX9070": (9, 3, "left"),
                    "RX9060XT": (-10, 9, "right"), "GT710": (9, 5, "left")},
            "CPU": {"AMD R7 9700X": (9, 4, "left"), "AMD R5 7500F": (-10, 9, "right"),
                    "AMD R7 7800X3D": (30, 30, "left"), "AMD R9 9950X3D": (-10, -18, "right"),
                    "Intel Core Ultra 9 285K": (-10, -4, "right")}}
MIN_MENTIONS = 10


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):   # 同分取平均名次
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def load(db_cat, cat):
    db = json.load(open(ROOT / "data" / "ga_database_v2.json", encoding="utf-8"))
    sent = json.load(open(HERE / "data" / "part_aspect_sentiment.json", encoding="utf-8"))
    bench = {}
    for item in db[db_cat]:
        if item.get("benchmark"):
            bench.setdefault(item["ptt_model"], []).append(float(item["benchmark"]))
    pts = []
    for model, bs in bench.items():
        s = sent.get(f"{cat}|{model}")
        if not s:
            continue
        a = s["aspects"]["效能"]
        if a["正面"] + a["負面"] < MIN_MENTIONS:
            continue
        pts.append((model, st.median(bs), a["score"], a["正面"], a["負面"]))
    return pts


def short(model):
    return (model.replace("Intel Core Ultra ", "Ultra ").replace("AMD ", "").replace("Intel ", ""))


def main(args):
    plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"wspace": 0.22})
    bg = "none" if args.bg == "transparent" else "white"
    fig.patch.set_facecolor(bg)

    for ax, (db_cat, cat, title, color) in zip(axes, PANELS):
        pts = load(db_cat, cat)
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        rho = spearman(xs, ys)
        ax.set_facecolor("none")
        ax.scatter(xs, ys, s=70, color=color, edgecolor="white", linewidth=1.4, zorder=3)
        ax.grid(color=GRID, lw=0.8, alpha=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=11, length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        ax.set_ylim(0.4, 0.95)
        ax.set_xlabel("PassMark 跑分", color=MUTED, fontsize=12)
        if cat == "GPU":
            ax.set_ylabel("效能口碑（修正後的正面比例）", color=MUTED, fontsize=12)
        verdict = "中度相關" if rho >= 0.4 else "幾乎無關"
        ax.set_title(f"{title}　Spearman ρ = {rho:.2f}（{verdict}）", loc="left", color=TEXT, fontsize=14,
                     fontweight="bold", pad=26)
        ax.text(0.0, 1.015, f"{len(pts)} 個型號，效能面向評價 ≥ {MIN_MENTIONS} 則", transform=ax.transAxes,
                fontsize=10.5, color=MUTED, va="bottom")
        # 點名幾個例子：點加一圈外框，標籤位置逐一指定
        for model, x, y, pos, neg in pts:
            if model not in CALLOUTS[cat]:
                continue
            dx, dy, ha = CALLOUTS[cat][model]
            ax.scatter([x], [y], s=150, facecolor="none", edgecolor=TEXT, linewidth=1.1, zorder=4)
            ax.annotate(f"{short(model)}（{pos} 正 {neg} 負）", (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=10.5, color=TEXT,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8, "shrinkA": 2, "shrinkB": 7})

    out = HERE / "results" / "v2_nocp" / f"perf_vs_sentiment{'' if args.bg == 'transparent' else '_white'}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=bg, transparent=args.bg == "transparent")
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", choices=["transparent", "white"], default="transparent")
    main(ap.parse_args())
