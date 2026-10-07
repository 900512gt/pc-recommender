#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
口碑對照實驗：同樣的預算與用途，比較「不看口碑／看口碑／使用者指定面向」選到的零件。

GA 有隨機性，同一組設定換亂數種子，選到的 CPU、顯卡就可能不同，所以每個條件都跑
多個種子，看的是分布與平均，不是單次結果。

同一次執行裡各條件用的是同一批種子，可以互相比。但兩次執行之間結果不會完全相同：
GA 內部有依賴 set 走訪順序的隨機挑選，而字串的雜湊值每次啟動 Python 都不同。
要重現同一份數字，執行時固定 PYTHONHASHSEED（見下方用法）。

條件：
  無口碑      口碑滑桿設 0（效能與 CP值 平分原本三項的份額）
  有口碑      預設（效能／口碑／CP值 等權），各面向平均
  指定面向    例如 GPU:穩定 —— 顯卡的口碑加重穩定（佔一半）

執行（在 code/ga_pc_builder/ 底下）：
  PYTHONHASHSEED=0 python scripts/sentiment_ablation.py
  python scripts/sentiment_ablation.py --budgets 40000 60000 --seeds 20 --prefs GPU:穩定 GPU:VRAM CPU:效能
"""
import sys
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, MATCHED_FILES  # noqa: E402
from data.catalog import PartCatalog  # noqa: E402
from data.sentiment import SentimentScorer  # noqa: E402
from advisor.compatibility import CompatibilityChecker  # noqa: E402
from core.ga_engine import GARecommender  # noqa: E402


def parse_pref(text):
    """'GPU:穩定' → ('GPU', '穩定')"""
    category, aspect = text.split(":", 1)
    return category, aspect


def run(catalog, scorer, checker, usage, budget, seed, weights=None, prefs=None):
    random.seed(seed)
    np.random.seed(seed)
    ga = GARecommender(catalog=catalog, scorer=scorer, checker=checker, usage=usage, budget=budget,
                       pop_size=300, generations=300, psu_tier="standard",
                       custom_weights=weights, aspect_prefs=prefs)
    best = ga.run(verbose=False)[0]
    return ga, best


def main(args):
    catalog = PartCatalog(DB_PATH)
    scorer = SentimentScorer(MATCHED_FILES, DB_PATH)
    checker = CompatibilityChecker()

    prefs = [parse_pref(p) for p in args.prefs]
    conditions = [("無口碑", {"perf": 50, "sent": 0, "cp": 50}, None), ("有口碑", None, None)]
    conditions += [(f"指定 {c}:{a}", None, {c: [a]}) for c, a in prefs]
    watch = sorted({c for c, _ in prefs} | {"GPU", "CPU"})   # 觀察哪些類別選到什麼

    for budget in args.budgets:
        print(f"\n{'=' * 78}\n{args.usage}　預算 {budget}　每個條件 {args.seeds} 個種子\n{'=' * 78}")
        header = f"{'條件':<14}{'總價':>8}{'效能':>7}{'整機口碑':>9}"
        header += "".join(f"{c + ':' + a:>10}" for c, a in prefs)
        print(header)
        picks = {}
        for name, weights, pref in conditions:
            rows = []
            picks[name] = {c: Counter() for c in watch}
            for seed in range(args.seeds):
                ga, best = run(catalog, scorer, checker, args.usage, budget, seed, weights, pref)
                row = [best.total_price, ga._perf_score(best),
                       # 整機口碑一律用「各面向平均」算，條件之間才能比
                       float(np.mean([scorer.get(c, p.short_name) for c, p in best.parts.items()]))]
                # 選到的零件在各個受觀察面向的分數（類別內正規化後，0 最差、1 最好）
                row += [scorer.get_aspects(c, best.parts[c].short_name)[a] for c, a in prefs]
                rows.append(row)
                for c in watch:
                    picks[name][c][best.parts[c].specs.get("ptt_model", "?")] += 1
            mean = np.mean(rows, axis=0)
            line = f"{name:<14}{mean[0]:>10.0f}{mean[1]:>9.3f}{mean[2]:>11.3f}"
            line += "".join(f"{v:>12.2f}" for v in mean[3:])
            print(line)

        for c in watch:
            print(f"\n  {c} 選到的型號（次數）")
            for name in picks:
                top = "、".join(f"{m} {n}" for m, n in picks[name][c].most_common(5))
                print(f"    {name:<14}{top}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", default="遊戲")
    ap.add_argument("--budgets", type=int, nargs="+", default=[40000, 60000])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--prefs", nargs="*", default=["GPU:穩定", "GPU:VRAM"],
                    help="要測的面向偏好，格式 類別:面向")
    main(ap.parse_args())
