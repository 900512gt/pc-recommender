#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
口碑權重掃描：口碑在「效能＋口碑＋CP 值」中的佔比由 0 掃到 0.95，看效能與口碑的取捨曲線。

要回答的問題不是「口碑會不會影響決策」（權重是自己設的，答案必然是會），而是：
  - 口碑能不能在不犧牲效能的情況下提升（曲線前段是否平坦）
  - 超過某個佔比後，每多要一點口碑要付出多少效能（曲線的轉折點）

效能與 CP 值平分剩下的佔比；預算與相容性的權重不變（等同前端三個滑桿的換算方式）。
不指定面向。修正後的 GA 同一預算各種子結果一致，每點跑少量種子確認即可。

執行（在 code/ga_pc_builder/ 底下）：
  PYTHONHASHSEED=0 python scripts/sentiment_sweep.py
  PYTHONHASHSEED=0 python scripts/sentiment_sweep.py --budgets 40000 60000 --seeds 3
輸出：results/v2_nocp/sentiment_sweep_<用途>.csv（每次 GA 一列）
"""
import os
import sys
import csv
import random
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from config import DB_PATH, MATCHED_FILES  # noqa: E402
from data.catalog import PartCatalog  # noqa: E402
from data.sentiment import SentimentScorer  # noqa: E402
from advisor.compatibility import CompatibilityChecker  # noqa: E402
from core.ga_engine import GARecommender  # noqa: E402

# 1/3 是前端預設等權；0.35、0.37 是補跑的點，用來定出 6 萬開始換顯卡的位置（雙贏區的上界）
SHARES = [0.0, 0.1, 0.2, 0.3, 1 / 3, 0.35, 0.37, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
WATCH = ["CPU", "GPU", "主機板"]   # 整機口碑實際會看的三個類別
_res = {}


def _init_worker():
    stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        _res["catalog"] = PartCatalog(DB_PATH)
        _res["scorer"] = SentimentScorer(MATCHED_FILES, DB_PATH)
        _res["checker"] = CompatibilityChecker()
    finally:
        sys.stdout.close()
        sys.stdout = stdout


def run_one(job: dict) -> dict:
    scorer = _res["scorer"]
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    share = job["share"]
    # 前端滑桿是相對比重：口碑佔 share，效能與 CP 值平分剩下的
    weights = {"perf": (1 - share) / 2 * 100, "sent": share * 100, "cp": (1 - share) / 2 * 100}
    ga = GARecommender(catalog=_res["catalog"], scorer=scorer, checker=_res["checker"],
                       usage=job["usage"], budget=job["budget"], pop_size=300, generations=300,
                       psu_tier="standard", custom_weights=weights)
    best = ga.run(verbose=False)[0]
    row = {
        "usage": job["usage"], "budget": job["budget"], "sent_share": share, "seed": job["seed"],
        "total_price": best.total_price,
        "perf": round(ga._perf_score(best), 4),
        "cp": round(ga._cp_score(best), 4),
        # 整機口碑：CPU、GPU、主機板的型號口碑平均（類別內正規化後，與 fitness 用的相同）
        "sentiment": round(float(np.mean([scorer.get(c, best.parts[c].short_name) for c in WATCH])), 4),
        "fitness": round(ga.fitness(best), 4),
    }
    for c in WATCH:
        part = best.parts[c]
        row[f"model_{c}"] = part.specs.get("ptt_model") or part.name
        row[f"price_{c}"] = part.price
        row[f"sent_{c}"] = round(scorer.get(c, part.short_name), 4)
    for c in ("CPU", "GPU"):
        row[f"bench_{c}"] = float(best.parts[c].specs.get("benchmark") or 0)
    return row


def summarize(rows, budget):
    print(f"\n預算 {budget:,}")
    print(f"  {'口碑佔比':>6} {'整機口碑':>8} {'效能':>6} {'CP值':>6} {'GPU跑分':>8} {'CPU跑分':>8} {'總價':>8}  CPU / GPU / 主機板（種子一致數）")
    for share in SHARES:
        rs = [r for r in rows if r["budget"] == budget and r["sent_share"] == share]
        if not rs:
            continue
        picks = Counter((r["model_CPU"], r["model_GPU"], r["model_主機板"]) for r in rs)
        (cpu, gpu, mb), n = picks.most_common(1)[0]
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"  {share:>6.0%} {m('sentiment'):>8.3f} {m('perf'):>6.3f} {m('cp'):>6.3f} {m('bench_GPU'):>8,.0f} "
              f"{m('bench_CPU'):>8,.0f} {m('total_price'):>8,.0f}  {cpu} / {gpu} / {mb}（{n}/{len(rs)}）")


def main(args):
    if os.environ.get("PYTHONHASHSEED") is None:
        print("注意：沒有固定 PYTHONHASHSEED，這次的數字下次執行無法完全重現。")
    jobs = [{"usage": args.usage, "budget": b, "share": s, "seed": seed}
            for b in args.budgets for s in SHARES for seed in range(args.seeds)]
    print(f"{args.usage}　預算 {args.budgets}　{len(SHARES)} 個口碑佔比 × {args.seeds} 個種子 = {len(jobs)} 次 GA")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        rows = []
        for i, row in enumerate(pool.map(run_one, jobs), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / "v2_nocp" / f"sentiment_sweep_{args.usage}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for b in args.budgets:
        summarize(rows, b)
    print(f"\n每次 GA 的原始結果：{out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", default="遊戲")
    ap.add_argument("--budgets", type=int, nargs="+", default=[30000, 40000, 60000, 80000])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
