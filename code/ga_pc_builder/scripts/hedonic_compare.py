#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
非跑分零件的效能：價格 vs hedonic 規格價值，對照實驗。

同一批種子、同樣的預算與用途，只換 perf_basis，看選到的零件、總價、跑分、
各類別花的錢有什麼不同。成對比較（同種子），差異才能歸因於 perf_basis。

執行（在 code/ga_pc_builder/ 底下）：
  PYTHONHASHSEED=0 python scripts/hedonic_compare.py --budgets 30000 40000 60000 80000 --seeds 10
輸出：
  畫面摘要，以及 results/hedonic_compare_<用途>.csv（每次 GA 一列）
"""
import os
import sys
import csv
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from config import DB_PATH, MATCHED_FILES  # noqa: E402
from data.catalog import PartCatalog  # noqa: E402
from data.sentiment import SentimentScorer  # noqa: E402
from data.hedonic import attach_spec_values, attach_efficiency  # noqa: E402
from advisor.compatibility import CompatibilityChecker  # noqa: E402
from core.ga_engine import GARecommender  # noqa: E402

CATS = ["CPU", "GPU", "主機板", "記憶體", "SSD", "電源", "機殼", "風冷"]
_res = {}


FRONTIER = os.environ.get("HEDONIC_FRONTIER", "vrs")   # vrs（主要）或 fdh（敏感度分析）


def _init():
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        cat = PartCatalog(DB_PATH)
        attach_spec_values(cat, verbose=False)   # 寫 spec_value；perf_basis=price 時用不到
        attach_efficiency(cat, FRONTIER)         # 寫 efficiency（Farrell/BCC）與 quality
        _res["catalog"] = cat
        _res["scorer"] = SentimentScorer(MATCHED_FILES, DB_PATH)
        _res["checker"] = CompatibilityChecker()


def run_one(job):
    if not _res:
        _init()
    random.seed(job["seed"]); np.random.seed(job["seed"])
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ga = GARecommender(catalog=_res["catalog"], scorer=_res["scorer"], checker=_res["checker"],
                           usage=job["usage"], budget=job["budget"], pop_size=job["pop"],
                           generations=job["gens"], perf_basis=job["basis"],
                           cp_mode=job["cp"])
        best = ga.run(verbose=False)[0]
        # 兩種效能定義都算一次，才能互相比（同一組配單在兩種尺度下的分數）
        ga_p = GARecommender(catalog=_res["catalog"], scorer=_res["scorer"], checker=_res["checker"],
                             usage=job["usage"], budget=job["budget"], perf_basis="price")
        ga_h = GARecommender(catalog=_res["catalog"], scorer=_res["scorer"], checker=_res["checker"],
                             usage=job["usage"], budget=job["budget"], perf_basis="hedonic", cp_mode="dea")
    row = {"basis": job["basis"], "cp": job["cp"], "budget": job["budget"], "seed": job["seed"],
           "total": best.total_price, "fitness": round(ga.fitness(best), 4),
           "perf_price": round(ga_p._perf_score(best), 4),
           "perf_hedonic": round(ga_h._perf_score(best), 4),
           "cp_price": round(ga_p._cp_score(best), 4),
           # 共同尺度：不論用哪種設定跑出來，都用同一把尺量
           "cost_eff": round(ga_h._cp_dea(best), 4),           # Farrell 成本效率（整機）
           "value_ratio": round(ga_h._cp_value(best) + 0.5, 3),  # Σ規格價值/Σ價格
           "sentiment": round(ga_h._sentiment_score(best), 4),
           "secondary": sum(best.parts[c].price for c in ("機殼", "風冷", "電源") if c in best.parts),
           "cpu_bench": best.parts["CPU"].specs.get("benchmark"),
           "gpu_bench": best.parts["GPU"].specs.get("benchmark")}
    for c in CATS:
        p = best.parts.get(c)
        row[f"{c}_name"] = p.name[:40] if p else ""
        row[f"{c}_price"] = p.price if p else 0
        sv = p.specs.get("spec_value") if p else None
        row[f"{c}_premium"] = round(p.price / sv, 2) if sv else ""
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[30000, 40000, 60000, 80000])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--usage", default="遊戲")
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--configs", nargs="+", default=["price:ratio", "hedonic:ratio"],
                    help="perf_basis:cp_mode，例如 price:ratio hedonic:noUnder hedonic:hedonic")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    jobs = [dict(basis=cfg.split(":")[0], cp=cfg.split(":")[1], budget=bud, seed=s,
                 usage=a.usage, pop=a.pop, gens=a.gens)
            for cfg in a.configs for bud in a.budgets for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        rows = list(ex.map(run_one, jobs))

    out = HERE / "results" / f"hedonic_compare_{a.usage}{a.tag}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"寫入 {out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
