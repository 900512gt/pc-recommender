#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
口碑對照實驗：同樣的預算與用途，比較「不看口碑／看口碑／使用者指定面向」選到的零件。

GA 有隨機性，同一組設定換亂數種子，選到的 CPU、顯卡就可能不同，所以每個條件都跑
多個種子，看的是分布與平均，不是單次結果。

各條件用的是同一批種子，「指定面向」的第 7 號種子直接跟「有口碑」的第 7 號種子比
（成對比較），差異才能歸因於面向偏好而不是運氣。要讓兩次執行的數字完全相同，
執行時要固定 PYTHONHASHSEED：GA 內部有依賴 set 走訪順序的隨機挑選，而字串的
雜湊值每次啟動 Python 都不同。

條件：
  無口碑      口碑滑桿設 0（效能與 CP值 平分原本三項的份額）
  有口碑      預設（效能／口碑／CP值 等權），各面向平均 —— 指定面向的對照組
  指定面向    例如 GPU:穩定 —— 顯卡的口碑加重穩定

對每個指定面向回答三個問題（都是相對「有口碑」）：
  有沒有換型號    幾個種子選到的該類別型號不一樣
  有沒有變好      選到的零件在該面向的分數、名次
  付出什麼代價    整機效能分數與總價的變化

執行（在 code/ga_pc_builder/ 底下）：
  PYTHONHASHSEED=0 python scripts/sentiment_ablation.py
  PYTHONHASHSEED=0 python scripts/sentiment_ablation.py --budgets 30000 40000 60000 \\
      --usage 遊戲 --seeds 20 --prefs all
輸出：
  畫面上的摘要，以及 results/sentiment_ablation_<用途>_term<比例>_part<比例>.csv
  （每次 GA 一列，供之後畫圖、重算）

比較指定面向的三種做法：
  --pref-term-share 0 --pref-share 0.5   加重：指定的面向佔該零件口碑的一半
  --pref-term-share 0.5                  獨立成項：從整體口碑的權重分一半給指定的面向
  --pref-term-share 0 --pref-share 1     只看指定：有指定面向的類別，口碑只看指定的面向
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

from config import DB_PATH, MATCHED_FILES, PREF_TERM_SHARE  # noqa: E402
from data.catalog import PartCatalog  # noqa: E402
from data.sentiment import SentimentScorer, PREF_SHARE  # noqa: E402
from advisor.compatibility import CompatibilityChecker  # noqa: E402
from core.ga_engine import GARecommender  # noqa: E402

BASELINE = "有口碑"
NO_SENT = "無口碑"
# 記錄每次選到什麼型號的類別：整機口碑實際會看的三個類別
WATCH = ["CPU", "GPU", "主機板"]

_res = {}   # 每個 worker process 各自載入一份資料


def _init_worker(common_scale=False):
    # 載入訊息每個 worker 都會印一次，實驗輸出裡不需要
    stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        _res["catalog"] = PartCatalog(DB_PATH)
        _res["scorer"] = SentimentScorer(MATCHED_FILES, DB_PATH, common_scale=common_scale)
        _res["checker"] = CompatibilityChecker()
    finally:
        sys.stdout.close()
        sys.stdout = stdout


def run_one(job: dict) -> dict:
    """跑一次 GA，回傳一列結果。job 的 measure 是要量的 (類別, 面向) 清單。"""
    scorer = _res["scorer"]
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    ga = GARecommender(catalog=_res["catalog"], scorer=scorer, checker=_res["checker"],
                       usage=job["usage"], budget=job["budget"],
                       pop_size=300, generations=300, psu_tier="standard",
                       custom_weights=job["weights"], aspect_prefs=job["prefs"],
                       pref_term_share=job["share"], pref_share=job["pref_share"])
    best = ga.run(verbose=False)[0]

    row = {
        "usage": job["usage"], "budget": job["budget"],
        "pref_term_share": job["share"], "pref_share": job["pref_share"],
        "condition": job["condition"], "seed": job["seed"],
        "total_price": best.total_price,
        "perf": round(ga._perf_score(best), 4),
        # 整機口碑一律用「各面向平均」算（不帶面向偏好），條件之間才能比
        "sentiment": round(float(np.mean([scorer.get(c, best.parts[c].short_name) for c in WATCH])), 4),
    }
    for c in WATCH:
        row[f"model_{c}"] = best.parts[c].specs.get("ptt_model") or best.parts[c].name

    # 選到的零件在每個受觀察面向的表現。norm 是 GA 實際看到的分數（類別內 0~1，
    # 評價不足時是類別平均的位置）；rank 只有評價夠的型號才有，其餘留空。
    for c, a in job["measure"]:
        name = best.parts[c].short_name
        row[f"norm_{c}:{a}"] = round(scorer.get_aspects(c, name).get(a, 0.5), 4)
        evidence = scorer.aspect_evidence(c, name)
        hit = next((r for r in (evidence or {}).get("aspects", []) if r["aspect"] == a), None)
        row[f"rank_{c}:{a}"] = hit["rank"] if hit and hit["rank"] is not None else ""
    return row


def summarize(rows: list, budget: int, measure: list, seeds: int):
    by = {(r["condition"], r["seed"]): r for r in rows if r["budget"] == budget}
    conditions = list(dict.fromkeys(r["condition"] for r in rows))

    def col(cond, key):
        return [by[(cond, s)][key] for s in range(seeds)]

    print(f"\n  {'條件':<16}{'總價':>8}{'效能':>8}{'整機口碑':>8}")
    for cond in conditions:
        print(f"  {cond:<16}{np.mean(col(cond, 'total_price')):>10.0f}"
              f"{np.mean(col(cond, 'perf')):>10.3f}{np.mean(col(cond, 'sentiment')):>12.3f}")

    print(f"\n  指定面向 相對「{BASELINE}」（同種子成對比較，{seeds} 個種子）")
    print(f"  {'面向':<12}{'換型號':>6}{'面向分數 對照→指定':>20}{'變好/持平/變差':>16}"
          f"{'名次 對照→指定':>18}{'效能變化':>10}{'總價變化':>10}")
    for c, a in measure:
        cond = f"指定 {c}:{a}"
        if (cond, 0) not in by:
            continue
        base_n, pref_n = col(BASELINE, f"norm_{c}:{a}"), col(cond, f"norm_{c}:{a}")
        changed = sum(b != p for b, p in zip(col(BASELINE, f"model_{c}"), col(cond, f"model_{c}")))
        up = sum(p > b + 1e-9 for b, p in zip(base_n, pref_n))
        down = sum(p < b - 1e-9 for b, p in zip(base_n, pref_n))

        def mean_rank(cond_name):
            # 只平均有名次的種子；有種子選到評價不足的型號時，括號標出實際平均了幾個
            ranks = [r for r in col(cond_name, f"rank_{c}:{a}") if r != ""]
            if not ranks:
                return "—"
            return f"{np.mean(ranks):.1f}" + ("" if len(ranks) == seeds else f"({len(ranks)})")

        perf_change = np.mean(col(cond, "perf")) / np.mean(col(BASELINE, "perf")) - 1
        price_change = np.mean(col(cond, "total_price")) - np.mean(col(BASELINE, "total_price"))
        print(f"  {c + ':' + a:<12}{changed:>5}/{seeds}"
              f"{np.mean(base_n):>14.2f} → {np.mean(pref_n):.2f}"
              f"{f'{up}/{seeds - up - down}/{down}':>20}"
              f"{mean_rank(BASELINE) + ' → ' + mean_rank(cond):>20}"
              f"{perf_change:>+11.1%}{price_change:>+11.0f}")

    for c in WATCH:
        print(f"\n  {c} 選到的型號（次數）")
        for cond in conditions:
            # 指定面向的條件只列自己那個類別，其他類別跟面向無關
            if cond.startswith("指定") and not cond.startswith(f"指定 {c}:"):
                continue
            top = "、".join(f"{m} {n}" for m, n in Counter(col(cond, f"model_{c}")).most_common(5))
            print(f"    {cond:<16}{top}")


def main(args):
    if os.environ.get("PYTHONHASHSEED") is None:
        print("注意：沒有固定 PYTHONHASHSEED，這次的數字下次執行無法完全重現。")

    _init_worker(args.common_scale)
    selectable = _res["scorer"].selectable
    if args.prefs == ["all"]:
        measure = [(c, a) for c in WATCH for a in selectable.get(c, [])]
    else:
        measure = [tuple(p.split(":", 1)) for p in args.prefs]
        bad = [f"{c}:{a}" for c, a in measure if a not in selectable.get(c, [])]
        if bad:
            sys.exit(f"這些面向資料不夠、GA 會直接忽略，測了沒有意義：{bad}\n可指定的面向：{selectable}")

    # --sliders 有給時，「有口碑」與所有「指定面向」都用同一組滑桿，兩者才只差在面向偏好
    sliders = dict(zip(("perf", "sent", "cp"), args.sliders)) if args.sliders else None
    conditions = [(NO_SENT, {"perf": 50, "sent": 0, "cp": 50}, None), (BASELINE, sliders, None)]
    conditions += [(f"指定 {c}:{a}", sliders, {c: [a]}) for c, a in measure]
    jobs = [{"usage": args.usage, "budget": budget, "condition": name, "seed": seed,
             "weights": weights, "prefs": prefs, "measure": measure,
             "share": args.pref_term_share, "pref_share": args.pref_share}
            for budget in args.budgets for name, weights, prefs in conditions
            for seed in range(args.seeds)]

    print(f"指定面向佔口碑權重的比例 {args.pref_term_share}"
          + (f"（不獨立成項：指定的面向佔該零件口碑的 {args.pref_share}）"
             if args.pref_term_share == 0 else ""))
    if sliders:
        print(f"滑桿（效能／口碑／CP值）{args.sliders}")
    print(f"{args.usage}　預算 {args.budgets}　{len(conditions)} 個條件 × {args.seeds} 個種子"
          f" = {len(jobs)} 次 GA（{args.workers} 個 process）")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.common_scale,)) as pool:
        rows = []
        for i, row in enumerate(pool.map(run_one, jobs), 1):
            rows.append(row)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    name = f"sentiment_ablation_{args.usage}_term{args.pref_term_share}_part{args.pref_share}"
    if args.common_scale:
        name += "_commonscale"
    if sliders:
        name += "_sliders" + "-".join(f"{v:g}" for v in args.sliders)
    out = Path(args.out) if args.out else HERE / "results" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for budget in args.budgets:
        print(f"\n{'=' * 100}\n{args.usage}　預算 {budget}\n{'=' * 100}")
        summarize(rows, budget, measure, args.seeds)
    print(f"\n每次 GA 的原始結果：{out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", default="遊戲")
    ap.add_argument("--budgets", type=int, nargs="+", default=[40000])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--prefs", nargs="*", default=["all"],
                    help="要測的面向偏好，格式 類別:面向；all 表示所有可指定的面向")
    ap.add_argument("--pref-term-share", type=float, default=PREF_TERM_SHARE,
                    help="指定面向佔口碑權重的比例；0 是舊做法（混進零件口碑，不獨立成項）")
    ap.add_argument("--pref-share", type=float, default=PREF_SHARE,
                    help="不獨立成項時，指定的面向佔該零件口碑的比例；1 表示只看指定的面向")
    ap.add_argument("--common-scale", action="store_true",
                    help="同類別的面向共用一把尺做正規化，而不是每個面向各自 min-max")
    ap.add_argument("--sliders", type=float, nargs=3, default=None, metavar=("效能", "口碑", "CP值"),
                    help="前端三個滑桿的值，例如 33 66 33；不給就是預設的等權")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="同時跑幾個 GA；結果與 process 數無關")
    ap.add_argument("--out", default=None, help="CSV 輸出路徑")
    main(ap.parse_args())
