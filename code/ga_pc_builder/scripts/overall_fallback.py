#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
對照實驗：把「總體」當作面向口碑的後備。不改正式資料，「接總體」的分數只存在這支程式裡。

現行：評價不到 MIN_MENTIONS 則的面向，放在類別平均的位置（類別內 0~1）
實驗：往「這個型號總體評價」的位置靠，總體則數越多靠得越多：
  總體分數 = (總體正面 + K × 類別總體正面比例) / (總體正負 + K)，再在類別內 min-max 得到總體位置
  w = 總體則數 / (總體則數 + K_FALLBACK)
  新位置 = 類別平均位置 × (1 - w) + 總體位置 × w
  總體自己不到 MIN_MENTIONS 則的型號不補，維持現行做法；評價夠的面向一律不動。

在 0~1 的位置上處理而不是在原始比例上靠攏：各面向的基準差很多（顯卡穩定平均 0.23、溫度 0.75），
直接往總體比例靠會把穩定拉高、溫度拉低。
不加權（K_FALLBACK=0）試過：同一個極端位置一次補進 4~5 個面向，十幾則總體就讓型號跳到第一
（5600GT 總體 10 正 1 負，CPU 第 13 名 → 第 1 名；A520M 掉到 0 分），所以要照則數加權。

執行（在 code/ga_pc_builder/ 底下）：
  PYTHONHASHSEED=0 python scripts/overall_fallback.py stats        # 總體標註的統計
  PYTHONHASHSEED=0 python scripts/overall_fallback.py scores       # 比較零件口碑分數
  PYTHONHASHSEED=0 python scripts/overall_fallback.py ga --seeds 20
輸出：results/overall_fallback/
"""
import os
import sys
import csv
import argparse
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent
BERT = ROOT / "bert_train"
OUT_DIR = HERE / "results" / "overall_fallback"
sys.path[:0] = [str(HERE), str(HERE / "scripts"), str(BERT), str(ROOT / "src" / "filter")]

from config import DB_PATH, MATCHED_FILES  # noqa: E402
from data.sentiment import SentimentScorer, MIN_MENTIONS  # noqa: E402
import sentiment_ablation as sa  # noqa: E402

K = 10
K_FALLBACK = float(os.environ.get("K_FALLBACK", 30))
WATCH = ["CPU", "GPU", "主機板"]
UNMENTIONED = "未提及"


def scoring_rows():
    """和 build_part_sentiment.py 同一批評論（相關、非高風險、非發售前），附上總體標註"""
    import build_part_sentiment as bps
    from exclude_prerelease import days_before_release
    rows = bps.load_jsonl(BERT / "aspects_標註_p3.jsonl")
    overall = {x["_index"]: x["總體"] for x in bps.load_jsonl(BERT / "aspects_總體_p4.jsonl")}
    risk = bps.flag_high_risk(rows)
    return [(r, overall.get(i, UNMENTIONED)) for i, r in enumerate(rows)
            if r["orig_relevant"] and id(r) not in risk and days_before_release(r) is None]


def overall_counts():
    """{(類別, 型號): [總體正面, 總體負面]}，同晶片的型號共用一份"""
    import build_part_sentiment as bps
    counts = defaultdict(lambda: [0, 0])
    for r, label in scoring_rows():
        if label != UNMENTIONED:
            key = (r["category"], bps.SAME_CHIP.get(r["model"], r["model"]))
            counts[key][0 if label == "正面" else 1] += 1
    for alias, chip in bps.SAME_CHIP.items():
        for cat in {c for c, _ in counts}:
            if (cat, chip) in counts:
                counts[(cat, alias)] = counts[(cat, chip)]
    return dict(counts)


class OverallFallbackScorer(SentimentScorer):
    def __init__(self, *a, overall=None, **kw):
        super().__init__(*a, **kw)
        db = self._db_models(DB_PATH)
        pooled = defaultdict(lambda: [0, 0])
        for (cat, _), (p, n) in overall.items():
            pooled[cat][0] += p
            pooled[cat][1] += n
        rate = {c: p / (p + n) for c, (p, n) in pooled.items()}
        raw = {k: (p + K * rate[k[0]]) / (p + n + K) for k, (p, n) in overall.items()}
        enough = {k for k, (p, n) in overall.items() if p + n >= MIN_MENTIONS}
        bounds = defaultdict(list)
        for k in enough:
            if not db or k in db:
                bounds[k[0]].append(raw[k])
        self.overall_pos = {}
        for k in enough:
            vals = bounds.get(k[0], [])
            if len(vals) >= 2 and max(vals) > min(vals):
                self.overall_pos[k] = max(0.0, min((raw[k] - min(vals)) / (max(vals) - min(vals)), 1.0))
        self.filled = {}
        for key in self.aspects:
            pos = self.overall_pos.get(key)
            if pos is None:
                continue
            sparse = [a for a in self.aspects[key] if self.sample_counts[key][a] < MIN_MENTIONS]
            n = sum(overall[key])
            w = n / (n + K_FALLBACK) if K_FALLBACK else 1.0
            avg = self.category_aspects.get(key[0], {})
            for a in sparse:
                self.aspects[key][a] = avg.get(a, 0.5) * (1 - w) + pos * w
            self.filled[key] = len(sparse)
            self.scores[key] = sum(self.aspects[key].values()) / len(self.aspects[key])


def quiet(fn, *a, **kw):
    stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout.close()
        sys.stdout = stdout


def make_scorer(variant, overall):
    if variant == "現行":
        return quiet(SentimentScorer, MATCHED_FILES, DB_PATH)
    return quiet(OverallFallbackScorer, MATCHED_FILES, DB_PATH, overall=overall)


def show_stats():
    labels, overlap, agree = Counter(), Counter(), Counter()
    by_cat, by_src = defaultdict(Counter), defaultdict(Counter)
    for r, label in scoring_rows():
        labels[label] += 1
        by_cat[r["category"]][label] += 1
        by_src[r["source"]][label] += 1
        # CP值 不算進口碑分數（build_part_sentiment.EXCLUDED_ASPECTS），這裡一樣不算
        rated = {v for a, v in r["aspects"].items() if v != UNMENTIONED and a != "CP值"}
        has = label != UNMENTIONED
        overlap["總體有、面向有" if has and rated else "只有總體" if has
                else "只有面向" if rated else "都沒有"] += 1
        if has and rated:
            agree["一致" if rated == {label} else "混合" if label in rated else "相反"] += 1
    print("算分用的評論", sum(labels.values()), dict(labels))
    print("重疊", dict(overlap))
    print("總體與面向同時有評價時", dict(agree))
    print("來源", {k: dict(v) for k, v in by_src.items()})
    for cat, c in sorted(by_cat.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {cat} {dict(c)}")

    overall = overall_counts()
    old, new = make_scorer("現行", overall), make_scorer("接總體", overall)
    db = old._db_models(DB_PATH)
    per = defaultdict(lambda: Counter())
    for k in old.aspects:
        if k not in db:
            continue
        p = per[k[0]]
        p["型號"] += 1
        p["總體達門檻"] += k in new.overall_pos
        for n in old.sample_counts[k].values():
            p["型號×面向"] += 1
            p["不足10則"] += n < MIN_MENTIONS
            p["可補"] += n < MIN_MENTIONS and k in new.overall_pos
    print("類別覆蓋（GA 資料庫裡的型號）")
    for cat, p in per.items():
        print(f"  {cat} {dict(p)}")
    total = sum(per.values(), Counter())
    print("  合計", dict(total))


def compare_scores():
    overall = overall_counts()
    old, new = make_scorer("現行", overall), make_scorer("接總體", overall)
    db = old._db_models(DB_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for cat in WATCH:
        keys = sorted(k for k in old.scores if k[0] == cat and k in db)
        rank_old = {k: i for i, k in enumerate(sorted(keys, key=lambda k: -old.scores[k]), 1)}
        rank_new = {k: i for i, k in enumerate(sorted(keys, key=lambda k: -new.scores[k]), 1)}
        sparse = [k for k in keys if sum(n >= MIN_MENTIONS for n in old.sample_counts[k].values()) <= 1]
        print(f"\n== {cat}：{len(keys)} 個型號，可用總體補的 {sum(k in new.overall_pos for k in keys)} 個")
        if len(sparse) >= 2:
            print(f"   稀疏型號（最多一個面向達 {MIN_MENTIONS} 則）{len(sparse)} 個，標準差 "
                  f"{statistics.pstdev(old.scores[k] for k in sparse):.3f} → "
                  f"{statistics.pstdev(new.scores[k] for k in sparse):.3f}")
        for k in sorted(keys, key=lambda k: rank_new[k]):
            p, n = overall.get(k, (0, 0))
            out.append({"category": cat, "model": k[1], "overall_pos": p, "overall_neg": n,
                        "filled": new.filled.get(k, 0), "score_old": round(old.scores[k], 4),
                        "score_new": round(new.scores[k], 4), "rank_old": rank_old[k],
                        "rank_new": rank_new[k]})
            print(f"   {k[1]:<26}{p}/{n}  補 {new.filled.get(k, 0)}  "
                  f"{old.scores[k]:.2f} → {new.scores[k]:.2f}  {rank_old[k]} → {rank_new[k]}")
    with open(OUT_DIR / "part_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


def init_worker(variant):
    sa._res["catalog"] = quiet(sa.PartCatalog, DB_PATH)
    sa._res["scorer"] = make_scorer(variant, overall_counts())
    sa._res["checker"] = sa.CompatibilityChecker()


def run_one(job):
    return sa.run_one(job)


def run_ga(args):
    conditions = [("有口碑", None), ("指定 GPU:效能", {"GPU": ["效能"]})]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in ("現行", "接總體"):
        jobs = [{"usage": "遊戲", "budget": b, "condition": c, "seed": s, "weights": None, "prefs": p,
                 "measure": [("GPU", "效能")], "share": sa.PREF_TERM_SHARE, "pref_share": sa.PREF_SHARE}
                for b in args.budgets for c, p in conditions for s in range(args.seeds)]
        print(f"{variant}：{len(jobs)} 次 GA", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(variant,)) as pool:
            rows = list(pool.map(run_one, jobs))
        with open(OUT_DIR / f"ga_{variant}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stats", "scores", "ga"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budgets", type=int, nargs="+", default=[30000, 40000, 60000, 80000])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    {"stats": show_stats, "scores": compare_scores, "ga": lambda: run_ga(args)}[args.mode]()
