"""
從 annotated.jsonl 抽出 PTT「不相關」推文，當 BERT 訓練的「全部未提及」負例，
數量與 PTT 相關評論 1:1（巴哈那邊相關 10,602、不相關 10,848，也是約 1:1）。

1. 候選：annotate.py 判 relevant=false 的推文，排除 HDD（相關只有 1 則，面向清單不含）
2. 排除與相關評論衝突的列：同 (型號, 內容) 在 relevant.jsonl 或 relevant_fixed.jsonl 也出現過
   （後者含 fix_relevant.py 改歸屬過來的列）
   （同一句在別的型號底下被判相關的不排除，那正是「在講別的型號」的負例）
3. 去重：(型號, 內容) 相同只留一筆
4. 依類別分層：每個類別抽的數量 = relevant_fixed.jsonl 裡該類別的相關則數
5. 另外從抽出的負例中抽約 100 則，輸出 CSV 供人工核對相關性標註的正確率
   （每個類別至少 3 則，其餘按比例）

用法（專案根目錄）：
    python src/filter/sample_irrelevant.py
    python src/filter/sample_irrelevant.py --review-n 100 --seed 42
    python src/filter/sample_irrelevant.py --postrelease    # 排除發售前的版本（見下）
輸出：
    src/filter/output/irrelevant_sampled.jsonl      ← 負例，欄位與 relevant_fixed.jsonl 相同
    src/filter/output/irrelevant_sample_review.csv  ← 人工核對用

--postrelease：配合 exclude_prerelease.py（兩邊的 --release-csv 要給同一份）。配額改看排除發售前之後的相關評論
（relevant_fixed_postrelease.jsonl），候選也套同一條發售日規則，
輸出到 irrelevant_sampled_postrelease.jsonl，不產生核對清單、不覆蓋上面兩個檔。
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from exclude_prerelease import days_before_release, use_release_csv

ROOT          = Path(__file__).resolve().parents[2]
ANNOTATED     = ROOT / "src/filter/output/annotated.jsonl"
RELEVANT_RAW  = ROOT / "src/database/input/ptt_comment/relevant.jsonl"
RELEVANT_FIX  = ROOT / "src/database/input/ptt_comment/relevant_fixed.jsonl"
OUT_FILE      = ROOT / "src/filter/output/irrelevant_sampled.jsonl"
REVIEW_CSV    = ROOT / "src/filter/output/irrelevant_sample_review.csv"
RELEVANT_POST = ROOT / "src/database/input/ptt_comment/relevant_fixed_postrelease.jsonl"
OUT_FILE_POST = ROOT / "src/filter/output/irrelevant_sampled_postrelease.jsonl"

EXCLUDE_CATS   = {"HDD"}
MIN_PER_CAT    = 3   # 核對清單每個類別至少幾則


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def key(x):
    return (x["model"], x["content"].strip())


def build_pool(annotated, relevant, drop_prerelease=False):
    """不相關候選：排除與相關評論衝突的列、去重，依類別分組"""
    relevant_keys = {key(x) for x in relevant}
    pool = defaultdict(list)
    seen = set()
    dropped = Counter()
    for x in annotated:
        if x["relevant"]:
            continue
        if x["category"] in EXCLUDE_CATS:
            dropped["排除類別"] += 1
            continue
        if drop_prerelease and days_before_release(x):
            dropped["發售前"] += 1
            continue
        k = key(x)
        if k in relevant_keys:
            dropped["與相關評論衝突"] += 1
            continue
        if k in seen:
            dropped["重複"] += 1
            continue
        seen.add(k)
        pool[x["category"]].append(x)
    return pool, dropped


def sample_negatives(pool, quota):
    """每個類別抽 quota[cat] 則；候選不夠就全拿"""
    sampled = []
    for cat in sorted(quota):
        rows = pool.get(cat, [])
        n = min(quota[cat], len(rows))
        sampled.extend(random.sample(rows, n))
    return sampled


def sample_review(sampled, target):
    """從負例中抽核對清單：每個類別至少 MIN_PER_CAT 則，其餘按比例"""
    by_cat = defaultdict(list)
    for x in sampled:
        by_cat[x["category"]].append(x)

    remaining = max(0, target - MIN_PER_CAT * len(by_cat))
    review = []
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        n = MIN_PER_CAT + round(remaining * len(rows) / len(sampled))
        review.extend(random.sample(rows, min(n, len(rows))))
    random.shuffle(review)
    return review


def main():
    parser = argparse.ArgumentParser(description="抽樣 PTT 不相關推文當 BERT 負例")
    parser.add_argument("--review-n", type=int, default=100, help="人工核對則數 (預設 100)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設 42)")
    parser.add_argument("--postrelease", action="store_true", help="排除發售前的版本")
    parser.add_argument("--release-csv", help="發售日改用這份清單，需與 exclude_prerelease.py 用同一份")
    args = parser.parse_args()
    if args.release_csv:
        use_release_csv(args.release_csv)
    random.seed(args.seed)

    annotated = load(ANNOTATED)
    relevant_fixed = load(RELEVANT_FIX)
    pool, dropped = build_pool(annotated, load(RELEVANT_RAW) + relevant_fixed,
                               drop_prerelease=args.postrelease)
    quota_rows = load(RELEVANT_POST) if args.postrelease else relevant_fixed
    quota = Counter(x["category"] for x in quota_rows
                    if x["category"] not in EXCLUDE_CATS)
    out_file = OUT_FILE_POST if args.postrelease else OUT_FILE

    sampled = sample_negatives(pool, quota)
    sampled.sort(key=lambda x: x["_index"])
    with open(out_file, "w", encoding="utf-8") as f:
        for x in sampled:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    review = [] if args.postrelease else sample_review(sampled, args.review_n)
    if review:
        with open(REVIEW_CSV, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["_index", "類別", "型號", "文章標題", "內容", "url",
                        "人工判斷(對=確實不相關 / 錯=其實有評價指定型號)", "備註"])
            for x in review:
                w.writerow([x["_index"], x["category"], x["model"],
                            x["title"], x["content"], x["url"], "", ""])

    got = Counter(x["category"] for x in sampled)
    rev = Counter(x["category"] for x in review)
    print(f"不相關候選 {sum(len(v) for v in pool.values())} 筆"
          f"（{'、'.join(f'{k} {n}' for k, n in dropped.most_common())}）")
    print(f"{'類別':<14}{'相關':>7}{'候選':>8}{'抽出':>7}{'核對':>6}")
    for cat, n in quota.most_common():
        print(f"{cat:<14}{n:>9}{len(pool.get(cat, [])):>10}{got[cat]:>9}{rev[cat]:>8}")
    print(f"{'合計':<14}{sum(quota.values()):>9}{'':>10}{len(sampled):>9}{len(review):>8}")
    print(f"輸出：{out_file.relative_to(ROOT)}")
    if review:
        print(f"核對清單：{REVIEW_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
