"""
從 relevant.jsonl 中分層抽樣約 100 則評論，輸出成 Excel 供人工校注。
策略：每個型號至少 2 則，其餘按比例分配，確保涵蓋所有型號。

用法：
    python sample_for_annotation.py relevant.jsonl [--n 100] [--seed 42] [--output sample.xlsx]
"""

import json
import random
import argparse
from collections import defaultdict, Counter
import pandas as pd


def load_records(path):
    records = []
    by_model = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
            by_model[r["model"]].append(r)
    return records, by_model


def stratified_sample(records, by_model, target=100, min_per_model=2):
    guaranteed = min_per_model * len(by_model)
    remaining = max(0, target - guaranteed)

    sampled = []
    for model, recs in by_model.items():
        base = min(min_per_model, len(recs))
        extra = max(0, round(remaining * len(recs) / len(records)))
        n = min(base + extra, len(recs))
        sampled.extend(random.sample(recs, n))

    # 調整至目標數量
    if len(sampled) > target:
        sampled = random.sample(sampled, target)
    elif len(sampled) < target:
        sampled_indices = {r["_index"] for r in sampled}
        pool = [r for r in records if r["_index"] not in sampled_indices]
        random.shuffle(pool)
        sampled.extend(pool[: target - len(sampled)])

    random.shuffle(sampled)
    return sampled


def to_excel(sampled, output_path):
    df = pd.DataFrame(sampled)
    cols = [
        "_index", "category", "brand", "series", "model",
        "title", "content", "label", "tag", "date", "url",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.rename(columns={"_index": "original_index", "label": "auto_label"})
    df["human_label"] = ""
    df["notes"] = ""
    df.to_excel(output_path, index=False, engine="openpyxl")
    return df


def print_stats(df):
    mc = Counter(df["model"])
    print(f"抽樣完成：共 {len(df)} 則，涵蓋 {len(mc)} 個型號")
    for m, c in sorted(mc.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")


def main():
    parser = argparse.ArgumentParser(description="分層抽樣評論供人工校注")
    parser.add_argument("input", help="relevant.jsonl 路徑")
    parser.add_argument("--n", type=int, default=100, help="抽樣數量 (預設 100)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設 42)")
    parser.add_argument("--output", default="output/sample_for_annotation.xlsx", help="輸出檔名")
    args = parser.parse_args()

    random.seed(args.seed)
    records, by_model = load_records(args.input)
    sampled = stratified_sample(records, by_model, target=args.n)
    df = to_excel(sampled, args.output)
    print_stats(df)
    print(f"已輸出至 {args.output}")


if __name__ == "__main__":
    main()