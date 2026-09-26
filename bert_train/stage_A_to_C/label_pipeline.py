# label_pipeline.py — 阶段A：用 GPT 产生面向标注资料（给 BERT 训练用）
#
# 用法：
#   python label_pipeline.py --sample 50      # 小样本测试（标 50 则看品质）
#   python label_pipeline.py --full           # 全部标注
#   python label_pipeline.py --sample 50 --dry # 不呼叫 API，看流程
#
# 输出：absa_labeled.jsonl
#   每行一则评论 + 面向标注，格式给 BERT 训练用

import os
import json
import argparse
from collections import defaultdict

from aspects import CORE_ASPECTS
from prompt_builder import build_labeling_prompt


# ── GPT API 呼叫 ─────────────────────────────────────────────────────────────

def call_gpt(prompt: str, model: str = "gpt-4o-mini") -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("請先安裝：pip install openai")
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是專業的電腦硬體評論標註員，只回傳 JSON。"},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def parse_response(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for key in ("reviews", "results", "data", "items", "labels"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return []


# ── 资料载入 ─────────────────────────────────────────────────────────────────

def load_reviews(paths: list[str], min_len: int = 5) -> list[dict]:
    """载入所有评论，保留原始 label 供后续比对 GPT 标注品质。"""
    reviews = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                content = r.get("content", "").strip()
                if len(content) < min_len:
                    continue
                reviews.append({
                    "category": r.get("category", ""),
                    "model":    r.get("model", ""),
                    "content":  content,
                    "orig_label": r.get("label", ""),   # 原本的正/负/中立标注
                    "GP": r.get("GP", 0),
                    "BP": r.get("BP", 0),
                })
    return reviews


# ── 主流程 ───────────────────────────────────────────────────────────────────

def run(paths, sample_size=None, batch_size=10, dry_run=False,
        output="absa_labeled.jsonl"):

    reviews = load_reviews(paths)
    print(f"载入 {len(reviews)} 则评论")

    # 依型号分组（同型号一起标，prompt 才有 model 上下文）
    by_model = defaultdict(list)
    for r in reviews:
        by_model[(r["category"], r["model"])].append(r)

    # 小样本模式：只取前 N 则（跨型号）
    if sample_size:
        # 从评论数最多的型号取样，确保有代表性
        sorted_groups = sorted(by_model.items(), key=lambda x: -len(x[1]))
        sampled = []
        for (cat, model), group in sorted_groups:
            for r in group:
                sampled.append(r)
                if len(sampled) >= sample_size:
                    break
            if len(sampled) >= sample_size:
                break
        # 重新分组
        by_model = defaultdict(list)
        for r in sampled:
            by_model[(r["category"], r["model"])].append(r)
        print(f"小样本模式：取 {len(sampled)} 则")

    labeled_data = []
    api_calls = 0

    for (category, model), group in by_model.items():
        contents = [r["content"] for r in group]
        print(f"\n▶ {model} ({category})  {len(contents)} 则")

        for i in range(0, len(contents), batch_size):
            batch = group[i:i+batch_size]
            batch_contents = [r["content"] for r in batch]
            prompt = build_labeling_prompt(category, model, batch_contents)

            if dry_run:
                print(f"  [dry-run] 批次 {i//batch_size+1}：{len(batch)} 则（不呼叫 API）")
                continue

            raw = call_gpt(prompt)
            api_calls += 1
            parsed = parse_response(raw)

            # 把 GPT 标注对回原始评论
            for item in parsed:
                idx = item.get("id", 0) - 1
                if 0 <= idx < len(batch):
                    orig = batch[idx]
                    labeled_data.append({
                        "category": category,
                        "model":    model,
                        "content":  orig["content"],
                        "orig_label": orig["orig_label"],   # 保留原标注供比对
                        "is_about_product": item.get("is_about_product", True),
                        "is_substantive":   item.get("is_substantive", True),
                        "aspects":      item.get("aspects", {}),
                        "extra_aspects": item.get("extra_aspects", {}),
                    })
            print(f"  批次 {i//batch_size+1}：标注 {len(parsed)} 则")

    if not dry_run:
        with open(output, "w", encoding="utf-8") as f:
            for d in labeled_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\n✓ 已输出 {len(labeled_data)} 则标注资料 → {output}")
        print(f"  API 呼叫次数：{api_calls}")
        # 粗估成本
        est_cost = api_calls * 0.003  # 每次约 0.003 美元（gpt-4o-mini）
        print(f"  估计成本：约 US${est_cost:.2f}")

    return labeled_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="小样本数量")
    ap.add_argument("--full",   action="store_true", help="标注全部")
    ap.add_argument("--dry",    action="store_true", help="不呼叫 API")
    ap.add_argument("--batch",  type=int, default=10)
    args = ap.parse_args()

    paths = ["matched_part1.jsonl", "matched_part2.jsonl", "matched_part3.jsonl"]

    run(
        paths,
        sample_size = None if args.full else (args.sample or 50),
        batch_size  = args.batch,
        dry_run     = args.dry,
    )
