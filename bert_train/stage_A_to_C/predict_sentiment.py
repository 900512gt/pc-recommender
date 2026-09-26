"""
predict_sentiment.py — 阶段C：用训练好的 BERT 跑全部评论，产出每个零件的面向分数

流程：
  1. 载入你 fine-tune 好的 BERT（bert_ckip_dims.pt）
  2. 跑全部 50,593 则评论 → 每则的五维度判断
  3. 依零件聚合 → 每个零件的面向分数（用「比例聚合」，有统计依据）
  4. 输出 part_sentiment.json 给 GA 用

用法：
  python predict_sentiment.py --model ckip
  python predict_sentiment.py --model ckip --input all_reviews.jsonl

输出：part_sentiment.json
  {
    "GPU|RTX5070": {
      "dimensions": {"效能": 0.78, "溫控": 0.32, ...},
      "counts": {"效能": {"正": 45, "负": 13}, ...},
      "review_count": 2942
    }
  }
"""

import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel

DIMENSIONS = ["效能", "溫控", "噪音", "保固", "CP值"]
ID2LABEL = {0: "未提及", 1: "正面", 2: "負面"}

MODEL_MAP = {
    "ckip":   "ckiplab/bert-base-chinese",
    "google": "bert-base-chinese",
    "hfl":    "hfl/chinese-roberta-wwm-ext",
}


# ── 模型结构（要跟训练时一致）──
class MultiDimBert(nn.Module):
    def __init__(self, model_name, n_dims=5, n_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleList([nn.Linear(hidden, n_classes) for _ in range(n_dims)])

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        return [head(pooled) for head in self.heads]


class InferDataset(Dataset):
    def __init__(self, contents, tokenizer, max_len=128):
        self.contents = contents
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.contents[idx], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def predict_all(model, contents, tokenizer, device, batch_size=64):
    """跑全部评论，回传每则的五维度预测。"""
    ds = InferDataset(contents, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size)

    all_preds = []   # 每则 → [dim0_label, dim1_label, ...]
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, mask)   # list of 5 x (batch, 3)

            # 转成 (batch, 5) 的预测
            batch_preds = torch.stack(
                [o.argmax(dim=1) for o in outputs], dim=1
            ).cpu().numpy()
            all_preds.extend(batch_preds)

            if (i + 1) % 50 == 0:
                done = min((i + 1) * batch_size, len(contents))
                print(f"  已处理 {done}/{len(contents)} 则")

    return all_preds


def aggregate_by_part(reviews, preds):
    """
    依零件聚合面向分数。
    分数 = 正面数 / (正面数 + 负面数)   ← 比例聚合，有统计依据
    没有任何正/负 → 分数 0.5（中性）
    """
    # part_counts[零件][维度] = {"正": n, "负": n}
    part_counts = defaultdict(lambda: {d: {"正": 0, "负": 0} for d in DIMENSIONS})
    part_reviews = defaultdict(int)

    for r, p in zip(reviews, preds):
        key = f"{r['category']}|{r['model']}"
        part_reviews[key] += 1
        for di, dim in enumerate(DIMENSIONS):
            label = ID2LABEL[int(p[di])]
            if label == "正面":
                part_counts[key][dim]["正"] += 1
            elif label == "負面":
                part_counts[key][dim]["负"] += 1

    # 算分数
    result = {}
    for key, dims in part_counts.items():
        scores = {}
        counts = {}
        for dim in DIMENSIONS:
            pos = dims[dim]["正"]
            neg = dims[dim]["负"]
            total = pos + neg
            # 比例聚合：正面占比
            scores[dim] = round(pos / total, 3) if total > 0 else 0.5
            counts[dim] = {"正": pos, "负": neg}

        result[key] = {
            "dimensions": scores,
            "counts": counts,               # 保留原始次数，可判断可信度
            "review_count": part_reviews[key],
        }

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ckip", "google", "hfl"], default="ckip")
    ap.add_argument("--input", default="all_reviews.jsonl")
    ap.add_argument("--output", default="part_sentiment.json")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"装置: {device}")

    model_name = MODEL_MAP[args.model]
    weights_path = f"bert_{args.model}_dims.pt"

    # 载入资料
    reviews = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    print(f"载入 {len(reviews)} 则评论")

    # 载入模型
    print(f"载入模型: {weights_path}")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = MultiDimBert(model_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("模型载入完成\n")

    # 预测
    print("开始预测...")
    contents = [r["content"] for r in reviews]
    preds = predict_all(model, contents, tokenizer, device, args.batch)
    print(f"预测完成\n")

    # 聚合
    print("聚合零件分数...")
    result = aggregate_by_part(reviews, preds)
    print(f"共 {len(result)} 个零件\n")

    # 输出
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✓ 已输出 {args.output}")

    # 展示几个范例
    print("\n=== 范例（评论数最多的 5 个零件）===")
    top = sorted(result.items(), key=lambda x: -x[1]["review_count"])[:5]
    for key, data in top:
        print(f"\n{key}  ({data['review_count']} 则评论)")
        for dim, score in data["dimensions"].items():
            c = data["counts"][dim]
            bar = "█" * int(score * 20)
            print(f"  {dim:5s} {score:.3f} (正{c['正']:4d}/负{c['负']:4d}) {bar}")


if __name__ == "__main__":
    main()
