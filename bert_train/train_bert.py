"""
train_bert.py — fine-tune BERT 做五大维度情感分类

一个模型同时输出 5 个维度（效能/温控/噪音/保固/CP值），
每个维度 3 分类（未提及=0 / 正面=1 / 负面=2）。

支援模型比较：改 --model 参数就能换 中研院 / Google BERT。

用法：
  # 先小资料试跑（确认脚本没错，约 5 分钟）
  python train_bert.py --model ckip --sample 1000 --epochs 1

  # 正式训练中研院 BERT
  python train_bert.py --model ckip --epochs 3

  # 正式训练 Google BERT（比较用）
  python train_bert.py --model google --epochs 3
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# ── 设定 ──
DIMENSIONS = ["效能", "溫控", "噪音", "保固", "CP值"]
LABEL2ID = {"未提及": 0, "正面": 1, "負面": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MODEL_MAP = {
    "ckip":   "ckiplab/bert-base-chinese",     # 中研院
    "google": "bert-base-chinese",              # Google 原版
    "hfl":    "hfl/chinese-roberta-wwm-ext",    # 哈工大(可选)
}


# ── Dataset ──
class ReviewDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["content"],
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        # 5 个维度的 label
        labels = torch.tensor(
            [LABEL2ID[item["dimensions"][d]] for d in DIMENSIONS],
            dtype=torch.long,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ── 模型：BERT + 5 个分类头 ──
class MultiDimBert(nn.Module):
    def __init__(self, model_name, n_dims=5, n_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        # 每个维度一个分类头
        self.heads = nn.ModuleList([
            nn.Linear(hidden, n_classes) for _ in range(n_dims)
        ])

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        # 每个维度各自输出
        return [head(pooled) for head in self.heads]


# ── 训练 ──
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"装置: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = MODEL_MAP[args.model]
    print(f"模型: {model_name}")

    # 载入资料
    data = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if args.sample:
        data = data[:args.sample]
    print(f"资料: {len(data)} 则")

    # 切 train/test
    train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
    print(f"训练: {len(train_data)}  测试: {len(test_data)}")

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_ds = ReviewDataset(train_data, tokenizer)
    test_ds = ReviewDataset(test_data, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch)

    model = MultiDimBert(model_name).to(device)

    # class weight：处理「未提及」过多的不平衡
    # 统计各维度各类别数量
    weights = []
    for di, dim in enumerate(DIMENSIONS):
        counts = [1, 1, 1]  # 平滑
        for d in train_data:
            counts[LABEL2ID[d["dimensions"][dim]]] += 1
        total = sum(counts)
        w = torch.tensor([total / c for c in counts], dtype=torch.float).to(device)
        w = w / w.sum() * 3  # 正规化
        weights.append(w)

    criterions = [nn.CrossEntropyLoss(weight=w) for w in weights]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # (batch, 5)

            optimizer.zero_grad()
            outputs = model(input_ids, mask)  # list of 5 x (batch, 3)

            loss = 0
            for di in range(len(DIMENSIONS)):
                loss += criterions[di](outputs[di], labels[:, di])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Step {i+1}/{len(train_dl)} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")

        # 每个 epoch 评估
        evaluate(model, test_dl, device)

    # 存模型
    save_path = f"bert_{args.model}_dims.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ 模型已存: {save_path}")
    # 存 tokenizer 名称供推论用
    with open(f"bert_{args.model}_config.json", "w") as f:
        json.dump({"model_name": model_name, "dimensions": DIMENSIONS}, f)


def evaluate(model, dl, device):
    model.eval()
    all_preds = [[] for _ in DIMENSIONS]
    all_labels = [[] for _ in DIMENSIONS]
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(input_ids, mask)
            for di in range(len(DIMENSIONS)):
                preds = outputs[di].argmax(dim=1).cpu().numpy()
                all_preds[di].extend(preds)
                all_labels[di].extend(labels[:, di].numpy())

    print("  === 各维度评估 ===")
    macro_f1s = []
    for di, dim in enumerate(DIMENSIONS):
        acc = accuracy_score(all_labels[di], all_preds[di])
        f1 = f1_score(all_labels[di], all_preds[di], average="macro", zero_division=0)
        macro_f1s.append(f1)
        print(f"    {dim}: Acc {acc:.3f}  Macro-F1 {f1:.3f}")
    print(f"  平均 Macro-F1: {np.mean(macro_f1s):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ckip", "google", "hfl"], default="ckip")
    ap.add_argument("--input", default="train_dimensions.jsonl")
    ap.add_argument("--sample", type=int, default=None, help="小资料试跑")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    train(args)
