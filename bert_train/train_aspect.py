"""
train_aspect.py — fine-tune BERT 做「型號 × 面向」情感分類

輸入是 build_aspect_dataset.py 產出的 aspect_dataset/{train,val,test}.jsonl，
每筆是 [CLS] 型號 面向 [SEP] 標題 內文 [SEP] → 未提及 / 正面 / 負面。

跟 train_bert.py 的差別：
  - 型號和面向寫在輸入裡，所以只有一個三分類的輸出（不是五個維度各一個頭）
  - 每個 epoch 用 val 評估並留下最好的那一輪，test 只在最後跑一次

用法：
  # 第一次要先產生訓練資料（aspect_dataset/ 不進版控，約 10 秒）
  python build_aspect_dataset.py

  # 先小資料試跑（確認腳本沒錯，幾分鐘）
  python train_aspect.py --model ckip --sample 2000 --epochs 1

  # 正式訓練
  python train_aspect.py --model ckip --epochs 4
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    DataCollatorWithPadding, get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ── 設定 ──
LABEL2ID = {"未提及": 0, "正面": 1, "負面": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MODEL_MAP = {
    "ckip":   "ckiplab/bert-base-chinese",     # 中研院
    "google": "bert-base-chinese",              # Google 原版
    "hfl":    "hfl/chinese-roberta-wwm-ext",    # 哈工大(可選)
}

HERE = Path(__file__).resolve().parent


def load_jsonl(path, sample=None):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    # val / test 檔案裡是依來源排的，試跑要隨機抽才看得到 PTT 和巴哈兩種
    return random.Random(42).sample(data, min(sample, len(data))) if sample else data


# ── Dataset ──
class AspectDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 太長只截後半（標題+內文），前半的「型號 面向」一定要完整留著
        enc = self.tokenizer(
            item["query"], item["text"],
            truncation="only_second", max_length=self.max_len,
        )
        enc["labels"] = LABEL2ID[item["label"]]
        return enc


# ── 評估 ──
def predict(model, dl, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in dl:
            labels.append(batch.pop("labels").numpy())
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(**batch).logits
            logits.append(out.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels)


def metrics(logits, labels):
    preds = logits.argmax(axis=1)
    has = labels != 0
    return {
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        # 只看有標註的樣本、只比正面/負面哪個分數高。
        # 對照基準：完全不讀文章、只看「型號 面向」去猜是 0.68，要明顯高過它才算有在讀內文。
        "polarity_acc": accuracy_score(labels[has], logits[has][:, 1:].argmax(axis=1) + 1) if has.any() else 0.0,
    }


def report(name, data, logits, labels):
    m = metrics(logits, labels)
    print(f"\n=== {name} ===")
    print(f"  Acc {m['acc']:.3f}  Macro-F1 {m['macro_f1']:.3f}  正負向正確率 {m['polarity_acc']:.3f}")
    print(classification_report(
        labels, logits.argmax(axis=1), labels=[0, 1, 2],
        target_names=[ID2LABEL[i] for i in range(3)], digits=3, zero_division=0,
    ))
    for key in ("source", "category"):
        print(f"  依 {key}：")
        for val in sorted({d[key] for d in data}):
            idx = np.array([d[key] == val for d in data])
            sub = metrics(logits[idx], labels[idx])
            print(f"    {val:<6} n={idx.sum():>6}  Macro-F1 {sub['macro_f1']:.3f}  正負向正確率 {sub['polarity_acc']:.3f}")
    return m


# ── 訓練 ──
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = MODEL_MAP[args.model]
    print(f"模型: {model_name}")

    data_dir = Path(args.data_dir)
    train_data = load_jsonl(data_dir / "train.jsonl", args.sample)
    val_data = load_jsonl(data_dir / "val.jsonl", args.sample)
    test_data = load_jsonl(data_dir / "test.jsonl", args.sample)
    print(f"訓練: {len(train_data)}  驗證: {len(val_data)}  測試: {len(test_data)}")
    print(f"訓練集標籤: {dict(Counter(d['label'] for d in train_data))}")

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    collate = DataCollatorWithPadding(tokenizer)  # 每個 batch 只補到該 batch 最長的那筆
    def loader(data, shuffle=False):
        return DataLoader(AspectDataset(data, tokenizer, args.max_len),
                          batch_size=args.batch, shuffle=shuffle, collate_fn=collate)
    train_dl, val_dl, test_dl = loader(train_data, True), loader(val_data), loader(test_data)

    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    # 訓練資料已經把「未提及」減量到正負面的 3 倍，預設不再加 class weight，
    # 兩個一起用等於補償兩次，模型會太容易判成正面/負面。要比較可以加 --class-weight。
    weight = None
    if args.class_weight:
        counts = [1, 1, 1]  # 平滑
        for d in train_data:
            counts[LABEL2ID[d["label"]]] += 1
        w = torch.tensor([sum(counts) / c for c in counts], dtype=torch.float)
        weight = (w / w.sum() * 3).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    save_dir = HERE / f"bert_{args.model}_aspect"
    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(**batch).logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Step {i+1}/{len(train_dl)} | Loss {loss.item():.4f}")

        print(f"Epoch {epoch+1} 平均 Loss: {total_loss / len(train_dl):.4f}")

        # 每個 epoch 用 val 評估，只留 val Macro-F1 最好的那一輪
        m = metrics(*predict(model, val_dl, device))
        print(f"  val: Acc {m['acc']:.3f}  Macro-F1 {m['macro_f1']:.3f}  正負向正確率 {m['polarity_acc']:.3f}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  ✓ 目前最佳，已存: {save_dir}")

    # 載回最好的那一輪，test 只在這裡跑一次
    model = BertForSequenceClassification.from_pretrained(save_dir).to(device)
    report("val（最佳 epoch）", val_data, *predict(model, val_dl, device))
    test_m = report("test", test_data, *predict(model, test_dl, device))

    # test_hard：整理資料時拿掉的難題（高風險、巴哈不相關但有面向），答案沿用 p3 標註。
    # 這些標註本身就可疑，分數只用來看「test 因為拿掉難題而高估了多少」，不是準確率。
    hard_m = None
    if (data_dir / "test_hard.jsonl").exists():
        hard_data = load_jsonl(data_dir / "test_hard.jsonl", args.sample)
        hard_logits, hard_labels = predict(model, loader(hard_data), device)
        hard_m = report("test_hard（參考用）", hard_data, hard_logits, hard_labels)
        print("  依拿掉的原因：")
        for why in sorted({d["held_reason"] for d in hard_data}):
            idx = np.array([d["held_reason"] == why for d in hard_data])
            sub = metrics(hard_logits[idx], hard_labels[idx])
            print(f"    {why:<22} n={idx.sum():>5}  Macro-F1 {sub['macro_f1']:.3f}  正負向正確率 {sub['polarity_acc']:.3f}")

    with open(save_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, "args": vars(args), "best_val_macro_f1": best_f1,
                   "test": test_m, "test_hard": hard_m}, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 模型與結果已存: {save_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ckip", "google", "hfl"], default="ckip")
    ap.add_argument("--data-dir", default=str(HERE / "aspect_dataset"))
    ap.add_argument("--sample", type=int, default=None, help="小資料試跑")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=416, help="資料最長約 412 個 token")
    ap.add_argument("--class-weight", action="store_true", help="loss 加上類別權重（預設不加）")
    args = ap.parse_args()
    train(args)
