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

  # 新資料（拿掉不相關評論＋總體面向），每個實驗存到不同資料夾，避免互相覆蓋
  python train_aspect.py --model ckip --epochs 4 --data-dir aspect_dataset_v2 --save-dir runs/v2_r3

  # 舊模型在新資料上重新評估（不訓練），和新模型用同一份 test 比較；結果存成 <模型資料夾>/eval_aspect_dataset_v2.json
  python train_aspect.py --eval-only --model-dir bert_ckip_aspect --data-dir aspect_dataset_v2

分項成績依來源、類別、「類別 面向」列出，各項都有正面／負面的 precision 與 recall。
資料含「總體」面向時，另外印一行「不含總體」的整體分數，和舊模型比較時看這一行。
資料夾裡有 test_no_opinion.jsonl、test_irrelevant.jsonl（答案全是未提及）時也會一起評估，改印誤判率。
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
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_recall_fscore_support

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
    m = {
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        # 只看有標註的樣本、只比正面/負面哪個分數高。
        # 對照基準：完全不讀文章、只看「型號 面向」去猜是 0.68，要明顯高過它才算有在讀內文。
        "polarity_acc": accuracy_score(labels[has], logits[has][:, 1:].argmax(axis=1) + 1) if has.any() else 0.0,
    }
    # 正面、負面各自的 precision / recall：目前主要問題是 precision 低（太常判成有評價），分項也要看得到。
    # 沒有這一類的答案時 recall 無從算起、沒有這一類的預測時 precision 無從算起，記成 None（印成「-」）
    p, r, _, _ = precision_recall_fscore_support(labels, preds, labels=[1, 2], zero_division=0)
    for k, c, pi, ri in (("pos", 1, p[0], r[0]), ("neg", 2, p[1], r[1])):
        m[f"{k}_p"] = float(pi) if (preds == c).any() else None
        m[f"{k}_r"] = float(ri) if (labels == c).any() else None
    return m


def distribution(ids):
    """各類別佔比，例如 {'未提及': 0.92, '正面': 0.05, '負面': 0.03}"""
    return {ID2LABEL[i]: float((ids == i).mean()) for i in range(3)}


def fmt_dist(d):
    return "、".join(f"{k} {v:.1%}" for k, v in d.items())


def fmt_sub(m):
    f = lambda v: "  - " if v is None else f"{v:.2f}"
    return (f"Macro-F1 {m['macro_f1']:.3f}  正 P/R {f(m['pos_p'])}/{f(m['pos_r'])}  "
            f"負 P/R {f(m['neg_p'])}/{f(m['neg_r'])}  正負向正確率 {m['polarity_acc']:.3f}")


def group_value(d, key):
    """分項的鍵；category_aspect 是「類別 面向」，同名面向（穩定、噪音…）在不同類別要分開看"""
    if key == "category_aspect":
        return f"{d.get('category')} {d.get('aspect')}"
    return d.get(key)


def report(name, data, logits, labels, by=("source", "category", "category_aspect")):
    """印出整體與分項分數，並回傳同一份內容（存進 result.json 用）"""
    m = metrics(logits, labels)
    print(f"\n=== {name} ===")
    print(f"  Acc {m['acc']:.3f}  Macro-F1 {m['macro_f1']:.3f}  正負向正確率 {m['polarity_acc']:.3f}")
    print(classification_report(
        labels, logits.argmax(axis=1), labels=[0, 1, 2],
        target_names=[ID2LABEL[i] for i in range(3)], digits=3, zero_division=0,
    ))
    # 「總體」面向是 2026-10 才加的，舊資料、舊模型都沒有。和舊模型比較要用不含總體的分數，
    # 總體本身的成績在下面「依 category_aspect」裡單獨看
    if any(d.get("aspect") == "總體" for d in data):
        idx = np.array([d.get("aspect") != "總體" for d in data])
        m["excluding_overall"] = dict(metrics(logits[idx], labels[idx]), n=int(idx.sum()))
        print(f"  不含總體 n={m['excluding_overall']['n']:>6}  {fmt_sub(m['excluding_overall'])}\n")
    for key in by:
        print(f"  依 {key}：")
        m[f"by_{key}"] = {}
        for val in sorted({group_value(d, key) for d in data}, key=str):
            idx = np.array([group_value(d, key) == val for d in data])
            sub = dict(metrics(logits[idx], labels[idx]), n=int(idx.sum()))
            m[f"by_{key}"][val] = sub
            print(f"    {str(val):<22} n={sub['n']:>6}  {fmt_sub(sub)}")
    return m


# 答案全是「未提及」的獨立測試集（build_aspect_dataset.py 另存的）：沒有正負評的答案，
# precision / recall 無從算起，改看誤判率 = 判成正面或負面的比例，越低越好
ALL_UNMENTIONED_SETS = ("test_no_opinion", "test_irrelevant")


def report_false_positive(name, data, logits, labels):
    preds = logits.argmax(axis=1)
    assert (labels == 0).all(), f"{name} 應該全部是未提及"

    def rate(idx):
        n = int(idx.sum())
        if n == 0:
            return {"n": 0}
        p = preds[idx]
        return {"n": n, "fp_rate": float((p != 0).mean()), "as_pos": float((p == 1).mean()), "as_neg": float((p == 2).mean())}

    def line(label, r):
        return (f"    {label:<22} n={r['n']:>6}  誤判率 {r['fp_rate']:.1%}（判成正面 {r['as_pos']:.1%}、負面 {r['as_neg']:.1%}）"
                if r["n"] else f"    {label:<22} n=     0")

    everything = np.ones(len(data), dtype=bool)
    m = rate(everything)
    print(f"\n=== {name}（答案全是未提及，看誤判率）===")
    print(line("全部", m))
    if any(d.get("aspect") == "總體" for d in data):
        m["excluding_overall"] = rate(np.array([d.get("aspect") != "總體" for d in data]))
        print(line("不含總體", m["excluding_overall"]))
    for key in ("source", "p3_had_opinion", "category"):
        if not any(key in d for d in data):
            continue
        print(f"  依 {key}：")
        m[f"by_{key}"] = {}
        for val in sorted({d.get(key) for d in data}, key=str):
            r = rate(np.array([d.get(key) == val for d in data]))
            m[f"by_{key}"][str(val)] = r
            print(line(str(val), r))
    # 誤判最多的面向（只列樣本數夠的）
    per = {}
    for val in {group_value(d, "category_aspect") for d in data}:
        r = rate(np.array([group_value(d, "category_aspect") == val for d in data]))
        if r["n"] >= 50:
            per[val] = r
    m["by_category_aspect"] = per
    print("  誤判率最高的面向（樣本數 ≥ 50）：" + ("" if per else "樣本數不足，不列"))
    for val, r in sorted(per.items(), key=lambda kv: -kv[1]["fp_rate"])[:8]:
        print(line(val, r))
    return m


def evaluate_set(name, data, model, loader, device):
    """依測試集種類選評分方式"""
    logits, labels = predict(model, loader(data), device)
    if name in ALL_UNMENTIONED_SETS:
        return report_false_positive(name, data, logits, labels)
    by = ("held_reason", "source", "category") if name == "test_hard" else ("source", "category", "category_aspect")
    return report(name, data, logits, labels, by=by)


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

    save_dir = Path(args.save_dir) if args.save_dir else HERE / f"bert_{args.model}_aspect"
    if (save_dir / "result.json").exists() and not args.overwrite:
        raise SystemExit(f"✗ {save_dir} 已經有訓練好的模型，繼續會覆蓋掉。\n"
                         f"  請用 --save-dir 指定新的資料夾；確定要覆蓋就加 --overwrite。")
    best_f1 = -1.0
    history = []  # 每個 epoch 的 val 分數
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(**batch).logits.float(), labels)
            if not torch.isfinite(loss):
                raise SystemExit(
                    f"\n✗ Epoch {epoch+1} Step {i+1} 的 loss 變成 {loss.item()}，訓練已停止。\n"
                    f"  常見原因是學習率太高，可以試 --lr 1e-5；"
                    f"目前為止最好的模型{'在 ' + str(save_dir) if best_f1 >= 0 else '尚未產生'}。"
                )
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
        val_logits, val_labels = predict(model, val_dl, device)
        m = metrics(val_logits, val_labels)
        pred_dist = distribution(val_logits.argmax(axis=1))
        print(f"  val: Acc {m['acc']:.3f}  Macro-F1 {m['macro_f1']:.3f}  正負向正確率 {m['polarity_acc']:.3f}")
        # 正面、負面的預測比例如果接近 0，就是全部猜未提及在偷懶
        print(f"    預測分布：{fmt_dist(pred_dist)}")
        print(f"    實際分布：{fmt_dist(distribution(val_labels))}")
        history.append(dict(m, epoch=epoch + 1, train_loss=total_loss / len(train_dl), pred_dist=pred_dist))
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  ✓ 目前最佳，已存: {save_dir}")

    # 載回最好的那一輪，test 只在這裡跑一次
    model = BertForSequenceClassification.from_pretrained(save_dir).to(device)
    val_m = report("val（最佳 epoch）", val_data, *predict(model, val_dl, device))
    test_m = report("test", test_data, *predict(model, test_dl, device))

    # test_hard：整理資料時拿掉的難題（高風險、巴哈不相關但有面向），答案沿用 p3 標註。
    # 這些標註本身就可疑，分數只用來看「test 因為拿掉難題而高估了多少」，不是準確率。
    hard_m = None
    if (data_dir / "test_hard.jsonl").exists():
        hard_data = load_jsonl(data_dir / "test_hard.jsonl", args.sample)
        hard_m = report("test_hard（參考用）", hard_data, *predict(model, loader(hard_data), device),
                        by=("held_reason", "source", "category"))

    extra = {}
    for name in ALL_UNMENTIONED_SETS:
        if (data_dir / f"{name}.jsonl").exists():
            extra[name] = evaluate_set(name, load_jsonl(data_dir / f"{name}.jsonl", args.sample), model, loader, device)

    with open(save_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, "args": vars(args), "val_history": history,
                   "val": val_m, "test": test_m, "test_hard": hard_m, **extra}, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 模型與結果已存: {save_dir}")


# ── 只評估 ──
def evaluate(args):
    """
    載入已訓練好的模型，在指定資料上評估，不訓練。
    用途：舊模型在新資料的 test 上重新評估，和新模型用同一份 test 比較。
    結果另存 eval_<資料夾名>.json，不會動到模型資料夾裡原本的 result.json。
    """
    if not args.model_dir:
        raise SystemExit("✗ --eval-only 需要用 --model-dir 指定模型資料夾")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")
    model_dir, data_dir = Path(args.model_dir), Path(args.data_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    collate = DataCollatorWithPadding(tokenizer)

    def loader(data):
        return DataLoader(AspectDataset(data, tokenizer, args.max_len), batch_size=args.batch, collate_fn=collate)

    out = {"model_dir": str(model_dir), "data_dir": str(data_dir), "args": vars(args)}
    for name in ("val", "test", "test_hard") + ALL_UNMENTIONED_SETS:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        print(f"\n##### {name}（{model_dir.name} × {data_dir.name}）")
        out[name] = evaluate_set(name, load_jsonl(path, args.sample), model, loader, device)
    dest = model_dir / f"eval_{data_dir.name}.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 評估結果已存: {dest}")


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
    ap.add_argument("--save-dir", default=None, help="模型與 result.json 存放處（預設 bert_<model>_aspect/）")
    ap.add_argument("--overwrite", action="store_true", help="允許覆蓋 --save-dir 裡已有的模型")
    ap.add_argument("--eval-only", action="store_true", help="不訓練，只用 --model-dir 的模型評估 --data-dir")
    ap.add_argument("--model-dir", default=None, help="--eval-only 時要評估的模型資料夾")
    args = ap.parse_args()
    evaluate(args) if args.eval_only else train(args)
