# dump_confusion.py — 用訓練好的模型跑測試集，存混淆矩陣（給報告畫圖用）
#
# train_aspect.py 評估時只存指標、不存混淆矩陣，這支補上。不訓練、不改任何資料。
# 「總體」面向一律排除：舊模型沒學過總體，和第 18 頁的比較方式一致。
#
# 用法（bert_train/ 底下）：
#   python dump_confusion.py --model-dir runs/v3b --data-dir aspect_dataset_v3b --out cm_v3b
# 輸出：
#   <out>/confusion.json   各測試集的 3×3 混淆矩陣（列 = 答案，欄 = 預測，順序：未提及、正面、負面）
#   <out>/<測試集>_logits.npy、_labels.npy   原始輸出，之後要看機率分布可以直接用
#
# 主要測試集裡若有「整則評論所有面向都未提及」的樣本（v3b 才有），另外存一份
# test_no_opinion_slice，方便看這類評論的誤判率。

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import confusion_matrix

from train_aspect import load_jsonl, AspectDataset, predict

SETS = ("test", "test_no_opinion", "test_irrelevant")


def no_opinion_rids(labels_path, overall_path):
    """相關、但所有面向（含總體）都未提及的評論的 rid，定義同 build_aspect_dataset.py --drop-no-opinion。
    要看原始標註而不是展開後的樣本：長評論的有評價面向若找不到證據所在的視窗會被捨棄，
    剩下的樣本可能全是未提及，從樣本判斷會把這種評論誤算成無評價。
    rid = 標註檔的行號；總體標註以 _index 對應同一個行號。"""
    overall = {x["_index"]: x["總體"] for x in load_jsonl(overall_path)}
    rids = set()
    for rid, r in enumerate(load_jsonl(labels_path)):
        if not r["orig_relevant"]:
            continue
        values = list(r["aspects"].values()) + [overall.get(rid, "未提及")]
        if all(v == "未提及" for v in values):
            rids.add(rid)
    return rids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=416)
    ap.add_argument("--labels", default="aspects_標註_p3.jsonl", help="建資料時用的面向標註（判斷無評價用）")
    ap.add_argument("--overall", default="aspects_總體_p4.jsonl", help="建資料時用的總體標註（判斷無評價用）")
    args = ap.parse_args()
    no_op_rids = no_opinion_rids(args.labels, args.overall)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"裝置: {device}", flush=True)

    tok = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()
    collate = DataCollatorWithPadding(tok)

    result = {"model_dir": args.model_dir, "data_dir": args.data_dir,
              "labels": ["未提及", "正面", "負面"], "excluding_overall": True}
    for name in SETS:
        path = Path(args.data_dir) / f"{name}.jsonl"
        if not path.exists():
            continue
        data = [d for d in load_jsonl(path) if d.get("aspect") != "總體"]
        no_op = np.array([d["rid"] in no_op_rids for d in data])
        dl = DataLoader(AspectDataset(data, tok, args.max_len), batch_size=args.batch, collate_fn=collate)
        logits, labels = predict(model, dl, device)
        np.save(out / f"{name}_logits.npy", logits)
        np.save(out / f"{name}_labels.npy", labels)
        preds = logits.argmax(1)
        cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
        result[name] = {"n": len(data), "confusion": cm.tolist()}
        print(f"{name}  n={len(data)}  {cm.tolist()}", flush=True)

        if name == "test":
            mask = no_op
            if mask.any():
                cm_slice = confusion_matrix(labels[mask], preds[mask], labels=[0, 1, 2])
                result["test_no_opinion_slice"] = {"n": int(mask.sum()), "confusion": cm_slice.tolist()}
                print(f"  其中無評價評論  n={int(mask.sum())}  {cm_slice.tolist()}", flush=True)

    (out / "confusion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已存：{out / 'confusion.json'}", flush=True)


if __name__ == "__main__":
    main()
