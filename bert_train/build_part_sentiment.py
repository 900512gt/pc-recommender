# build_part_sentiment.py — 從 p3 面向標註直接算出各型號的面向口碑分數
#
# 口碑分數直接用標註結果加總，不經過 BERT 預測：
#   這一輪所有相關評論都標過面向了，標註比模型預測準；BERT 留給之後新進的評論用。
#
# 處理步驟：
#   1. 只留「相關」的評論（orig_relevant）
#   2. 拿掉高風險筆數（疑似把別的型號的評價記到自己頭上），規則沿用 build_aspect_dataset.py
#   3. 拿掉早於發售日的評論，發售日用 exclude_prerelease.py 內建的完整表
#      （之前只照組員清單的 21 個型號排除，其他型號沒處理到）
#   4. 依（類別, 型號, 面向）加總正面、負面則數
#        面向分數 = (正面 + K × 類別平均) / (正面 + 負面 + K)
#          類別平均 = 同類別所有型號在這個面向的正面比例；K = PRIOR_STRENGTH
#          則數少的往類別平均靠，則數多的幾乎就是自己的正面比例；
#          完全沒人提的面向就是類別平均，所以不需要「滿幾則才給分」的門檻
#        型號分數 = 各面向分數取平均（各面向等權）
#      同一顆晶片的不同版本（SAME_CHIP）評價合在一起算，共用同一組分數
#
# 面向用各類別自己的清單（CPU 6 個、風冷 4 個…），不合併成五大面向。
#
# 用法（專案根目錄）：
#   python bert_train/build_part_sentiment.py
#   python bert_train/build_part_sentiment.py --keep-high-risk     # 高風險筆數也算進去
# 輸出：
#   data/aspect_reviews_final.jsonl     ← 實際拿來算分數的評論
#   data/aspect_reviews_dropped.jsonl   ← 被拿掉的相關評論與原因，供複查
#   code/ga_pc_builder/data/part_aspect_sentiment.json  ← 各型號的面向分數（GA 讀這份）
#       "_prior" 是各類別的平均，資料庫裡沒有任何評論的型號用它

import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "filter"))

from build_aspect_dataset import load_jsonl, write_jsonl, find_high_risk, UNMENTIONED  # noqa: E402
from exclude_prerelease import days_before_release  # noqa: E402

# 往類別平均靠的力道：相當於每個面向先墊 10 則「類別平均」的評價。
# 試過 0 / 5 / 10 / 20：同晶片的 K、KF 版（265K 與 265KF）效能分數差距是 0.25 / 0.20 / 0.17 / 0.13，
# 顯卡型號分數的標準差是 0.086 / 0.059 / 0.049 / 0.039。太小壓不住雜訊，太大型號之間分不出高下。
PRIOR_STRENGTH = 10
SCORE_FILE = ROOT / "code" / "ga_pc_builder" / "data" / "part_aspect_sentiment.json"

# 同一顆晶片的不同版本：Intel 型號結尾的 F 只代表沒有內建顯示，效能、溫度、功耗都相同。
# 分開算的話樣本少的那一版分數會亂跳（265K 與 265KF 的效能分數曾差到 0.25），
# GA 會為了這個不存在的口碑差距改選比較貴的版本。
# 認定標準：型號只差一個 F，且 PassMark 跑分相差 3% 以內。用明確的對照表而不是
# 「去掉 F」的規則——AMD 的 7500F、8400F 是獨立的產品，沒有對應的無 F 版。
SAME_CHIP = {
    "Intel Core Ultra 7 265KF": "Intel Core Ultra 7 265K",
    "Intel Core Ultra 5 245KF": "Intel Core Ultra 5 245K",
    "Intel Core Ultra 5 225F":  "Intel Core Ultra 5 225",
    "Intel i5-14400F":          "Intel i5-14400",
}


def flag_high_risk(rows):
    """
    回傳 {id(row): 原因}。find_high_risk 的結果跟傳進去的資料範圍有關，
    所以照 build_aspect_dataset.py 的步驟 1 準備同一批資料再判，兩邊拿掉的才會是同一批。
    """
    kept = []
    for r in rows:
        if not r["orig_relevant"] and r["evidence"]:
            if r["source"] == "baha":
                continue
            r = dict(r, evidence={}, _orig=r)
        kept.append(r)
    return {id(kept[i].get("_orig", kept[i])): why for i, why in find_high_risk(kept).items()}


def aggregate(rows, k):
    """回傳 {'類別|型號': {...}, '_prior': {類別: {...}}}，型號鍵的格式與 part_sentiment.json 相同"""
    counts = defaultdict(lambda: defaultdict(lambda: {"正面": 0, "負面": 0}))
    reviews = defaultdict(Counter)
    members = defaultdict(set)   # 晶片 → 實際出現過的型號；同晶片的型號共用同一份統計
    for r in rows:
        key = f"{r['category']}|{SAME_CHIP.get(r['model'], r['model'])}"
        members[key].add(r["model"])
        reviews[key][r["source"]] += 1
        for aspect, label in r["aspects"].items():
            cell = counts[key][aspect]   # 沒被提到的面向也要出現在輸出裡
            if label != UNMENTIONED:
                cell[label] += 1

    # 類別平均：同類別所有型號合起來的正面比例；整個類別都沒人提的面向給 0.5
    pooled = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for key, aspects in counts.items():
        for aspect, c in aspects.items():
            pooled[key.split("|")[0]][aspect][0] += c["正面"]
            pooled[key.split("|")[0]][aspect][1] += c["負面"]
    prior = {cat: {a: (pos / (pos + neg) if pos + neg else 0.5) for a, (pos, neg) in aspects.items()}
             for cat, aspects in pooled.items()}

    out = {"_prior": {
        cat: {"score": round(sum(p.values()) / len(p), 4),
              "aspects": {a: round(v, 4) for a, v in p.items()}}
        for cat, p in sorted(prior.items())
    }}
    for key in sorted(counts):
        cat = key.split("|")[0]
        aspects = {}
        for aspect, c in counts[key].items():
            n = c["正面"] + c["負面"]
            aspects[aspect] = {
                "正面": c["正面"], "負面": c["負面"],
                "raw": round(c["正面"] / n, 4) if n else None,
                "score": round((c["正面"] + k * prior[cat][aspect]) / (n + k) if n + k
                               else prior[cat][aspect], 4),
            }
        entry = {
            "score": round(sum(a["score"] for a in aspects.values()) / len(aspects), 4),
            "review_count": sum(reviews[key].values()),
            "ptt": reviews[key]["ptt"], "baha": reviews[key]["baha"],
            "aspects": aspects,
        }
        for model in sorted(members[key]):
            others = sorted(members[key] - {model})
            out[f"{cat}|{model}"] = dict(entry, shared_with=others) if others else entry
    return out


def main(args):
    rows = load_jsonl(args.input)
    print(f"讀入 {len(rows)} 筆：{args.input}")

    risk = flag_high_risk(rows)
    final, dropped = [], []
    for r in rows:
        if not r["orig_relevant"]:
            continue
        early = days_before_release(r)
        if id(r) in risk and not args.keep_high_risk:
            dropped.append(dict(r, drop_reason=risk[id(r)]))
        elif early is not None:
            dropped.append(dict(r, drop_reason="prerelease", days_before_release=early))
        else:
            final.append(r)

    scores = aggregate(final, args.prior_strength)

    out = Path(args.output_dir)
    write_jsonl(out / "aspect_reviews_final.jsonl", final)
    write_jsonl(out / "aspect_reviews_dropped.jsonl", dropped)
    with open(args.score_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=1)

    # ── 報告 ──
    n_rel = len(final) + len(dropped)
    print("\n=== 評論（一筆 = 一則評論 × 一個型號）===")
    print(f"  相關                {n_rel:>7}")
    for why, n in Counter(r["drop_reason"] for r in dropped).most_common():
        print(f"  拿掉 {why:<22}{n:>5}")
    print(f"  拿來算分數          {len(final):>7}  {dict(Counter(r['source'] for r in final))}")

    models = {key: s for key, s in scores.items() if "|" in key}
    cells = [a for s in models.values() for a in s["aspects"].values()]
    n10 = sum(a["正面"] + a["負面"] >= 10 for a in cells)
    print(f"\n=== 分數（往類別平均靠的力道 K = {args.prior_strength}）===")
    print(f"  型號                {len(models):>7}")
    print(f"  型號 × 面向         {len(cells):>7}")
    print(f"  其中正負評價達 10 則 {n10:>6}（{n10 / len(cells):.1%}），其餘主要由類別平均決定")
    shared = sorted({" + ".join(sorted([k.split("|")[1]] + v["shared_with"]))
                     for k, v in models.items() if v.get("shared_with")})
    print(f"  同晶片合併計算       {len(shared):>6} 組：{'；'.join(shared)}")
    print("\n  各類別（型號數 / 類別平均 / 型號分數最低~最高）")
    by_cat = defaultdict(list)
    for key, s in models.items():
        by_cat[key.split("|")[0]].append(s["score"])
    for cat, vals in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"    {cat:<6}{len(vals):>4}   {scores['_prior'][cat]['score']:.2f}   {min(vals):.2f} ~ {max(vals):.2f}")
    print(f"\n評論輸出目錄：{out}")
    print(f"分數檔：{args.score_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "aspects_標註_p3.jsonl"))
    ap.add_argument("--output-dir", default=str(ROOT / "data"))
    ap.add_argument("--score-file", default=str(SCORE_FILE))
    ap.add_argument("--prior-strength", type=float, default=PRIOR_STRENGTH,
                    help="往類別平均靠的力道，0 表示不修正（沒人提的面向仍給類別平均）")
    ap.add_argument("--keep-high-risk", action="store_true", help="高風險筆數也算進分數")
    main(ap.parse_args())
