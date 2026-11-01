"""
PTT 相關性標註的盲測核對：相關、不相關各抽一半混在一起，清單上不顯示 GPT 的判斷，
人工標完後再對答案，算出整體正確率與兩側各自的準確度（可與巴哈的數字並列）。

母體就是要進 BERT 訓練的資料：
  相關   ← src/database/input/ptt_comment/relevant_fixed.jsonl
  不相關 ← src/filter/output/irrelevant_sampled.jsonl（sample_irrelevant.py 的輸出，
           已出現在 irrelevant_sample_review.csv 的不重複抽）
兩側都依類別分層：每個類別至少 2 則，其餘按比例。排除 HDD。

用法（專案根目錄）：
    python src/filter/relevance_blind_check.py                 # 產生清單與答案
    python src/filter/relevance_blind_check.py --score         # 標完後對答案
    python src/filter/relevance_blind_check.py --score 路徑    # 清單另存到別處（csv 或 xlsx）時
輸出：
    src/filter/output/ptt_relevance_blind_review.csv  ← 人工標這份
    src/filter/output/ptt_relevance_blind_key.csv     ← GPT 的判斷，標完前不要看
「人工判斷」欄填：相關 / 不相關 / 邊界（邊界案例會算出「算錯」與「算對」兩個數字）
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
RELEVANT     = ROOT / "src/database/input/ptt_comment/relevant_fixed.jsonl"
IRRELEVANT   = ROOT / "src/filter/output/irrelevant_sampled.jsonl"
PREV_REVIEW  = ROOT / "src/filter/output/irrelevant_sample_review.csv"
REVIEW_CSV   = ROOT / "src/filter/output/ptt_relevance_blind_review.csv"
KEY_CSV      = ROOT / "src/filter/output/ptt_relevance_blind_key.csv"

EXCLUDE_CATS = {"HDD"}
MIN_PER_CAT  = 2
HUMAN_COL    = "人工判斷(相關 / 不相關 / 邊界)"


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def stratified(rows, target):
    """依類別分層：每個類別至少 MIN_PER_CAT 則，其餘按比例"""
    by_cat = defaultdict(list)
    for x in rows:
        by_cat[x["category"]].append(x)
    remaining = max(0, target - MIN_PER_CAT * len(by_cat))
    out = []
    for cat in sorted(by_cat):
        n = MIN_PER_CAT + round(remaining * len(by_cat[cat]) / len(rows))
        out.extend(random.sample(by_cat[cat], min(n, len(by_cat[cat]))))
    return out


def generate(per_side, seed):
    random.seed(seed)
    relevant = [x for x in load(RELEVANT) if x["category"] not in EXCLUDE_CATS]
    irrelevant = load(IRRELEVANT)
    if PREV_REVIEW.exists():
        with open(PREV_REVIEW, encoding="utf-8-sig", newline="") as f:
            seen = {int(r["_index"]) for r in csv.DictReader(f)}
        irrelevant = [x for x in irrelevant if x["_index"] not in seen]

    picked = stratified(relevant, per_side) + stratified(irrelevant, per_side)
    random.shuffle(picked)

    with open(REVIEW_CSV, "w", encoding="utf-8-sig", newline="") as f, \
         open(KEY_CSV, "w", encoding="utf-8-sig", newline="") as k:
        w, kw = csv.writer(f), csv.writer(k)
        w.writerow(["編號", "類別", "型號", "文章標題", "內容", "url", HUMAN_COL, "備註"])
        kw.writerow(["編號", "_index", "類別", "GPT判斷", "label"])
        for i, x in enumerate(picked, 1):
            w.writerow([i, x["category"], x["model"], x["title"], x["content"], x["url"], "", ""])
            kw.writerow([i, x["_index"], x["category"],
                         "相關" if x["relevant"] else "不相關", x["label"] or ""])

    cnt = Counter((x["category"], x["relevant"]) for x in picked)
    print(f"{'類別':<14}{'相關':>6}{'不相關':>6}")
    for cat in sorted({c for c, _ in cnt}):
        print(f"{cat:<14}{cnt[(cat, True)]:>8}{cnt[(cat, False)]:>8}")
    print(f"共 {len(picked)} 則（相關 {sum(x['relevant'] for x in picked)}、"
          f"不相關 {sum(not x['relevant'] for x in picked)}）")
    print(f"人工標這份：{REVIEW_CSV.relative_to(ROOT)}")
    print(f"答案（先別看）：{KEY_CSV.relative_to(ROOT)}")


def read_review(path):
    """讀人工標完的清單（csv 或 xlsx）→ {編號: 人工判斷}"""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        rows = [[("" if v is None else str(v)) for v in r]
                for r in load_workbook(path, read_only=True).active.iter_rows(values_only=True)]
    else:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
    header = rows[0]
    id_col = header.index("編號")
    human_col = next(i for i, h in enumerate(header) if h.startswith("人工判斷"))
    return {int(float(r[id_col])): r[human_col].strip() for r in rows[1:] if r and r[id_col]}


def pct(a, b):
    return f"{a}/{b} = {a / b:.1%}" if b else "—"


def score(path):
    human = read_review(path)
    with open(KEY_CSV, encoding="utf-8-sig", newline="") as f:
        key = {int(r["編號"]): r for r in csv.DictReader(f)}

    blank = [i for i in key if human.get(i, "") == ""]
    bad = {i: v for i, v in human.items() if v and v not in ("相關", "不相關", "邊界")}
    if bad:
        print(f"[錯誤] 有 {len(bad)} 則填了看不懂的值：{dict(list(bad.items())[:5])}")
        return
    if blank:
        print(f"[注意] 還有 {len(blank)} 則沒標，以下只算已標的 {len(key) - len(blank)} 則")

    # strict：邊界算錯；lenient：邊界算對
    stat = {g: Counter() for g in ("相關", "不相關")}
    wrong = []
    for i, k in key.items():
        h = human.get(i, "")
        if not h:
            continue
        s = stat[k["GPT判斷"]]
        s["n"] += 1
        s["strict"] += h == k["GPT判斷"]
        s["lenient"] += h in (k["GPT判斷"], "邊界")
        s["邊界"] += h == "邊界"
        if h != k["GPT判斷"]:
            wrong.append((i, k["GPT判斷"], h))

    n = sum(s["n"] for s in stat.values())
    if not n:
        print("沒有任何已標的列")
        return
    print(f"整體正確率          邊界算錯 {pct(sum(s['strict'] for s in stat.values()), n)}"
          f"   邊界算對 {pct(sum(s['lenient'] for s in stat.values()), n)}")
    for g, s in stat.items():
        print(f"GPT 判「{g}」的準確度  邊界算錯 {pct(s['strict'], s['n'])}"
              f"   邊界算對 {pct(s['lenient'], s['n'])}   （邊界 {s['邊界']} 則）")
    print(f"\n人工與 GPT 不一致的 {len(wrong)} 則（編號, GPT, 人工）：")
    for w in wrong:
        print(f"  {w[0]:>4}  {w[1]} → {w[2]}")


def main():
    parser = argparse.ArgumentParser(description="PTT 相關性標註盲測核對")
    parser.add_argument("--per-side", type=int, default=50, help="相關、不相關各抽幾則 (預設 50)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設 42)")
    parser.add_argument("--score", nargs="?", const=str(REVIEW_CSV), default=None,
                        help="對答案；可指定標完的檔案路徑（csv 或 xlsx）")
    args = parser.parse_args()

    if args.score:
        score(args.score)
    else:
        generate(args.per_side, args.seed)


if __name__ == "__main__":
    main()
