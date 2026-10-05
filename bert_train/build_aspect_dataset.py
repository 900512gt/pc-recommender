# build_aspect_dataset.py — 把 p3 面向標註整理成 BERT 訓練資料
#
# 前提假設（兩階段）：
#   第一階段（相關性）已經判定「這則評論有沒有在評價這個型號」
#   第二階段（BERT）只負責判斷「對這個型號的這個面向是正面/負面/未提及」
#
# 處理步驟：
#   1. 不相關的資料：
#        本來就沒面向          → 保留當負例
#        PTT 不相關但有面向    → 面向全部改成「未提及」（困難負例）
#        巴哈 不相關但有面向   → 不放進訓練，另存待確認
#   2. 相關資料裡的高風險筆數（疑似把別的型號的評價記到自己頭上）→ 不放進訓練，另存待重標
#   3. 每筆依面向展開成多筆：query = "型號 面向"，text = "標題 內文"；內文太長就切視窗
#      （訓練時用 tokenizer(query, text) 就會變成 [CLS] 型號 面向 [SEP] 標題 內文 [SEP]）
#   4. 以討論串為單位切 train / val / test，同一串不會跨集合
#   5. 只在 train 對「未提及」做減量抽樣；val / test 保持原始分布
#
# 用法：
#   python build_aspect_dataset.py
#   python build_aspect_dataset.py --neg-ratio 0      # 不做減量抽樣
#   python build_aspect_dataset.py --max-chars 300    # 內文視窗改短

import re
import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
UNMENTIONED = "未提及"


# ── 讀寫 ──
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── 型號在文字裡的寫法 ──
def model_numbers(model: str) -> set:
    """型號裡的數字段，例如 'AMD R7 7800X3D' → {'7800'}，'DDR5-6000' → {'6000'}"""
    return set(re.findall(r"\d{3,5}", model))


def mention_regex(model: str):
    """
    找出型號在內文出現的位置用的 regex。
    論壇上很少寫全名（RTX5070Ti 常寫成 5070ti / 5070 Ti），所以以數字段＋後綴為主，
    沒有數字段的型號（海韻 CORE GX）退回用名稱裡的字。
    """
    pats = [re.escape(model)]
    for m in re.finditer(r"(\d{3,5})([A-Za-z0-9]*)", model):
        digits, suffix = m.groups()
        pats.append(rf"(?<!\d){digits}\s?{re.escape(suffix)}(?![0-9A-Za-z])")
    if len(pats) == 1:
        tokens = model.split()
        tokens = tokens[1:] if len(tokens) > 1 else tokens  # 第一個字通常是品牌
        pats += [re.escape(t) for t in tokens if len(t) >= 3]
    return re.compile("|".join(pats), re.IGNORECASE)


def windows(content: str, model: str, max_chars: int) -> list:
    """
    內文太長時切成多個視窗，每個視窗從某次型號出現位置的前面一點開始
    （評價通常寫在型號後面，所以前 1/3、後 2/3），已被前一個視窗蓋到的出現位置不再另開。
    只用型號位置決定視窗，不用 evidence —— 推論時沒有 evidence 可看，訓練要跟推論一致。
    """
    if len(content) <= max_chars:
        return [content]
    starts = [m.start() for m in mention_regex(model).finditer(content)]
    if not starts:
        return [content[:max_chars]]

    wins, covered = [], -1
    for s in starts:
        if s < covered:
            continue
        a = min(max(0, s - max_chars // 3), max(0, len(content) - max_chars))
        wins.append(content[a:a + max_chars])
        covered = a + max_chars
    return wins


def locate(evidence: str, wins: list):
    """證據落在第幾個視窗；逐字找不到時改用片段比對（證據常被「…」拼接或加引號）。找不到回傳 None。"""
    for i, w in enumerate(wins):
        if evidence in w:
            return i
    frags = [f.strip() for f in re.split(r"…+|\.{2,}|[「」；;]", evidence) if len(f.strip()) >= 6]
    hits = [sum(f in w for f in frags) for w in wins]
    return hits.index(max(hits)) if frags and max(hits) > 0 else None


# ── 步驟 2：高風險筆數 ──
def find_high_risk(rows):
    """
    回傳 {row 的 index: 原因}。兩種情況都是「證據很可能在講別的型號」：
      shared_evidence      同一則評論配到另一個同類別型號，而且兩筆用了一模一樣的 (面向, 證據)
      other_model_evidence 證據裡寫的是同類別別的型號數字，卻沒有寫到自己的型號
    這是啟發式規則，會誤抓到正常的比較句，所以挑出來的是「待重標」而不是「確定錯」。
    """
    risk = {}

    by_content = defaultdict(list)
    for i, r in enumerate(rows):
        by_content[(r["content"], r["category"])].append(i)
    for idxs in by_content.values():
        for a in idxs:
            for b in idxs:
                if a < b and rows[a]["model"] != rows[b]["model"]:
                    if set(rows[a]["evidence"].items()) & set(rows[b]["evidence"].items()):
                        risk[a] = risk[b] = "shared_evidence"

    nums_by_cat = defaultdict(set)
    for r in rows:
        nums_by_cat[r["category"]] |= {n for n in model_numbers(r["model"]) if len(n) >= 4}
    for i, r in enumerate(rows):
        if i in risk:
            continue
        own = model_numbers(r["model"])
        if not own:
            continue
        for ev in r["evidence"].values():
            others = set(re.findall(r"\d{4,5}", ev)) & nums_by_cat[r["category"]]
            if others and not (others & own):
                risk[i] = "other_model_evidence"
                break
    return risk


# ── 步驟 4：以討論串分組 ──
def assign_groups(rows):
    """
    同一個 url（同一串）一定同組；另外內文完全相同的長文也併成同組
    （同一篇文配到多個型號、或被引用轉貼），避免同一段文字同時出現在 train 和 test。
    短推文不併（「好便宜」這種會把不相干的串全部連在一起）。
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = []
    for r in rows:
        thread = "u:" + (r.get("url") or r.get("title") or r["content"])
        keys.append(thread)
        find(thread)
        if len(r["content"]) >= 30:
            parent[find("c:" + r["content"])] = find(thread)
    return [find(k) for k in keys]


def split_groups(groups, seed, val_ratio, test_ratio):
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    n_test = int(len(uniq) * test_ratio)
    n_val = int(len(uniq) * val_ratio)
    split = {}
    for i, g in enumerate(uniq):
        split[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return split


# ── 主流程 ──
def main(args):
    rows = load_jsonl(args.input)
    print(f"讀入 {len(rows)} 筆：{args.input}")

    # 步驟 1：不相關的資料
    kept, held_baha = [], []
    n_zeroed = 0
    for r in rows:
        r = dict(r)
        r["zeroed"] = False
        if not r["orig_relevant"] and r["evidence"]:
            if r["source"] == "baha":
                held_baha.append(r)
                continue
            r["aspects_p3"], r["evidence_p3"] = r["aspects"], r["evidence"]
            r["aspects"] = {a: UNMENTIONED for a in r["aspects"]}
            r["evidence"] = {}
            r["zeroed"] = True
            n_zeroed += 1
        kept.append(r)

    # 步驟 2：相關資料裡的高風險筆數
    risk = find_high_risk(kept)
    held_risk = [dict(kept[i], risk_reason=why) for i, why in sorted(risk.items())]
    kept = [r for i, r in enumerate(kept) if i not in risk]

    # 步驟 3 + 4：展開成 (型號, 面向) 一筆，並依討論串分到 train / val / test
    groups = assign_groups(kept)
    split = split_groups(groups, args.seed, args.val_ratio, args.test_ratio)

    examples = {"train": [], "val": [], "test": []}
    n_long = n_long_labeled = n_long_dropped = 0
    for r, g in zip(kept, groups):
        wins = windows(r["content"], r["model"], args.max_chars)
        is_long = len(r["content"]) > args.max_chars
        n_long += is_long
        for aspect, label in r["aspects"].items():
            # 短文：整則就是一個視窗。長文：有標註的面向只配到證據所在的視窗，
            # 證據不在任何視窗裡就不產生樣本（看不到依據的標註不能拿來訓練）；
            # 未提及的面向配第一個視窗。
            wi = 0
            if is_long and label != UNMENTIONED:
                n_long_labeled += 1
                wi = locate(r["evidence"][aspect], wins)
                if wi is None:
                    n_long_dropped += 1
                    continue
            examples[split[g]].append({
                "query": f"{r['model']} {aspect}",
                "text": f"{r['title']} {wins[wi]}" if r.get("title") else wins[wi],
                "label": label,
                "source": r["source"],
                "category": r["category"],
                "model": r["model"],
                "aspect": aspect,
                "orig_relevant": r["orig_relevant"],
                "zeroed": r["zeroed"],
                "url": r.get("url"),
            })

    # 步驟 5：只在 train 對「未提及」減量
    train_before = Counter(e["label"] for e in examples["train"])
    if args.neg_ratio > 0:
        labeled = [e for e in examples["train"] if e["label"] != UNMENTIONED]
        hard = [e for e in examples["train"] if e["label"] == UNMENTIONED and e["zeroed"]]
        rest = [e for e in examples["train"] if e["label"] == UNMENTIONED and not e["zeroed"]]
        quota = max(0, int(len(labeled) * args.neg_ratio) - len(hard))
        rng = random.Random(args.seed)
        rest = rng.sample(rest, min(quota, len(rest)))
        examples["train"] = labeled + hard + rest
        rng.shuffle(examples["train"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, exs in examples.items():
        write_jsonl(out / f"{name}.jsonl", exs)
    write_jsonl(out / "held_out_baha_irrelevant.jsonl", held_baha)
    write_jsonl(out / "held_out_high_risk.jsonl", held_risk)

    # ── 報告 ──
    print("\n=== 評論層級（一筆 = 一則評論 × 一個型號）===")
    print(f"  原始                              {len(rows):>7}")
    print(f"  巴哈 不相關但有面向 → 暫不訓練     {len(held_baha):>7}")
    print(f"  高風險 → 暫不訓練                 {len(held_risk):>7}  {dict(Counter(r['risk_reason'] for r in held_risk))}")
    print(f"  送進訓練流程                      {len(kept):>7}")
    print(f"    其中 PTT 不相關改成全未提及      {n_zeroed:>7}")
    for (src, rel), n in sorted(Counter((r['source'], r['orig_relevant']) for r in kept).items()):
        print(f"    {src:<5}{'相關' if rel else '不相關'}  {n:>7}")
    n_group = Counter(split[g] for g in set(groups))
    n_rows = Counter(split[g] for g in groups)
    print(f"  討論串數 {dict(n_group)}")
    print(f"  評論數   {dict(n_rows)}")

    print("\n=== 樣本層級（一筆 = 一則評論 × 一個型號 × 一個面向）===")
    print(f"  train 減量前 {dict(train_before)}")
    for name, exs in examples.items():
        c = Counter(e["label"] for e in exs)
        print(f"  {name:<5} {len(exs):>7}  " + "  ".join(f"{k} {c[k]}" for k in [UNMENTIONED, "正面", "負面"]))

    print(f"\n內文超過 {args.max_chars} 字、切視窗處理的評論：{n_long}")
    print(f"  這些評論裡有標註的面向 {n_long_labeled} 個，"
          f"證據不在任何視窗內而捨棄 {n_long_dropped} 個")
    print(f"\n輸出目錄：{out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "aspects_標註_p3.jsonl"))
    ap.add_argument("--output-dir", default=str(HERE / "aspect_dataset"))
    ap.add_argument("--max-chars", type=int, default=400, help="內文最多保留幾個字")
    ap.add_argument("--neg-ratio", type=float, default=3.0,
                    help="train 裡 未提及 : (正面+負面) 的上限，0 表示不減量")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
