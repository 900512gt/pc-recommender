"""
merge_and_score.py
合併 PTT relevant.jsonl + 巴哈 matched_part*.jsonl → 輸出 scores.csv

PTT label：正評 / 負評 / 中立（已標記，直接使用）
巴哈 label：正評 / 負評 / 中立（已標記，直接使用）
巴哈 category 需從中文對映至 DB 英文 key

MIN_OPINIONS = 5（正評+負評合計，中立不計）

使用流程：
  1. python merge_and_score.py                   → 產生 scores.csv
  2. python ga_db_manager.py score data/raw/scores.csv    → 寫回 DB
"""

import json, csv
from pathlib import Path
from collections import defaultdict

PTT_FILE     = "input/ptt_comment/relevant.jsonl"
BAHA_FILES   = [f"input/baha_comment/matched_part{i}.jsonl" for i in [1,2,3]]
_ROOT        = Path(__file__).resolve().parents[2]
DB_SRC       = _ROOT / 'data' / 'ga_database.json'
CSV_OUT      = _ROOT / 'data' / 'raw' / 'scores.csv'
MIN_OPINIONS = 5
ALPHA        = 20   # 先驗強度，值越大越往 prior 拉

BAHA_CAT_MAP = {
    "CPU": "CPU", "GPU": "GPU", "SSD": "SSD", "HDD": "HDD",
    "主機板": "MB", "記憶體": "RAM", "機殼": "CASE",
    "電源": "PSU", "風冷": "AIR_COOLER", "水冷": "WATER_COOLER",
}

def norm_label(l):
    if l == "正評":            return "正評"
    if l in ("負評", "负評"):  return "負評"
    if l == "中立":            return "中立"
    return None

# ── 讀取 DB（僅取型號清單做過濾用）────────────────────
with open(DB_SRC, encoding="utf-8") as f:
    db = json.load(f)
db_models = {cat: set(i["ptt_model"] for i in items) for cat, items in db.items()}

# ── 統計容器 ─────────────────────────────────────────
stat = defaultdict(lambda: {"正評": 0, "負評": 0, "中立": 0, "ptt": 0, "baha": 0, "total": 0})

def accumulate(lines, source, cat_fn):
    total, skip = 0, 0
    for line in lines:
        d     = json.loads(line)
        cat   = cat_fn(d.get("category", ""))
        model = d.get("model", "").strip()
        label = norm_label(d.get("label", ""))
        if not cat or not label:
            continue
        if model not in db_models.get(cat, set()):
            skip += 1
            continue
        stat[(cat, model)][label]  += 1
        stat[(cat, model)][source] += 1
        stat[(cat, model)]["total"] += 1
        total += 1
    return total, skip

# ── 1. PTT ────────────────────────────────────────────
with open(PTT_FILE, encoding="utf-8") as f:
    pt, ps = accumulate(f, "ptt", lambda c: c if c in db_models else None)
print(f"PTT  ：讀入 {pt:,} 筆（略過 DB 外型號 {ps:,} 筆）")

# ── 2. 巴哈 ──────────────────────────────────────────
bt = bs = 0
for path in BAHA_FILES:
    with open(path, encoding="utf-8") as f:
        a, b = accumulate(f, "baha", lambda c: BAHA_CAT_MAP.get(c))
        bt += a; bs += b
print(f"巴哈 ：讀入 {bt:,} 筆（略過 DB 外型號 {bs:,} 筆）")

# ── 3. 計算分數（Bayesian smoothing 兩遍法）──────────────
results, skipped = [], []

# 第一遍：累積各類別的 pos/neg 總數（只算過門檻的商品）
cat_totals = defaultdict(lambda: {"pos": 0, "neg": 0})
for (cat, model), s in stat.items():
    pos, neg = s["正評"], s["負評"]
    if pos + neg < MIN_OPINIONS:
        continue
    cat_totals[cat]["pos"] += pos
    cat_totals[cat]["neg"] += neg

# 各類別先驗負評率
prior = {}
for cat, t in cat_totals.items():
    total_op = t["pos"] + t["neg"]
    prior[cat] = (t["neg"] / total_op) if total_op > 0 else 0.5

# 印出各類別先驗，方便肉眼確認
print("\n📊 各類別先驗負評率（prior_neg_rate）：")
for cat in sorted(prior):
    t = cat_totals[cat]
    print(f"  {cat:<14} {prior[cat]*100:5.1f}%  "
          f"（正={t['pos']:,} 負={t['neg']:,} 合={t['pos']+t['neg']:,}）")

# 第二遍：套用 Bayesian smoothing
for (cat, model), s in sorted(stat.items()):
    pos, neg = s["正評"], s["負評"]
    opinions = pos + neg
    if opinions < MIN_OPINIONS:
        skipped.append((cat, model, pos, neg, s["中立"], s["ptt"], s["baha"]))
        continue
    p               = prior.get(cat, 0.5)
    smoothed_neg    = (neg + ALPHA * p) / (opinions + ALPHA)
    raw_neg_rate    = round(neg / opinions, 6)
    neg_rate        = round(smoothed_neg, 6)
    score           = round((1 - smoothed_neg) * 100, 4)
    results.append({
        "category":      cat,
        "ptt_model":     model,
        "neg_rate":      neg_rate,
        "raw_neg_rate":  raw_neg_rate,
        "score":         score,
        "comment_count": s["total"],
        "正評": pos, "負評": neg, "中立": s["中立"],
        "有效評論數": opinions,
        "PTT筆數":  s["ptt"],
        "巴哈筆數": s["baha"],
        "prior_neg_rate": round(p, 6),
        "alpha":          ALPHA,
    })

# 印出 smoothing 前後差異最大的前 10 個商品
deltas = sorted(results, key=lambda r: abs(r["score"] - (1 - r["raw_neg_rate"]) * 100), reverse=True)
print("\n🔍 Smoothing 前後差異最大前 10 名：")
print(f"  {'類別':<14} {'型號':<35} {'opinions':>8}  raw_score → smoothed   (Δ)")
print("  " + "─" * 80)
for r in deltas[:10]:
    raw_score     = round((1 - r["raw_neg_rate"]) * 100, 4)
    delta         = r["score"] - raw_score
    print(f"  [{r['category']:<12}] {r['ptt_model']:<35} "
          f"opinions={r['有效評論數']:>4}  "
          f"{raw_score:>6.2f} → {r['score']:>6.2f}  "
          f"(Δ={delta:+.2f})")

# ── 4. 輸出 scores.csv ───────────────────────────────
with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "category", "ptt_model", "neg_rate", "raw_neg_rate", "score",
        "comment_count", "正評", "負評", "中立", "有效評論數",
        "PTT筆數", "巴哈筆數", "prior_neg_rate", "alpha",
    ])
    writer.writeheader()
    writer.writerows(results)

# ── 5. 報表 ───────────────────────────────────────────
print(f"\n✅ {CSV_OUT}：{len(results)} 個型號\n")

print(f"{'類別':<14} {'型號':<35} {'score':>6}  {'neg%(s)':>7}  {'neg%(r)':>7}  {'正':>4} {'負':>4} {'中':>4}  {'PTT':>5} {'巴哈':>5}")
print("─" * 100)
prev = None
for r in results:
    if r["category"] != prev:
        print()
        prev = r["category"]
    print(f"{r['category']:<14} {r['ptt_model']:<35} "
          f"{r['score']:>6.1f}  {r['neg_rate']*100:>6.1f}%  {r['raw_neg_rate']*100:>6.1f}%  "
          f"{r['正評']:>4} {r['負評']:>4} {r['中立']:>4}  "
          f"{r['PTT筆數']:>5} {r['巴哈筆數']:>5}")

if skipped:
    print(f"\n⚠️  有效評論數 < {MIN_OPINIONS}，略過 {len(skipped)} 個型號（score 維持 null）：")
    for cat, model, pos, neg, neu, ptt, baha in skipped:
        print(f"  [{cat}] {model}：正={pos} 負={neg} 中={neu} (PTT={ptt} 巴哈={baha})")

print(f"\n下一步：")
print(f"  python ga_db_manager.py score {CSV_OUT}")