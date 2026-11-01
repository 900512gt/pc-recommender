"""
修正 relevant.jsonl 的兩類問題，輸出新檔，不動原檔：

1. 前綴/後綴誤配（RTX5070 vs RTX5070Ti、9950X vs 9950X3D、NR200 vs NR200P ...）
   (a) 文章層：標題只提到兄弟型號、沒提到自己 → 這篇不是在講自己
       - 推文內容有明確提到自己 → 保留
       - 推文有提到兄弟型號 → 走 (b)
       - 推文沒提型號、標題只指向一個資料庫內的兄弟型號 → 改歸屬給它
       - 其他 → 刪除
   兄弟型號包含資料庫外的變體：GPU 的 Ti/Super/XT…、CPU 的 K/KF/F/X3D…
   （例如 14700K、12400F 不在資料庫，它們的文章不算 i7-14700 / i5-12400），
   以及 GPU 的簡寫（RTX5070 底下的「70ti」）
   (b) 推文層：內容只提到兄弟型號、沒提到自己 → 刪除
       不改歸屬，因為 annotate.py 的 label 是針對「原型號」判的，
       例如 5070Ti 串裡的「價差買5070還能省2K」對 5070Ti 是負評、對 5070 卻是正評
2. 同型號重複推文：(model, content) 相同只留一筆
   - 若其中有「原本就屬於該型號」的列，優先保留它，標籤只看這些列
     （改歸屬過來的列，標籤是針對舊型號判的，不參與投票）
   - 標籤取多數決；平手保留 _index 最小那筆並列入複查清單
3. 標籤格式：簡體「负評」→「負評」；標籤為 null 的列刪除

用法（專案根目錄）：
    python src/filter/fix_relevant.py
輸出：
    src/database/input/ptt_comment/relevant_fixed.jsonl
    src/filter/output/relevant_fix_review.csv   ← 每一筆改動/刪除，供人工複查
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
IN_FILE    = ROOT / "src/database/input/ptt_comment/relevant.jsonl"
OUT_FILE   = ROOT / "src/database/input/ptt_comment/relevant_fixed.jsonl"
REVIEW_CSV = ROOT / "src/filter/output/relevant_fix_review.csv"

# GPU 常見的變體後綴；就算該變體不在資料庫（例如 5080Ti、4070S），也視為「另一張卡」
GPU_SUFFIXES = ["", "ti", "super", "s", "xt", "xtx", "gre", "tisuper"]
# CPU 同理（14700K、12400F…）；不含「光數字」，因為常有人把 7800X3D 簡稱成 7800
CPU_SUFFIXES = ["k", "kf", "ks", "f", "t", "x", "x3d", "g", "gt", "xt"]
# 數字後面接這些字時，允許中間有空格（「5070 Ti」→「5070ti」）
# 單字母後綴（B850M、14400F…）不放進來，避免「5070 m.2」被誤接成 5070m
SPACED_SUFFIX = re.compile(r"(?<=\d)[\s\-]+(?=(?:ti|super|xtx|xt|gre|x3d|kf)(?![a-z]))")


# ══════════════════════════════════════════════════════
#  型號關鍵字與比對
# ══════════════════════════════════════════════════════

def core_of(model):
    """型號 → 比對用關鍵字，例如 RTX5070Ti→5070ti、Intel i5-14400F→14400f、Montech Air 100→air100
    （Core Ultra 5 225 → 225，前一個字是數字時不合併）"""
    tokens = model.lower().split()
    core = tokens[-1].split("-")[-1]
    core = re.sub(r"^(rtx|rx)", "", core)
    if core.isdigit() and len(core) < 4 and len(tokens) >= 2 and tokens[-2].isalpha():
        core = tokens[-2] + core
    return core


def normalize(text):
    t = (text or "").lower()
    return SPACED_SUFFIX.sub("", t)


def mentions(core, norm_text):
    """core 以獨立型號形式出現：前面不是數字、後面不是英數字（所以 5070 不會命中 5070ti）
    字母接數字處允許空格（「air 100」也算 air100）"""
    pat = r"\s*".join(re.escape(p) for p in re.findall(r"[a-z]+|[^a-z]+", core))
    return re.search(rf"(?<!\d){pat}(?![a-z0-9])", norm_text) is not None


def build_siblings(models_by_cat):
    """每個型號 → 兄弟關鍵字集合；另外記錄哪些關鍵字對應到資料庫內型號"""
    siblings = {}
    core_to_model = {}
    for cat, models in models_by_cat.items():
        cores = {m: core_of(m) for m in models}
        for m, c in cores.items():
            core_to_model[(cat, c)] = m
        for m, c in cores.items():
            sib = {c2 for m2, c2 in cores.items()
                   if m2 != m and (c2.startswith(c) or c.startswith(c2))}
            if cat == "GPU":
                base = re.match(r"^(\d{4})[a-z]*$", c)
                if base:
                    sib |= {base.group(1) + s for s in GPU_SUFFIXES}
                    if c.isdigit():
                        sib.add(c[2:] + "ti")    # 簡寫：RTX5070 串裡的「70ti」
                    sib.discard(c)
            elif cat == "CPU":
                base = re.match(r"^(\d{3,5})", c)
                if base:
                    sib |= {base.group(1) + s for s in CPU_SUFFIXES}
                    sib.discard(c)
            siblings[m] = sib
    return siblings, core_to_model


# ══════════════════════════════════════════════════════
#  第一類：前綴/後綴誤配
# ══════════════════════════════════════════════════════

def fix_variants(rows, siblings, core_to_model, cat_info, review):
    out = []
    for x in rows:
        m, cat = x["model"], x["category"]
        sib = siblings.get(m, set())
        if not sib:
            out.append(x)
            continue
        core = core_of(m)
        title, content = normalize(x["title"]), normalize(x["content"])

        content_self = mentions(core, content)
        content_sib = sorted(s for s in sib if mentions(s, content))
        title_self = mentions(core, title)
        title_sib = sorted(s for s in sib if mentions(s, title))

        if content_sib and not content_self:
            review.append(("刪除(推文誤配)", m, "", f"推文只提到 {'/'.join(content_sib)}", x))
            continue
        if not title_self and title_sib and not content_self:
            reason = f"標題只提到 {'/'.join(title_sib)}"
            target = title_sib[0] if len(title_sib) == 1 else None
        else:
            out.append(x)
            continue

        new_model = core_to_model.get((cat, target)) if target else None
        if new_model:
            y = dict(x, model=new_model, _reassigned=True, **cat_info[new_model])
            out.append(y)
            review.append(("改歸屬", m, new_model, reason, x))
        else:
            review.append(("刪除(文章誤配)", m, "", reason, x))
    return out


# ══════════════════════════════════════════════════════
#  第二類：同型號重複推文
# ══════════════════════════════════════════════════════

def dedup(rows, review):
    groups = defaultdict(list)
    for x in rows:
        groups[(x["model"], x["content"].strip())].append(x)

    keep_ids = {}
    for key, xs in groups.items():
        xs.sort(key=lambda r: (r.get("_reassigned", False), r["_index"]))
        first = xs[0]
        if len(xs) > 1:
            voters = [r for r in xs if not r.get("_reassigned")] or xs
            votes = Counter(r["label"] for r in voters).most_common()
            tie = len(votes) > 1 and votes[0][1] == votes[1][1]
            label = first["label"] if tie else votes[0][0]
            vote_str = "、".join(f"{l}×{n}" for l, n in votes)
            if label != first["label"] or tie:
                note = f"標籤投票 {vote_str}" + (" → 平手，保留第一筆，請人工確認" if tie else f" → 採 {label}")
                review.append(("標籤調整", first["model"], "", note, first))
            first = dict(first, label=label)
            for dup in xs[1:]:
                review.append(("刪除(重複)", dup["model"], "",
                               f"與 _index={xs[0]['_index']} 重複", dup))
        keep_ids[id(xs[0])] = first

    # 保持原始順序
    return [keep_ids[id(x)] for x in rows if id(x) in keep_ids]


# ══════════════════════════════════════════════════════
#  標籤格式
# ══════════════════════════════════════════════════════

LABEL_FIX = {"负評": "負評", "正评": "正評"}


def fix_labels(rows, review):
    out = []
    for x in rows:
        label = LABEL_FIX.get(x["label"], x["label"])
        if label not in ("正評", "負評", "中立"):
            review.append(("刪除(標籤無效)", x["model"], "", f"label={x['label']!r}", x))
            continue
        if label != x["label"]:
            review.append(("標籤調整", x["model"], "", f"{x['label']} → {label}", x))
            x = dict(x, label=label)
        out.append(x)
    return out


# ══════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════

def main():
    rows = [json.loads(l) for l in open(IN_FILE, encoding="utf-8") if l.strip()]

    models_by_cat = defaultdict(set)
    cat_info = {}
    for x in rows:
        models_by_cat[x["category"]].add(x["model"])
        cat_info.setdefault(x["model"], {"brand": x["brand"], "series": x["series"]})
    siblings, core_to_model = build_siblings(models_by_cat)

    review = []
    fixed = fix_labels(rows, review)
    fixed = fix_variants(fixed, siblings, core_to_model, cat_info, review)
    fixed = dedup(fixed, review)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for x in fixed:
            x.pop("_reassigned", None)
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["處理", "原型號", "新型號", "原因", "_index", "標籤label", "文章標題", "內容", "url"])
        for action, old, new, reason, x in review:
            w.writerow([action, old, new, reason, x["_index"], x["label"],
                        x["title"], x["content"], x["url"]])

    cnt = Counter(a for a, *_ in review)
    print(f"輸入 {len(rows)} 筆 → 輸出 {len(fixed)} 筆")
    for a, n in cnt.most_common():
        print(f"  {a}：{n}")
    print(f"輸出：{OUT_FILE.relative_to(ROOT)}")
    print(f"複查清單：{REVIEW_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
