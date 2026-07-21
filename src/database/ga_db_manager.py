"""
ga_db_manager.py
GA 資料庫檢視與管理工具（適用扁平化結構）

功能：
  show                          - 顯示所有類別彙總
  show [類別]                   - 顯示該類別的 ptt_model 彙總
  show [類別] [ptt_model]       - 展開看該 ptt_model 的所有商品明細
  add                           - 互動式新增單一商品
  delete                        - 互動式刪除單一商品
  update                        - 互動式修改/新增/刪除商品欄位
  filter [類別] [欄位] [值]      - 篩選：刪除不符合條件的商品
  score scores.csv              - 批次寫入 neg_rate 與 score（依 ptt_model）
  export                        - 重新格式化輸出 JSON

使用範例：
  python ga_db_manager.py show
  python ga_db_manager.py show GPU
  python ga_db_manager.py show GPU RTX5080
  python ga_db_manager.py add
  python ga_db_manager.py delete
  python ga_db_manager.py update
  python ga_db_manager.py update --all
  python ga_db_manager.py filter CASE tempered_glass 是
  python ga_db_manager.py score scores.csv
  python ga_db_manager.py export
"""

import json
import os
import sys
import csv
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / 'data' / 'ga_database.json'

CAT_DISPLAY = {
    "CPU": "CPU", "GPU": "GPU", "RAM": "記憶體", "MB": "主機板",
    "SSD": "SSD", "HDD": "HDD", "AIR_COOLER": "風冷",
    "WATER_COOLER": "水冷", "CASE": "機殼", "PSU": "電源",
}

# 各類別允許的欄位（update 時做合法性檢查）
ALLOWED_FIELDS = {
    "CPU": [
        "socket", "tdp", "memory_type", "cores", "threads",
        "base_ghz", "boost_ghz", "cache_mb", "igpu",
    ],
    "GPU": [
        "length_mm", "vram_gb", "gpu_chip", "boost_mhz",
        "cuda_sp", "power_conn",
    ],
    "RAM": [
        "type", "speed", "capacity_gb", "single_gb", "kit", "cl",
    ],
    "MB": [
        "socket", "form_factor", "memory_type", "m2_slots", "wifi",
    ],
    "SSD": [
        "interface", "capacity_gb", "form_factor",
        "read_mbs", "write_mbs", "nand",
    ],
    "HDD": [
        "rpm", "capacity_gb", "cache_mb", "size",
    ],
    "AIR_COOLER": [
        "supported_sockets", "max_tdp", "height_mm",
        "cooler_type", "heat_pipes",
    ],
    "WATER_COOLER": [
        "supported_sockets", "radiator_size", "max_tdp",
        "fan_count", "lcd",
    ],
    "CASE": [
        "supported_mb", "max_gpu_mm", "max_cooler_mm", "max_water_mm",
        "psu_form_factor", "tempered_glass", "front_io",
    ],
    "PSU": [
        "wattage", "form_factor", "rating", "modular",
        "length_cm", "atx_version", "pcie5", "warranty_yr",
    ],
}

# 共用欄位（所有類別都有）
COMMON_FIELDS = [
    "ptt_model", "name", "brand", "series",
    "price", "comment_count", "neg_rate", "score",
]

# ── 讀寫 ──────────────────────────────────────────────

def load_db():
    if not os.path.exists(DB_PATH):
        print(f"[錯誤] 找不到 {DB_PATH}")
        sys.exit(1)
    with open(DB_PATH, encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print(f"✅ 已儲存至 {DB_PATH}")

# ── 工具 ──────────────────────────────────────────────

def parse_val(s):
    """字串轉型：嘗試 int → float → 原字串；'null' → None"""
    s = s.strip()
    if s.lower() == "null":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def pick_cat(db, prompt="類別："):
    print(f"可用類別：{', '.join(db.keys())}")
    cat = input(prompt).strip().upper()
    if cat not in db:
        print(f"[錯誤] 找不到類別 {cat}")
        return None
    return cat

def pick_ptt_model(db, cat):
    models = sorted(set(i["ptt_model"] for i in db[cat]))
    print(f"\n{CAT_DISPLAY.get(cat, cat)} 的 ptt_model 清單：")
    for idx, m in enumerate(models, 1):
        print(f"  {idx:3d}. {m}")
    raw = input("\nptt_model 名稱或編號：").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(models):
            return models[idx]
        print("[錯誤] 編號超出範圍")
        return None
    if raw in models:
        return raw
    print(f"[錯誤] 找不到 ptt_model：{raw}")
    return None

def pick_product(db, cat, ptt_model):
    items = [i for i in db[cat] if i["ptt_model"] == ptt_model]
    if not items:
        print("[錯誤] 找不到商品")
        return None, None
    print(f"\n「{ptt_model}」底下的商品：")
    for idx, item in enumerate(items, 1):
        print(f"  {idx:3d}. {item['name'][:55]}  NT${item['price']:,}")
    raw = input("\n商品編號：").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(items):
            return items[idx], idx
    print("[錯誤] 無效編號")
    return None, None

# ── show ──────────────────────────────────────────────

def cmd_show(args):
    db   = load_db()
    cat  = args[0].upper() if args else None
    ptt  = " ".join(args[1:]) if len(args) > 1 else None

    cats = [cat] if cat and cat in db else list(db.keys())
    if cat and cat not in db:
        print(f"[錯誤] 找不到類別 {cat}")
        return

    for c in cats:
        items = db[c]
        display = CAT_DISPLAY.get(c, c)

        # 展開單一 ptt_model 的商品明細
        if ptt:
            matched = [i for i in items if i["ptt_model"] == ptt]
            if not matched:
                print(f"[錯誤] 在 {c} 找不到 ptt_model：{ptt}")
                return
            print(f"\n{'='*65}")
            print(f"  {display} / {ptt}（{len(matched)} 個商品）")
            print(f"{'='*65}")
            for idx, item in enumerate(matched, 1):
                score_str = f"{item['score']:.1f}" if item.get("score") is not None else "未填"
                print(f"\n  [{idx}] {item['name']}")
                print(f"       NT${item['price']:,}  score:{score_str}  評論:{item['comment_count']}則")
                # 顯示規格欄位
                spec_keys = [k for k in item if k not in COMMON_FIELDS]
                for k in spec_keys:
                    v = item[k]
                    if v is not None:
                        print(f"       {k}: {v}")
            return

        # 彙總顯示各 ptt_model
        print(f"\n{'='*65}")
        print(f"  {display}（{len(items)} 個商品，共 {len(set(i['ptt_model'] for i in items))} 個 ptt_model）")
        print(f"{'='*65}")
        from itertools import groupby
        sorted_items = sorted(items, key=lambda x: -x["comment_count"])
        seen = {}
        for item in sorted_items:
            m = item["ptt_model"]
            if m not in seen:
                seen[m] = {"count": 0, "min_price": float("inf"),
                           "max_price": 0, "comment": item["comment_count"],
                           "score": item.get("score")}
            seen[m]["count"] += 1
            seen[m]["min_price"] = min(seen[m]["min_price"], item["price"])
            seen[m]["max_price"] = max(seen[m]["max_price"], item["price"])

        for m, info in seen.items():
            score_str = f"{info['score']:.1f}" if info["score"] is not None else "未填"
            price_str = (f"NT${info['min_price']:,}"
                        if info["min_price"] == info["max_price"]
                        else f"NT${info['min_price']:,}~{info['max_price']:,}")
            print(f"  {m:35s} {info['count']:3d}款  {price_str:20s}  "
                  f"score:{score_str:5s}  {info['comment']}則")

# ── add ───────────────────────────────────────────────

def cmd_add():
    db  = load_db()
    cat = pick_cat(db)
    if not cat:
        return

    ptt_model = pick_ptt_model(db, cat)
    if not ptt_model:
        return

    # 從既有商品取得共用資訊
    ref = next((i for i in db[cat] if i["ptt_model"] == ptt_model), None)
    if not ref:
        print(f"[錯誤] 找不到 {ptt_model} 的參考資料")
        return

    print(f"\n新增商品至 {CAT_DISPLAY.get(cat)} / {ptt_model}")
    print(f"（comment_count={ref['comment_count']}，neg_rate 和 score 將自動繼承）")

    name  = input("商品全名：").strip()
    brand = input(f"品牌（預設:{ref['brand']}）：").strip() or ref["brand"]
    price_raw = input("含稅價：").strip()
    try:
        price = int(price_raw)
    except ValueError:
        print("[錯誤] 價格必須是整數")
        return

    new_item = {
        "ptt_model":         ptt_model,
        "name":              name,
        "brand":             brand,
        "series":            ref["series"],
        "price":             price,
        "comment_count": ref["comment_count"],
        "neg_rate":          ref.get("neg_rate"),
        "score":             ref.get("score"),
    }

    print(f"\n請填入 {CAT_DISPLAY.get(cat)} 的規格欄位（直接 Enter 跳過）：")
    for field in ALLOWED_FIELDS.get(cat, []):
        raw = input(f"  {field}：").strip()
        if raw:
            new_item[field] = parse_val(raw)
        else:
            new_item[field] = None

    db[cat].append(new_item)
    save_db(db)
    print(f"✅ 已新增：{name}")

# ── delete ────────────────────────────────────────────

def cmd_delete():
    db  = load_db()
    cat = pick_cat(db)
    if not cat:
        return

    ptt_model = pick_ptt_model(db, cat)
    if not ptt_model:
        return

    item, _ = pick_product(db, cat, ptt_model)
    if not item:
        return

    ans = input(f"\n確定刪除「{item['name'][:50]}」？(y/N) ").strip().lower()
    if ans != "y":
        print("取消")
        return

    db[cat] = [i for i in db[cat] if i is not item]
    save_db(db)
    print(f"✅ 已刪除")

# ── update ────────────────────────────────────────────

def cmd_update(all_mode=False):
    """
    all_mode=False：修改單一商品的欄位
    all_mode=True ：對整個 ptt_model 所有商品批次操作
    """
    db  = load_db()
    cat = pick_cat(db)
    if not cat:
        return

    ptt_model = pick_ptt_model(db, cat)
    if not ptt_model:
        return

    if all_mode:
        targets = [i for i in db[cat] if i["ptt_model"] == ptt_model]
        print(f"\n對「{ptt_model}」底下 {len(targets)} 個商品批次操作")
    else:
        item, _ = pick_product(db, cat, ptt_model)
        if not item:
            return
        targets = [item]

    allowed = COMMON_FIELDS + ALLOWED_FIELDS.get(cat, [])

    print(f"\n操作類型：")
    print(f"  1. 修改/新增欄位值")
    print(f"  2. 刪除欄位")
    op = input("選擇 (1/2)：").strip()

    field = input("欄位名稱：").strip()

    if op == "2":
        # 刪除欄位
        if field in COMMON_FIELDS:
            print(f"[錯誤] {field} 是共用欄位，不能刪除")
            return
        for t in targets:
            t.pop(field, None)
        save_db(db)
        print(f"✅ 已從 {len(targets)} 個商品刪除欄位 {field}")
        return

    # 修改/新增欄位
    if field not in allowed:
        ans = input(f"⚠️  {field} 不在 {cat} 的允許欄位清單中，確定要新增嗎？(y/N) ").strip().lower()
        if ans != "y":
            print("取消")
            return

    raw = input(f"新的值（目前：{targets[0].get(field, '（無）')}）：").strip()
    new_val = parse_val(raw)

    for t in targets:
        t[field] = new_val

    save_db(db)
    print(f"✅ 已更新 {len(targets)} 個商品的 {field} = {new_val}")

# ── filter ────────────────────────────────────────────

def cmd_filter(args):
    """
    刪除不符合條件的商品
    用法：filter [類別] [欄位] [值]
    """
    if len(args) < 3:
        print("[錯誤] 用法：python ga_db_manager.py filter [類別] [欄位] [值]")
        print("  例如：python ga_db_manager.py filter CASE tempered_glass 是")
        return

    cat   = args[0].upper()
    field = args[1]
    value = parse_val(args[2])

    db = load_db()
    if cat not in db:
        print(f"[錯誤] 找不到類別 {cat}")
        return

    before = len(db[cat])
    db[cat] = [i for i in db[cat] if i.get(field) == value]
    after   = len(db[cat])
    removed = before - after

    ans = input(f"將刪除 {removed} 個商品（保留 {after} 個），確定？(y/N) ").strip().lower()
    if ans != "y":
        print("取消")
        return

    save_db(db)
    print(f"✅ 已刪除 {removed} 個不符合「{field}={value}」的商品")

# ── score ─────────────────────────────────────────────

def cmd_score(csv_path):
    """
    CSV 格式（含標題列）：
      category,ptt_model,neg_rate,score
      GPU,RTX5080,0.0841,91.59
    """
    if not os.path.exists(csv_path):
        print(f"[錯誤] 找不到 {csv_path}")
        return

    db      = load_db()
    updated = 0
    not_found = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat       = row.get("category", "").strip().upper()
            ptt_model = row.get("ptt_model", "").strip()
            try:
                neg_rate = float(row.get("neg_rate", ""))
                score    = float(row.get("score", ""))
            except ValueError:
                print(f"[略過] {cat}/{ptt_model}：neg_rate 或 score 格式錯誤")
                continue

            if cat not in db:
                not_found.append(f"{cat}/{ptt_model}（類別不存在）")
                continue

            targets = [i for i in db[cat] if i["ptt_model"] == ptt_model]
            if not targets:
                not_found.append(f"{cat}/{ptt_model}（ptt_model 不存在）")
                continue

            for item in targets:
                item["neg_rate"] = round(neg_rate, 6)
                item["score"]    = round(score, 4)
            updated += len(targets)
            print(f"  ✅ {cat}/{ptt_model}: {len(targets)} 個商品寫入 score={score:.2f}")

    save_db(db)
    print(f"\n✅ 共更新 {updated} 個商品的分數")
    if not_found:
        print(f"⚠️  找不到的 ptt_model（{len(not_found)} 個）：")
        for nf in not_found:
            print(f"  {nf}")

# ── export ────────────────────────────────────────────

def cmd_export():
    db = load_db()
    save_db(db)
    total = sum(len(v) for v in db.values())
    print(f"共 {total} 個商品")
    for cat, items in db.items():
        models  = len(set(i["ptt_model"] for i in items))
        scored  = sum(1 for i in items if i.get("score") is not None)
        print(f"  {CAT_DISPLAY.get(cat, cat):8s}: {len(items):4d} 個商品  "
              f"{models:3d} 個 ptt_model  {scored} 個已評分")

# ── 主程式 ────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or args[0] in ("help", "--help", "-h"):
        print(__doc__)
        return

    cmd  = args[0].lower()
    rest = args[1:]

    if cmd == "show":
        cmd_show(rest)
    elif cmd == "add":
        cmd_add()
    elif cmd == "delete":
        cmd_delete()
    elif cmd == "update":
        all_mode = "--all" in rest
        cmd_update(all_mode)
    elif cmd == "filter":
        cmd_filter(rest)
    elif cmd == "score":
        if not rest:
            print("[錯誤] 請提供 CSV 路徑，例如：python ga_db_manager.py score scores.csv")
        else:
            cmd_score(rest[0])
    elif cmd == "export":
        cmd_export()
    else:
        print(f"[錯誤] 不認識的指令：{cmd}")
        print("執行 python ga_db_manager.py help 查看說明")


if __name__ == "__main__":
    main()