"""
PTT 硬體評論爬蟲
版本：v1.0（第一階段：骨架 + CPU）

使用方式：
  python main.py CPU              # 只跑 CPU
  python main.py CPU GPU SSD      # 跑多個類別
  python main.py --all            # 跑所有有資料的類別

支援類別：
  CPU  GPU  RAM  MB  SSD  HDD
  AIR_COOLER  WATER_COOLER  CASE  PSU

依賴：
  pip install requests beautifulsoup4 openpyxl
"""

import sys
from datetime import datetime

from hardware_config import (
    VALID_CATEGORIES,
    CATEGORY_DISPLAY,
    CATEGORY_ITEMS,
)
from scraper  import scrape_category
from exporter import export_excel

# ══════════════════════════════════════════════════════
#  命令列解析
# ══════════════════════════════════════════════════════

def parse_args(argv):
    """
    回傳要執行的類別清單（大寫字串）
    若參數有誤則印出說明並結束程式
    """
    args = argv[1:]

    if not args:
        _print_usage()
        sys.exit(0)

    if "--all" in args:
        # 只跑有型號清單的類別（空清單的類別自動跳過）
        return [c for c in VALID_CATEGORIES if CATEGORY_ITEMS.get(c)]

    categories = []
    unknown    = []
    for a in args:
        upper = a.upper()
        if upper in VALID_CATEGORIES:
            categories.append(upper)
        else:
            unknown.append(a)

    if unknown:
        print(f"[錯誤] 不認識的類別：{', '.join(unknown)}")
        print(f"       支援類別：{', '.join(VALID_CATEGORIES)}")
        sys.exit(1)

    if not categories:
        _print_usage()
        sys.exit(0)

    return categories


def _print_usage():
    print("使用方式：")
    print("  python main.py <類別1> [類別2 ...]")
    print("  python main.py --all")
    print()
    print("支援類別：")
    for cat in VALID_CATEGORIES:
        count = len(CATEGORY_ITEMS.get(cat, []))
        ready = f"{count} 個型號" if count else "（尚未建立）"
        print(f"  {cat:<15} {CATEGORY_DISPLAY.get(cat, cat)}  {ready}")

# ══════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════

def main():
    categories = parse_args(sys.argv)

    # 過濾掉沒有型號清單的類別
    runnable = []
    skipped  = []
    for cat in categories:
        if CATEGORY_ITEMS.get(cat):
            runnable.append(cat)
        else:
            skipped.append(cat)

    if skipped:
        print(f"[略過] 以下類別尚無型號清單：{', '.join(skipped)}")

    if not runnable:
        print("[結束] 沒有可執行的類別。")
        sys.exit(0)

    # ── 開始爬蟲 ─────────────────────────────────────
    print("=" * 60)
    print(f"  PTT 硬體評論爬蟲")
    print(f"  執行類別：{', '.join(CATEGORY_DISPLAY.get(c, c) for c in runnable)}")
    print(f"  開始時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    for cat in runnable:
        item_list = CATEGORY_ITEMS[cat]
        display   = CATEGORY_DISPLAY.get(cat, cat)

        print(f"\n{'='*60}")
        print(f"  【{display}】{len(item_list)} 個型號")
        print(f"{'='*60}")

        all_rows, summary_stats = scrape_category(cat, item_list)
        results[cat] = (all_rows, summary_stats)

        # 類別小結
        real = [r for r in all_rows if r.get("資料層級") != "查無資料"]
        has_data = sum(1 for s in summary_stats if s.get("總評論數", 0) > 0)
        print(f"\n  ✅ {display} 完成：{has_data}/{len(item_list)} 個型號有資料，"
              f"共 {len(real)} 則評論")

    # ── 輸出 Excel ───────────────────────────────────
    print(f"\n{'='*60}")
    print("  輸出 Excel...")
    filename = export_excel(results, runnable)
    print(f"  ✅ 完成！檔案：{filename}")
    print("=" * 60)

    # ── 全域總結 ─────────────────────────────────────
    print("\n📊 執行總結：")
    for cat in runnable:
        all_rows, summary_stats = results[cat]
        real     = [r for r in all_rows if r.get("資料層級") != "查無資料"]
        has_data = [s for s in summary_stats if s.get("總評論數", 0) > 0]
        display  = CATEGORY_DISPLAY.get(cat, cat)
        print(f"\n  【{display}】")
        for s in sorted(has_data, key=lambda x: -x.get("總評論數", 0))[:5]:
            print(f"    [{s['總評論數']:4d} 則] {s['型號']}")
        no_data = [s for s in summary_stats if s.get("總評論數", 0) == 0]
        if no_data:
            print(f"    查無資料：{len(no_data)} 個型號")


if __name__ == "__main__":
    main()