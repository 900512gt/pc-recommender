"""
排除 PTT 相關評論中「早於發售日」的推文，輸出新檔，不動原檔。
處理範圍：GPU、CPU 全部型號，加上 5 個有發售日的主機板晶片組；其他類別不處理。

發售前的推文是在討論傳聞、規格洩漏、預購價，不是實際使用後的評價。
判斷方式：推文日期 < 該型號的發售日 → 排除。日期缺漏或年份不明的列無法判斷，保留。

發售日取「官方第一次開賣的日期」（中國限定的型號取中國開賣日），台灣實際到貨可能晚幾天。

用法（專案根目錄）：
    python src/filter/exclude_prerelease.py
    python src/filter/exclude_prerelease.py --release-csv PTT_發售前評論清單.csv
        ↑ 改用組員清單裡的「型號 → 推定發售日」，只處理清單裡有的型號
輸出：
    src/database/input/ptt_comment/relevant_fixed_postrelease.jsonl  ← 排除後的相關評論
    src/filter/output/prerelease_excluded.csv                        ← 被排除的每一筆，供複查
"""
import argparse
import csv
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
IN_FILE    = ROOT / "src/database/input/ptt_comment/relevant_fixed.jsonl"
OUT_FILE   = ROOT / "src/database/input/ptt_comment/relevant_fixed_postrelease.jsonl"
REVIEW_CSV = ROOT / "src/filter/output/prerelease_excluded.csv"

STRICT_CATS = {"GPU", "CPU"}   # 這些類別的每個型號都必須有發售日

# 型號 → 發售日。2026-10-04 整理；標 * 的當天有上網查證，其餘為公開的上市日期
RELEASE = {
    # ── GPU ──
    "RTX5080":      "2025-01-30",
    "RTX5070Ti":    "2025-02-20",
    "RTX5070":      "2025-03-05",
    "RTX5060Ti":    "2025-04-16",
    "RTX5060":      "2025-05-19",   # *
    "RTX5050":      "2025-07-01",   # *
    "RX9070XT":     "2025-03-06",
    "RX9070":       "2025-03-06",
    "RX9060XT":     "2025-06-05",
    "RX9070GRE":    "2025-05-08",   # * 中國
    "RX7650GRE":    "2025-02-07",   # * 中國
    "Arc B580":     "2024-12-13",
    "RTX3050":      "2022-01-27",
    "GT1030":       "2017-05-17",
    "GT710":        "2016-01-26",
    "GT730":        "2014-06-18",
    # ── CPU ──
    "AMD R5 3400G":   "2019-07-07",
    "AMD R5 5500GT":  "2024-01-31",   # *
    "AMD R5 5600GT":  "2024-01-31",   # *
    "AMD R5 5500X3D": "2025-06-05",   # * 拉丁美洲
    "AMD R5 5600XT":  "2024-10-31",   # *
    "AMD R5 7500F":   "2023-07-22",   # * 中國；全球 7/23–7/24
    "AMD R5 8400F":   "2024-04-01",   # * 中國；全球零售約 5 月中
    "AMD R5 8500G":   "2024-01-31",   # *
    "AMD R5 8600G":   "2024-01-31",   # *
    "AMD R7 8700G":   "2024-01-31",   # *
    "AMD R5 9500F":   "2025-09-16",   # * 中國
    "AMD R5 9600X":   "2024-08-08",
    "AMD R7 9700X":   "2024-08-08",
    "AMD R9 9900X":   "2024-08-15",
    "AMD R9 9950X":   "2024-08-15",
    "AMD R9 9900X3D": "2025-03-12",
    "AMD R9 9950X3D": "2025-03-12",
    "AMD R7 7700":    "2023-01-10",
    "AMD R7 7800X3D": "2023-04-06",
    "Intel Core Ultra 5 225":   "2025-01-13",   # *
    "Intel Core Ultra 5 225F":  "2025-01-13",   # *
    "Intel Core Ultra 5 235":   "2025-01-13",   # *
    "Intel Core Ultra 5 245K":  "2024-10-24",
    "Intel Core Ultra 5 245KF": "2024-10-24",
    "Intel Core Ultra 7 265K":  "2024-10-24",
    "Intel Core Ultra 7 265KF": "2024-10-24",
    "Intel Core Ultra 9 285K":  "2024-10-24",
    "Intel i3-14100":  "2024-01-08",   # *
    "Intel i5-14400":  "2024-01-08",   # *
    "Intel i5-14400F": "2024-01-08",   # *
    "Intel i7-14700":  "2024-01-08",   # *
    "Intel i5-12400":  "2022-01-04",
    # ── 主機板晶片組（發售日取自組員的 PTT_發售前評論清單.csv，未另外查證）──
    "Z890":  "2024-10-24",
    "X870E": "2024-09-30",
    "X870":  "2024-09-30",
    "B850":  "2025-01-15",
    "B850M": "2025-01-15",
}
RELEASE = {m: date.fromisoformat(d) for m, d in RELEASE.items()}


def use_release_csv(path):
    """改用外部清單的發售日（欄位：型號、推定發售日），取代內建的 RELEASE"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        table = {r["型號"]: date.fromisoformat(r["推定發售日"]) for r in csv.DictReader(f)}
    RELEASE.clear()
    RELEASE.update(table)


def parse_date(s):
    """PTT 日期是 MM/DD/YYYY；缺漏或年份不明（MM/DD/????）回傳 None"""
    m = re.fullmatch(r"(\d\d)/(\d\d)/(\d{4})", s or "")
    return date(int(m[3]), int(m[1]), int(m[2])) if m else None


def days_before_release(row):
    """早於發售日幾天；型號沒有發售日、日期不明、或已發售 → None"""
    release = RELEASE.get(row["model"])
    d = parse_date(row.get("date"))
    if release is None or d is None:
        return None
    gap = (release - d).days
    return gap if gap > 0 else None


def check_models(rows):
    missing = sorted({x["model"] for x in rows
                      if x["category"] in STRICT_CATS and x["model"] not in RELEASE})
    if missing:
        raise SystemExit(f"[錯誤] 這些型號沒有發售日，請補進 RELEASE：{missing}")


def main():
    parser = argparse.ArgumentParser(description="排除 PTT 相關評論中早於發售日的推文")
    parser.add_argument("--release-csv", help="改用這份清單的發售日（欄位：型號、推定發售日）")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(IN_FILE, encoding="utf-8") if l.strip()]
    if args.release_csv:
        use_release_csv(args.release_csv)
        print(f"發售日來源：{args.release_csv}（{len(RELEASE)} 個型號）")
    else:
        check_models(rows)

    kept, excluded = [], []
    for x in rows:
        gap = days_before_release(x)
        (excluded if gap else kept).append((x, gap))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for x, _ in kept:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(REVIEW_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["類別", "型號", "發售日", "推文日期", "早幾天", "_index", "label", "文章標題", "內容", "url"])
        for x, gap in sorted(excluded, key=lambda t: (t[0]["category"], t[0]["model"], -t[1])):
            w.writerow([x["category"], x["model"], RELEASE[x["model"]].isoformat(), x["date"], gap,
                        x["_index"], x["label"], x["title"], x["content"], x["url"]])

    undated = sum(1 for x in rows if x["model"] in RELEASE and parse_date(x.get("date")) is None)
    total = Counter(x["category"] for x in rows)
    out   = Counter(x["category"] for x, _ in excluded)
    by_model = Counter(x["model"] for x, _ in excluded)
    all_model = Counter(x["model"] for x in rows)

    print(f"輸入 {len(rows)} 筆 → 保留 {len(kept)} 筆，排除 {len(excluded)} 筆"
          f"（其中早 60 天以上 {sum(1 for _, g in excluded if g > 60)} 筆）")
    for cat in sorted(out):
        print(f"  {cat}：{total[cat]} → {total[cat] - out[cat]}（排除 {out[cat]}）")
    print(f"  日期不明、無法判斷而保留：{undated} 筆")
    print("排除最多的型號：")
    for m, n in by_model.most_common(12):
        print(f"  {m:<28} 排除 {n:>4} / {all_model[m]:<5} 發售日 {RELEASE[m]}")
    gone = [m for m in by_model if by_model[m] == all_model[m]]
    if gone:
        print(f"整個型號都被排除（請確認是真的都在發售前，還是型號誤配）：{gone}")
    print(f"輸出：{OUT_FILE.relative_to(ROOT)}")
    print(f"複查清單：{REVIEW_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
