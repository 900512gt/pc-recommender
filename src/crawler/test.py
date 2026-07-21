"""
PTT CPU 評論爬蟲 v3
- 抓取所有推文（推 / 噓 / →），不限負評
- 每則評論 = 一行
- 日期格式：MM/DD/YYYY（MM/DD 來自推文，YYYY 從文章 header 補）
- URL 去重：同一篇文章同一型號只進去一次
- Fallback：型號搜不到 → 改用系列關鍵字

使用方式:
  python ptt_cpu_scraper_v3.py

依賴:
  pip install requests beautifulsoup4 openpyxl
"""

import requests
import time
from bs4 import BeautifulSoup
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime
from collections import Counter

# ══════════════════════════════════════════════════════
#  設定區
# ══════════════════════════════════════════════════════

BOARDS    = ["PC_Shopping", "Hardware"]
MAX_PAGES = 10
SLEEP     = 0.8

SKIP_PREFIXES = [
    "[情報]", "[開箱]", "[售]", "[WTS]", "[賣]",
    "[徵]", "[WTB]", "[交]",
]

# ══════════════════════════════════════════════════════
#  型號清單
# ══════════════════════════════════════════════════════

CPU_LIST = [
    {"model": "Intel i3-14100",           "brand": "Intel", "series": "i3",           "search_kw": "14100",     "fallback": "14代 i3"},
    {"model": "Intel i5-12400",           "brand": "Intel", "series": "i5",           "search_kw": "12400",     "fallback": "12代 i5"},
    {"model": "Intel i5-14400F",          "brand": "Intel", "series": "i5",           "search_kw": "14400F",    "fallback": "14代 i5"},
    {"model": "Intel i5-14400",           "brand": "Intel", "series": "i5",           "search_kw": "14400",     "fallback": "14代 i5"},
    {"model": "Intel i7-14700F",          "brand": "Intel", "series": "i7",           "search_kw": "14700F",    "fallback": "14代 i7"},
    {"model": "Intel i7-14700",           "brand": "Intel", "series": "i7",           "search_kw": "14700",     "fallback": "14代 i7"},
    {"model": "Intel Core Ultra 5 225F",  "brand": "Intel", "series": "Core Ultra 5", "search_kw": "225F",      "fallback": "Core Ultra 5"},
    {"model": "Intel Core Ultra 5 225",   "brand": "Intel", "series": "Core Ultra 5", "search_kw": "Ultra 225", "fallback": "Core Ultra 5"},
    {"model": "Intel Core Ultra 5 235",   "brand": "Intel", "series": "Core Ultra 5", "search_kw": "Ultra 235", "fallback": "Core Ultra 5"},
    {"model": "Intel Core Ultra 5 245KF", "brand": "Intel", "series": "Core Ultra 5", "search_kw": "245KF",     "fallback": "Arrow Lake"},
    {"model": "Intel Core Ultra 5 245K",  "brand": "Intel", "series": "Core Ultra 5", "search_kw": "245K",      "fallback": "Arrow Lake"},
    {"model": "Intel Core Ultra 7 265KF", "brand": "Intel", "series": "Core Ultra 7", "search_kw": "265KF",     "fallback": "Arrow Lake"},
    {"model": "Intel Core Ultra 7 265K",  "brand": "Intel", "series": "Core Ultra 7", "search_kw": "265K",      "fallback": "Arrow Lake"},
    {"model": "Intel Core Ultra 9 285K",  "brand": "Intel", "series": "Core Ultra 9", "search_kw": "285K",      "fallback": "Arrow Lake"},
    {"model": "AMD R5 3400G",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "3400G",     "fallback": "Ryzen 5 3000"},
    {"model": "AMD R5 5500GT",            "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "5500GT",    "fallback": "Ryzen 5 5000"},
    {"model": "AMD R5 5600XT",            "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "5600XT",    "fallback": "Ryzen 5 5000"},
    {"model": "AMD R5 5600GT",            "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "5600GT",    "fallback": "Ryzen 5 5000"},
    {"model": "AMD R5 5500X3D",           "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "5500X3D",   "fallback": "X3D AM4"},
    {"model": "AMD R5 7500F",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "7500F",     "fallback": "Ryzen 5 7000"},
    {"model": "AMD R5 8400F",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "8400F",     "fallback": "Ryzen 5 8000"},
    {"model": "AMD R5 8500G",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "8500G",     "fallback": "Ryzen 5 8000"},
    {"model": "AMD R5 8600G",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "8600G",     "fallback": "Ryzen 5 8000"},
    {"model": "AMD R5 9500F",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "9500F",     "fallback": "Ryzen 9000"},
    {"model": "AMD R5 9600X",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "9600X",     "fallback": "Ryzen 9000"},
    {"model": "AMD R7 7700",              "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "R7 7700",   "fallback": "Ryzen 7 7000"},
    {"model": "AMD R7 7800X3D",           "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "7800X3D",   "fallback": "X3D AM5"},
    {"model": "AMD R7 8700G",             "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "8700G",     "fallback": "Ryzen 7 8000"},
    {"model": "AMD R7 9700X",             "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "9700X",     "fallback": "Ryzen 9000"},
    {"model": "AMD R7 9400X3D",           "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "9400X3D",   "fallback": "X3D AM5"},
    {"model": "AMD R7 9450X3D",           "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "9450X3D",   "fallback": "X3D AM5"},
    {"model": "AMD R9 9900X",             "brand": "AMD",   "series": "Ryzen 9",      "search_kw": "9900X",     "fallback": "Ryzen 9000"},
    {"model": "AMD R9 9900X3D",           "brand": "AMD",   "series": "Ryzen 9",      "search_kw": "9900X3D",   "fallback": "X3D AM5"},
    {"model": "AMD R9 9950X",             "brand": "AMD",   "series": "Ryzen 9",      "search_kw": "9950X",     "fallback": "Ryzen 9000"},
    {"model": "AMD R9 9950X3D",           "brand": "AMD",   "series": "Ryzen 9",      "search_kw": "9950X3D",   "fallback": "X3D AM5"},
]

# ══════════════════════════════════════════════════════
#  HTTP
# ══════════════════════════════════════════════════════

SESSION = requests.Session()
SESSION.cookies.set("over18", "1", domain="www.ptt.cc")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch(url, retries=3):
    for i in range(retries):
        try:
            r = SESSION.get(url, headers=HEADERS, timeout=10)
            r.encoding = "utf-8"
            return r.text
        except Exception as e:
            print(f"    [重試 {i+1}/{retries}] {e}")
            time.sleep(2)
    return None


# ══════════════════════════════════════════════════════
#  日期工具
#  - 文章 header 取年份（PTT 格式：Mon Jan  1 12:00:00 2024）
#  - 推文日期格式：MM/DD HH:MM → 補年份 → MM/DD/YYYY
# ══════════════════════════════════════════════════════

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def get_article_year(soup) -> int | None:
    """從文章 header 取發文年份"""
    for meta in soup.select("div.article-metaline"):
        tag = meta.select_one("span.article-meta-tag")
        val = meta.select_one("span.article-meta-value")
        if tag and val and "時間" in tag.text:
            parts = val.text.strip().split()
            try:
                return int(parts[4])   # Wed Sep 24 HH:MM:SS YYYY
            except Exception:
                pass
    return None


def format_push_date(raw_date: str, article_year: int | None) -> str:
    """
    PTT push-ipdatetime 格式：'IP MM/DD HH:MM'，例如 '36.231.86.26 09/24 12:57'
    取倒數第二個欄位（MM/DD），補上年份 → MM/DD/YYYY
    """
    if not raw_date:
        return ""
    parts = raw_date.strip().split()
    # parts[-2] = MM/DD，parts[-1] = HH:MM
    date_part = parts[-2] if len(parts) >= 2 else parts[0]
    year = str(article_year) if article_year else "????"
    return f"{date_part}/{year}"


# ══════════════════════════════════════════════════════
#  搜尋看板：取得文章清單
# ══════════════════════════════════════════════════════

def should_skip(title):
    for prefix in SKIP_PREFIXES:
        if title.startswith(prefix):
            return True
    return False


def search_board(board, keyword):
    articles = []
    url = f"https://www.ptt.cc/bbs/{board}/search?q={requests.utils.quote(keyword)}"

    for page in range(1, MAX_PAGES + 1):
        html = fetch(url)
        if not html:
            break

        soup  = BeautifulSoup(html, "html.parser")
        items = soup.select("div.r-ent")
        kept  = 0

        for item in items:
            title_tag = item.select_one("div.title a")
            if not title_tag:
                continue
            title = title_tag.text.strip()
            if should_skip(title):
                continue
            articles.append({
                "title": title,
                "link":  "https://www.ptt.cc" + title_tag["href"],
                "board": board,
            })
            kept += 1

        print(f"      [{board}] p{page}: {len(items)} 篇 → 保留 {kept} 篇")

        prev = soup.select_one("a.btn.wide:-soup-contains('上頁')")
        if not prev or not prev.get("href"):
            break
        url = "https://www.ptt.cc" + prev["href"]
        time.sleep(SLEEP)

    return articles


# ══════════════════════════════════════════════════════
#  解析文章：抓所有推文，每則一筆 dict
# ══════════════════════════════════════════════════════

def parse_article(art, model, brand, series, data_level):
    """
    回傳 list of comment rows（每則推文一筆）
    若抓取失敗回傳 []
    """
    html = fetch(art["link"])
    if not html:
        return []

    soup         = BeautifulSoup(html, "html.parser")
    article_year = get_article_year(soup)
    rows         = []

    for push in soup.select("div.push"):
        tag_span  = push.find("span", class_="push-tag")
        cnt_span  = push.find("span", class_="push-content")
        # push-date 的 class 是 "f2 push-date"，用 find + class_ 最穩
        date_span = push.find("span", class_="push-ipdatetime")
        if not tag_span or not cnt_span:
            continue

        tag      = tag_span.text.strip()          # 推 / 噓 / →
        content  = cnt_span.text.strip().lstrip(": ").strip()
        raw_date = date_span.text.strip() if date_span else ""
        pub_date = format_push_date(raw_date, article_year)

        rows.append({
            "對應型號": model,
            "品牌":     brand,
            "系列":     series,
            "資料層級": data_level,
            "文章標題": art["title"],
            "文章連結": art["link"],
            "推噓類型": tag,
            "評論內容": content,
            "發佈日期": pub_date,
        })

    return rows


# ══════════════════════════════════════════════════════
#  爬單一型號（URL 去重 + fallback）
# ══════════════════════════════════════════════════════

def scrape_model(cpu_info):
    model       = cpu_info["model"]
    search_kw   = cpu_info["search_kw"]
    fallback_kw = cpu_info["fallback"]
    brand       = cpu_info["brand"]
    series      = cpu_info["series"]

    def _run(kw, data_level, seen_urls):
        all_rows = []
        articles = []
        for board in BOARDS:
            articles += search_board(board, kw)

        # URL 去重
        unique = []
        for art in articles:
            if art["link"] not in seen_urls:
                seen_urls.add(art["link"])
                unique.append(art)
            else:
                print(f"        [略過重複] {art['title'][:40]}")

        for i, art in enumerate(unique):
            print(f"      [{i+1}/{len(unique)}] {art['title'][:50]}")
            rows = parse_article(art, model, brand, series, data_level)
            all_rows.extend(rows)
            time.sleep(SLEEP)

        return all_rows

    seen_urls = set()

    # 第一輪：型號關鍵字
    print(f"  🔍 型號搜尋：「{search_kw}」")
    rows = _run(search_kw, "型號", seen_urls)
    if rows:
        return rows, "型號"

    # Fallback：系列關鍵字
    print(f"  ⚠️  改用系列搜尋：「{fallback_kw}」")
    rows = _run(fallback_kw, "系列(fallback)", seen_urls)
    if rows:
        return rows, "系列(fallback)"

    # 查無資料
    return [{
        "對應型號": model, "品牌": brand, "系列": series,
        "資料層級": "查無資料", "文章標題": "查無相關文章",
        "文章連結": "", "推噓類型": "-",
        "評論內容": f"以「{search_kw}」及「{fallback_kw}」均無搜尋結果",
        "發佈日期": "-",
    }], "查無資料"


# ══════════════════════════════════════════════════════
#  輸出 Excel
# ══════════════════════════════════════════════════════

# 清除 Excel 不支援的控制字元（避免 xlsx 格式損壞）
import re as _re
_ILLEGAL_CHARS = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

def sanitize(val):
    if val is None:
        return ""
    if isinstance(val, float) and val != val:   # NaN check
        return ""
    if isinstance(val, str):
        return _ILLEGAL_CHARS.sub("", val)
    return val


def export_excel(all_rows, summary_stats):
    thin   = Border(left=Side(style="thin"), right=Side(style="thin"),
                    top=Side(style="thin"),  bottom=Side(style="thin"))
    h_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    h_fill = PatternFill("solid", fgColor="1F3864")
    c_aln  = Alignment(horizontal="center", vertical="center", wrap_text=True)
    l_aln  = Alignment(horizontal="left",   vertical="center", wrap_text=True)

    # 推噓顏色
    push_fill   = PatternFill("solid", fgColor="E2EFDA")  # 推  → 綠
    boo_fill    = PatternFill("solid", fgColor="FCE4D6")  # 噓  → 紅
    arrow_fill  = PatternFill("solid", fgColor="FFFFFF")  # →  → 白
    fb_fill     = PatternFill("solid", fgColor="EBF3FB")  # fallback → 藍
    nodata_fill = PatternFill("solid", fgColor="F2F2F2")  # 查無 → 灰

    wb = Workbook()

    # ── Sheet 1: 評論明細（每則推文一行）────────────────
    ws = wb.active
    ws.title = "評論明細"

    cols   = ["對應型號", "品牌", "系列", "資料層級",
              "文章標題", "文章連結", "推噓類型", "評論內容", "發佈日期"]
    widths = [22, 8, 16, 16,
              45, 52, 10, 72, 14]
    center_cols = {1, 2, 3, 4, 7, 9}

    for ci, (h, w) in enumerate(zip(cols, widths), 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.font = h_font; cell.fill = h_fill
        cell.alignment = c_aln; cell.border = thin
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.row_dimensions[1].height = 24
    ws.freeze_panes = "A2"

    norm_font = Font(name="Arial", size=9)
    gray_font = Font(name="Arial", size=9, color="999999", italic=True)

    for ri, row in enumerate(all_rows, 2):
        level  = row.get("資料層級", "")
        tag    = row.get("推噓類型", "")
        nodata = (level == "查無資料")

        if nodata:
            row_fill = nodata_fill
        elif "fallback" in level:
            row_fill = fb_fill
        elif tag == "推":
            row_fill = push_fill
        elif tag == "噓":
            row_fill = boo_fill
        else:
            row_fill = arrow_fill

        for ci, key in enumerate(cols, 1):
            val  = sanitize(row.get(key, ""))
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.fill         = row_fill
            cell.border       = thin
            cell.alignment    = c_aln if ci in center_cols else l_aln
            cell.font         = gray_font if nodata else norm_font
            if key == "發佈日期":
                cell.number_format = "@"   # 強制文字，避免 Excel 自動轉日期格式

        ws.row_dimensions[ri].height = 38

    # ── Sheet 2: 各型號統計 ───────────────────────────
    ws2 = wb.create_sheet("各型號統計")
    s_cols  = ["型號", "品牌", "系列", "資料層級",
               "文章數", "推數", "噓數", "→數", "總評論數", "有無資料"]
    s_widths = [22, 8, 16, 16, 8, 8, 8, 8, 10, 10]

    for ci, (h, w) in enumerate(zip(s_cols, s_widths), 1):
        cell = ws2.cell(row=1, column=ci, value=h)
        cell.font = h_font; cell.fill = h_fill
        cell.alignment = c_aln; cell.border = thin
        ws2.column_dimensions[get_column_letter(ci)].width = w
    ws2.row_dimensions[1].height = 24
    ws2.freeze_panes = "A2"

    red_fill = PatternFill("solid", fgColor="FCE4D6")
    grn_fill = PatternFill("solid", fgColor="E2EFDA")
    ylw_fill = PatternFill("solid", fgColor="FFF2CC")

    for ri, stat in enumerate(summary_stats, 2):
        total    = stat.get("總評論數", 0)
        boo_cnt  = stat.get("噓數", 0)
        row_fill = (nodata_fill if total == 0 else
                    red_fill   if boo_cnt >= 10 else
                    ylw_fill   if "fallback" in stat.get("資料層級", "") else
                    grn_fill)
        for ci, key in enumerate(s_cols, 1):
            cell = ws2.cell(row=ri, column=ci, value=stat.get(key, ""))
            cell.fill = row_fill; cell.border = thin
            cell.alignment = c_aln
            cell.font = Font(name="Arial", size=10)
        ws2.row_dimensions[ri].height = 22

    # ── Sheet 3: 執行摘要 ─────────────────────────────
    ws3 = wb.create_sheet("執行摘要")
    real = [r for r in all_rows if r.get("資料層級") != "查無資料"]
    meta = [
        ("執行時間",       datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("搜尋看板",       ", ".join(BOARDS)),
        ("最大翻頁數",     MAX_PAGES),
        ("搜尋型號總數",   len(CPU_LIST)),
        ("有資料型號數",   sum(1 for s in summary_stats if s.get("總評論數", 0) > 0)),
        ("查無資料型號數", sum(1 for s in summary_stats if s.get("總評論數", 0) == 0)),
        ("總評論數",       len(real)),
        ("推數",           sum(1 for r in real if r.get("推噓類型") == "推")),
        ("噓數",           sum(1 for r in real if r.get("推噓類型") == "噓")),
        ("→ 數",           sum(1 for r in real if r.get("推噓類型") == "→")),
        ("評論最多型號",   max(summary_stats, key=lambda x: x.get("總評論數", 0))["型號"] if summary_stats else "-"),
    ]
    for ri, (k, v) in enumerate(meta, 1):
        ws3.cell(row=ri, column=1, value=k).font = Font(name="Arial", bold=True, size=11)
        ws3.cell(row=ri, column=2, value=v).font = Font(name="Arial", size=11)
    ws3.column_dimensions["A"].width = 22
    ws3.column_dimensions["B"].width = 40

    filename = f"ptt_CPU評論_全型號_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    wb.save(filename)
    return filename


# ══════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PTT CPU 評論爬蟲 v3 - 全推文，每則一行")
    print(f"  {len(CPU_LIST)} 個型號 | 看板：{', '.join(BOARDS)}")
    print("=" * 60)

    all_rows      = []
    summary_stats = []

    for idx, cpu in enumerate(CPU_LIST, 1):
        print(f"\n[{idx:02d}/{len(CPU_LIST)}] ── {cpu['model']} ──────────────")
        rows, level = scrape_model(cpu)
        all_rows.extend(rows)

        real = [r for r in rows if r.get("資料層級") != "查無資料"]
        unique_arts = len(set(r["文章連結"] for r in real if r.get("文章連結")))

        summary_stats.append({
            "型號":     cpu["model"],
            "品牌":     cpu["brand"],
            "系列":     cpu["series"],
            "資料層級": level,
            "文章數":   unique_arts,
            "推數":     sum(1 for r in real if r.get("推噓類型") == "推"),
            "噓數":     sum(1 for r in real if r.get("推噓類型") == "噓"),
            "→數":      sum(1 for r in real if r.get("推噓類型") == "→"),
            "總評論數": len(real),
            "有無資料": "✅ 有" if real else "❌ 無",
        })
        print(f"  → {level}，{unique_arts} 篇，{len(real)} 則評論")

    print("\n" + "=" * 60)
    print(f"  共 {len(all_rows)} 筆，輸出 Excel...")
    filename = export_excel(all_rows, summary_stats)
    print(f"  ✅ 完成！檔案：{filename}")
    print("=" * 60)

    has  = [s for s in summary_stats if s["總評論數"] > 0]
    none = [s for s in summary_stats if s["總評論數"] == 0]
    print(f"\n📊 有資料型號（{len(has)} 個）：")
    for s in sorted(has, key=lambda x: -x["總評論數"]):
        print(f"   [{s['總評論數']:4d} 則] {s['型號']}  ({s['資料層級']})")
    print(f"\n📭 查無資料型號（{len(none)} 個）：")
    for s in none:
        print(f"   {s['型號']}")


if __name__ == "__main__":
    main()