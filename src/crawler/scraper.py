"""
scraper.py
PTT 爬蟲共用核心

負責：
  - HTTP 請求（含重試）
  - PTT 站內搜尋（逐頁翻頁）
  - 文章推文解析
  - 單一型號爬取流程（含 fallback 與 URL 去重）
"""

import requests
import time
from bs4 import BeautifulSoup

# ══════════════════════════════════════════════════════
#  全域設定
# ══════════════════════════════════════════════════════

BOARDS    = ["PC_Shopping", "Hardware"]
MAX_PAGES = 10
SLEEP     = 0.8

SKIP_PREFIXES = [
    "[開箱]", "[售]", "[WTS]", "[賣]",
    "[徵]", "[WTB]", "[交]",
]

INFO_PREFIX = "[情報]"  # 情報文單獨標記，不跳過但會反映在資料層級

SESSION = requests.Session()
SESSION.cookies.set("over18", "1", domain="www.ptt.cc")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ══════════════════════════════════════════════════════
#  HTTP
# ══════════════════════════════════════════════════════

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
# ══════════════════════════════════════════════════════

def get_article_year(soup):
    """從文章 header 取發文年份"""
    for meta in soup.select("div.article-metaline"):
        tag = meta.select_one("span.article-meta-tag")
        val = meta.select_one("span.article-meta-value")
        if tag and val and "時間" in tag.text:
            parts = val.text.strip().split()
            try:
                return int(parts[4])
            except Exception:
                pass
    return None


def format_push_date(raw_date, article_year):
    """
    PTT push-ipdatetime 格式：'IP MM/DD HH:MM'
    取倒數第二個欄位（MM/DD），補上年份 → MM/DD/YYYY
    """
    if not raw_date:
        return ""
    parts = raw_date.strip().split()
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
    """在單一看板搜尋關鍵字，回傳文章清單"""
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
#  解析文章推文
# ══════════════════════════════════════════════════════

def parse_article(art, model, brand, series, data_level, content_filter=None):
    """
    進入文章頁面，抓取所有推文
    回傳 list of row dicts；抓取失敗回傳 []

    content_filter（可選）：品牌關鍵字清單，例如 ["金士頓", "Kingston", "Beast"]
      - 若文章標題已包含任一關鍵字 → 保留該文章所有推文
      - 若標題不含關鍵字 → 只保留推文內容有提到關鍵字的推文
      - 若 content_filter 為 None → 不過濾，保留所有推文（預設行為）

    資料層級格式：
      型號                → 一般討論文，搜尋命中型號關鍵字
      型號(情報)          → 情報文，搜尋命中型號關鍵字
      系列(fallback)      → 一般討論文，fallback 搜尋
      系列(fallback+情報) → 情報文，fallback 搜尋
    """
    html = fetch(art["link"])
    if not html:
        return []

    # 判斷是否為情報文，並合併進 data_level
    is_info = art["title"].startswith(INFO_PREFIX)
    if is_info:
        if "fallback" in data_level:
            effective_level = "系列(fallback+情報)"
        else:
            effective_level = f"{data_level}(情報)"
    else:
        effective_level = data_level

    # content_filter：判斷文章標題是否已含品牌關鍵字
    # 若是，則該文章所有推文都保留（不需逐則過濾）
    title_has_brand = False
    if content_filter:
        title_has_brand = any(kw in art["title"] for kw in content_filter)

    soup         = BeautifulSoup(html, "html.parser")
    article_year = get_article_year(soup)
    rows         = []

    for push in soup.select("div.push"):
        tag_span  = push.find("span", class_="push-tag")
        cnt_span  = push.find("span", class_="push-content")
        date_span = push.find("span", class_="push-ipdatetime")
        if not tag_span or not cnt_span:
            continue

        tag      = tag_span.text.strip()
        content  = cnt_span.text.strip().lstrip(": ").strip()
        raw_date = date_span.text.strip() if date_span else ""
        pub_date = format_push_date(raw_date, article_year)

        # 若標題已含品牌關鍵字則全部保留，否則逐則過濾推文內容
        if content_filter and not title_has_brand:
            if not any(kw in content for kw in content_filter):
                continue

        rows.append({
            "對應型號": model,
            "品牌":     brand,
            "系列":     series,
            "資料層級": effective_level,
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

def scrape_item(item_info):
    """
    爬取單一型號的所有推文
    item_info 必須包含 model / brand / series / search_kw / fallback
    選填 content_filter：品牌關鍵字清單，用於記憶體等以規格搜尋的類別
    回傳 (rows, level)
    """
    model          = item_info["model"]
    search_kw      = item_info["search_kw"]
    fallback_kw    = item_info["fallback"]
    brand          = item_info["brand"]
    series         = item_info["series"]
    content_filter = item_info.get("content_filter", None)  # 可選

    def _run(kw, data_level, seen_urls):
        all_rows = []
        articles = []
        for board in BOARDS:
            articles += search_board(board, kw)

        unique = []
        for art in articles:
            if art["link"] not in seen_urls:
                seen_urls.add(art["link"])
                unique.append(art)
            else:
                print(f"        [略過重複] {art['title'][:40]}")

        for i, art in enumerate(unique):
            print(f"      [{i+1}/{len(unique)}] {art['title'][:50]}")
            rows = parse_article(
                art, model, brand, series, data_level,
                content_filter=content_filter,
            )
            all_rows.extend(rows)
            time.sleep(SLEEP)

        return all_rows

    seen_urls = set()

    print(f"  🔍 型號搜尋：「{search_kw}」")
    rows = _run(search_kw, "型號", seen_urls)
    if rows:
        return rows, "型號"

    print(f"  ⚠️  改用系列搜尋：「{fallback_kw}」")
    rows = _run(fallback_kw, "系列(fallback)", seen_urls)
    if rows:
        return rows, "系列(fallback)"

    return [{
        "對應型號": model, "品牌": brand, "系列": series,
        "資料層級": "查無資料", "文章標題": "查無相關文章",
        "文章連結": "", "推噓類型": "-",
        "評論內容": f"以「{search_kw}」及「{fallback_kw}」均無搜尋結果",
        "發佈日期": "-",
    }], "查無資料"

# ══════════════════════════════════════════════════════
#  爬整個類別
# ══════════════════════════════════════════════════════

def scrape_category(category_name, item_list):
    """
    爬取一整個類別，回傳 (all_rows, summary_stats)

    summary_stats 每筆：
      型號 / 品牌 / 系列 / 資料層級 /
      文章數 / 推數 / 噓數 / →數 / 總評論數 / 有無資料
    """
    all_rows      = []
    summary_stats = []
    total         = len(item_list)

    for idx, item in enumerate(item_list, 1):
        print(f"\n[{idx:02d}/{total}] ── {item['model']} ──────────────")
        rows, level = scrape_item(item)
        all_rows.extend(rows)

        real = [r for r in rows if r.get("資料層級") != "查無資料"]

        # 套用與 exporter 相同的清理規則，確保統計數字一致
        def _is_valid(r):
            content = str(r.get("評論內容", "") or "").strip()
            return bool(content) and len(content) >= 2

        real = [r for r in real if _is_valid(r)]
        unique_arts = len(set(r["文章連結"] for r in real if r.get("文章連結")))

        summary_stats.append({
            "型號":     item["model"],
            "品牌":     item["brand"],
            "系列":     item["series"],
            "資料層級": level,
            "文章數":   unique_arts,
            "推數":     sum(1 for r in real if r.get("推噓類型") == "推"),
            "噓數":     sum(1 for r in real if r.get("推噓類型") == "噓"),
            "→數":      sum(1 for r in real if r.get("推噓類型") == "→"),
            "總評論數": len(real),
            "有無資料": "✅ 有" if real else "❌ 無",
        })
        print(f"  → {level}，{unique_arts} 篇，{len(real)} 則評論")

    return all_rows, summary_stats