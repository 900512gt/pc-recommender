"""
category_keyword_eval.py
量化 CATEGORY_KEYWORDS（chunk_text.py）這個關鍵字分類器的準確度。

retriever_chroma_v2.py 的 _semantic_search 會先用 CATEGORY_KEYWORDS 猜測
使用者問題屬於哪個零件類別，猜到才會把 Chroma 查詢限制在該類別（見該檔案
_semantic_search 開頭的 target_cats 邏輯）。這個猜測完全靠人工挑關鍵字、
單純字串比對（`kw in query.lower()`），從沒被量化測過：
  - recall：使用者用口語同義詞問，系統漏篩、放行全類別搜尋的機率有多高
  - precision：關鍵字剛好是別的字串的子字串，誤觸發不相關類別的機率有多高

用一組人工標註 ground truth 的測試問題，跑實際的分類邏輯（跟
retriever_chroma_v2.py._semantic_search 裡的寫法完全一致），統計
recall / precision 並列出所有失敗案例，輸出成 Markdown 報告方便引用。

使用方式（從專案根目錄執行，不需要 API key，純本地邏輯測試）：
  python src/rag/category_keyword_eval.py
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.chunk_text import CATEGORY_KEYWORDS

GPU   = {"GPU", "顯示卡"}
CPU   = {"CPU", "處理器"}
SSD   = {"SSD", "固態硬碟"}
MB    = {"MB", "主機板"}
RAM   = {"記憶體", "RAM"}
PSU   = {"電源", "PSU"}
CASE  = {"機殼", "CASE"}
COOL  = {"水冷", "WATER_COOLER", "風冷", "AIR_COOLER"}
NONE_: set[str] = set()

# (query, expected_categories, 說明)
# expected = None  → 刻意設計成廣泛型問題，不篩類別才是正確行為，只記錄不計分
# expected = set() → 完全離題／資料庫沒有的品項，應該是空集合
# expected = {...} → 明確在問某個類別，應該篩出這個集合
TEST_CASES: list[tuple[str, set[str] | None, str]] = [
    # ── A. 標準關鍵字直接命中（baseline，預期應該全過）──────────────
    ("顯卡推薦",                       GPU,   "A-標準關鍵字"),
    ("CPU 選哪顆比較好",                CPU,   "A-標準關鍵字"),
    ("SSD 要買多大容量",                SSD,   "A-標準關鍵字"),
    ("主機板品牌推薦",                  MB,    "A-標準關鍵字"),
    ("記憶體要買幾條",                  RAM,   "A-標準關鍵字"),
    ("電源瓦數怎麼算",                  PSU,   "A-標準關鍵字"),
    ("機殼推薦",                       CASE,  "A-標準關鍵字"),
    ("水冷好還是風冷好",                COOL,  "A-標準關鍵字（複合）"),
    ("散熱器有推薦的嗎",                COOL,  "A-標準關鍵字（散熱→雙類別）"),

    # ── B. 口語同義詞，不在關鍵字表內（測 recall gap）────────────────
    ("獨顯選哪張比較好",                GPU,   "B-口語同義詞（獨顯）"),
    ("顯示晶片選哪顆",                  GPU,   "B-口語同義詞（顯示晶片）"),
    ("硬碟推薦",                       {"HDD", "硬碟"}, "B-口語同義詞（硬碟，資料庫有 HDD 但關鍵字表沒收錄）"),
    ("PSU 推薦",                      PSU,   "B-英文縮寫同義詞（PSU 本身不是關鍵字，只有『電源』是）"),
    ("case 用哪款好",                  CASE,  "B-英文同義詞（case 本身不是關鍵字，只有『機殼』是）"),
    ("供電穩不穩定",                    PSU,   "B-口語同義詞（供電）"),
    ("M.2 固態硬碟該怎麼挑",             SSD,   "B-正control（含『固態』關鍵字，應該過）"),

    # ── C. 子字串誤觸發（測 precision / false positive）─────────────
    ("這張顯卡有8MB快取，正常嗎",         GPU,   "C-假陽性風險（『mb』藏在『8MB』裡）"),
    ("SSD 讀寫速度 700MB/s 算快嗎",      SSD,   "C-假陽性風險（『mb』藏在『700MB/s』裡）"),
    ("有 8GB 顯存的顯卡推薦嗎",           GPU,   "C-正control（不該誤觸發，8GB 沒有 mb）"),

    # ── D. 完全離題／資料庫沒有的品項（應該是空集合）─────────────────
    ("今天天氣如何",                    NONE_, "D-離題"),
    ("推薦一款好用的滑鼠",               NONE_, "D-資料庫沒有的品項"),
    ("晚餐吃什麼比較好",                 NONE_, "D-離題"),
    ("幫我寫一首詩",                    NONE_, "D-離題"),

    # ── E. 刻意設計成廣泛、不該被單一類別限制住的問題 ─────────────────
    ("我想組一台文書機，該怎麼配",         None,  "E-廣泛型問題（正確行為是不篩類別）"),
    ("五萬預算怎麼配一台電腦",            None,  "E-廣泛型問題（正確行為是不篩類別）"),
    ("有沒有推薦的零件",                 None,  "E-廣泛型問題（正確行為是不篩類別）"),
]


def predict(query: str) -> set[str]:
    """完全複製 retriever_chroma_v2.py._semantic_search 的分類邏輯。"""
    q_lower = query.lower()
    target_cats: set[str] = set()
    for kw, cats in CATEGORY_KEYWORDS.items():
        if kw in q_lower:
            target_cats |= cats
    return target_cats


def main():
    total = len(TEST_CASES)
    correct = 0
    recall_fail = []
    precision_fail = []
    skipped_broad = 0
    rows = []

    for query, expected, note in TEST_CASES:
        actual = predict(query)

        if expected is None:
            skipped_broad += 1
            rows.append((query, "(不特別限定)", sorted(actual), "廣泛型，不計分", note))
            continue

        missing = expected - actual
        extra = actual - expected

        if not missing and not extra:
            correct += 1
            mark = "✅"
        else:
            mark = "❌"
            if missing:
                recall_fail.append((query, expected, actual, note))
            if extra:
                precision_fail.append((query, expected, actual, note))
        rows.append((query, sorted(expected), sorted(actual), mark, note))

    scored = total - skipped_broad
    accuracy = correct / scored if scored else 0.0

    print(f"總測試案例：{total}（{skipped_broad} 題廣泛型不計分）")
    print(f"計分案例：{scored}，正確：{correct}，準確率：{accuracy:.1%}")
    print(f"Recall 失敗：{len(recall_fail)} 題，Precision 失敗：{len(precision_fail)} 題")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "data" / "rag_eval_results" / f"category_keyword_eval_{timestamp}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# CATEGORY_KEYWORDS 分類準確度量化報告",
        "",
        f"- 時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 測試對象：`src/rag/chunk_text.py` 的 `CATEGORY_KEYWORDS`"
        "（`retriever_chroma_v2.py._semantic_search` 用它決定要不要對 Chroma 查詢加 category 過濾）",
        f"- 總測試案例：{total}（其中 {skipped_broad} 題為刻意設計的廣泛型問題，只記錄不計分）",
        f"- **計分案例：{scored}，正確：{correct}，準確率：{accuracy:.1%}**",
        f"- Recall 失敗（該篩的類別漏篩）：{len(recall_fail)} 題",
        f"- Precision 失敗（篩到不該有的類別）：{len(precision_fail)} 題",
        "",
        "## 完整測試結果",
        "",
        "| 查詢 | 預期類別 | 實際類別 | 結果 | 說明 |",
        "|---|---|---|---|---|",
    ]
    for query, expected, actual, mark, note in rows:
        lines.append(f"| {query} | {expected} | {actual} | {mark} | {note} |")

    lines += ["", "## Recall 失敗案例（口語同義詞漏篩，讓查詢退回全類別搜尋）", ""]
    if recall_fail:
        for q, exp, act, note in recall_fail:
            lines.append(f"- **「{q}」**（{note}）")
            lines.append(f"  - 預期至少要有：`{sorted(exp)}`")
            lines.append(f"  - 實際只有：`{sorted(act)}`")
    else:
        lines.append("（無）")

    lines += ["", "## Precision 失敗案例（子字串誤觸發不相關類別）", ""]
    if precision_fail:
        for q, exp, act, note in precision_fail:
            lines.append(f"- **「{q}」**（{note}）")
            lines.append(f"  - 預期：`{sorted(exp)}`")
            lines.append(f"  - 實際多出：`{sorted(act - exp)}`")
    else:
        lines.append("（無）")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n報告已寫入：{out_path}")


if __name__ == "__main__":
    main()
