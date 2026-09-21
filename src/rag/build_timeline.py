"""
build_timeline.py
把完整的論壇評論依「型號 × 月份 × 正負評」聚合，輸出 data/rag_timeline.json，
給口碑時間軸圖表使用。

為什麼不直接用 rag_evidence.jsonl：那份是蒸餾時抽樣過的（RTX5080 母體 3669 則
只取 300 則），而且是依「來源 × 標籤」分層抽樣，正負比例本身就被抽樣策略改動過，
拿來畫逐月比例會失真。時間軸必須讀完整的原始評論重新聚合。

輸出只有計數、沒有評論內容，所以檔案很小（幾百 KB），可以安全打包進 API image
——原始評論所在的 src/filter/ 與 src/database/ 都被 .dockerignore 排除。

使用方式（從專案根目錄執行）：
  python src/rag/build_timeline.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

PTT_FILE = ROOT / "src" / "filter" / "output" / "relevant.jsonl"
BAHA_FILES = [
    ROOT / "src" / "database" / "input" / "baha_comment" / f"matched_part{i}.jsonl"
    for i in (1, 2, 3)
]
OUT_FILE = ROOT / "data" / "rag_timeline.json"

LABEL_FIX = {"负評": "負評"}
LABEL_KEY = {"正評": "positive", "負評": "negative", "中立": "neutral"}


def _month(raw: str | None) -> str | None:
    """PTT 是 MM/DD/YYYY、巴哈是 YYYY-MM-DD HH:MM:SS，都取到 YYYY-MM。
    髒資料會回 None 不計入：PTT 有幾筆是 '????'，巴哈有幾筆是沒解析成絕對時間的
    '6 小時前 編輯'。"""
    if not raw:
        return None
    raw = raw.strip()
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return f"{parts[2]}-{parts[0].zfill(2)}"
        return None
    head = raw.split(" ")[0][:7]
    if len(head) == 7 and head[4] == "-" and head[:4].isdigit() and head[5:].isdigit():
        return head
    return None


def _month_range(start: str, end: str) -> list[str]:
    """把 '2025-01' 到 '2025-04' 展開成連續月份。中間沒有評論的月份要補 0，
    不然折線圖會把不連續的月份畫成等距，看起來像資料連續。"""
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def main() -> None:
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    totals: Counter = Counter()
    undated = 0

    def feed(path: Path) -> None:
        nonlocal undated
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                model = row.get("model")
                if not model:
                    continue
                totals[model] += 1
                month = _month(row.get("date"))
                if not month:
                    undated += 1
                    continue
                label = LABEL_FIX.get(row.get("label"), row.get("label"))
                key = LABEL_KEY.get(label)
                if key:
                    counts[model][month][key] += 1

    feed(PTT_FILE)
    for path in BAHA_FILES:
        feed(path)

    print(f"聚合 {sum(totals.values())} 則評論，{len(totals)} 個型號（日期無法解析 {undated} 則）")

    out = {}
    for model, by_month in counts.items():
        if not by_month:
            continue
        months = sorted(by_month)
        out[model] = {
            "total": totals[model],
            "months": [
                {"month": mo, **{k: by_month[mo].get(k, 0) for k in ("positive", "negative", "neutral")}}
                for mo in _month_range(months[0], months[-1])
            ],
        }

    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    size_kb = OUT_FILE.stat().st_size / 1024
    print(f"已寫入 {OUT_FILE.relative_to(ROOT)}：{len(out)} 個型號，{size_kb:.0f} KB")


if __name__ == "__main__":
    main()
