"""
timeline.py
查詢某型號的逐月口碑走勢，給前端畫時間軸圖表。

讀的是 build_timeline.py 產生的 data/rag_timeline.json（只有計數、沒有評論內容），
不是原始評論——那些在 src/filter/ 與 src/database/ 底下，都被 .dockerignore
排除在 API image 之外。
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
TIMELINE_FILE = ROOT / "data" / "rag_timeline.json"

# 只取最近這麼多個月。兩個理由：聊天視窗只有 384px 寬，A520M 那種橫跨 67 個月的
# 型號畫出來每個月不到 4px；而且久遠的口碑對「現在該不該買」也沒有參考價值。
# 窗口結尾對齊該型號「最後一次有人討論的月份」而不是今天，停產型號才看得到它
# 當年的走勢，而不是一整排空白。
WINDOW_MONTHS = 18

# 資料太少的型號畫出來會誤導：三則評論的月份算出「負評率 67%」看起來很嚴重，
# 實際上只是兩則抱怨。門檻是對「窗口內」而非全部歷史計算的——A520M 總數 53 則
# 看似及格，但攤在 67 個月上，最近 18 個月根本沒幾則。
MIN_TOTAL_COMMENTS = 50
MIN_ACTIVE_MONTHS = 6


def _count(month: dict) -> int:
    return month["positive"] + month["negative"] + month["neutral"]


class TimelineStore:
    def __init__(self, path: Path = TIMELINE_FILE):
        self._by_model: dict[str, dict] = {}
        if not path.exists():
            print(f"[RAG] 找不到 {path.name}，口碑時間軸停用（跑 build_timeline.py 產生）")
            return
        self._by_model = json.loads(path.read_text(encoding="utf-8"))
        print(f"[RAG] 載入 {len(self._by_model)} 個型號的口碑時間軸")

    def get(self, model: str) -> dict | None:
        """回傳該型號最近一段時間的逐月走勢；查無此型號或窗口內資料不足時回 None。"""
        entry = self._by_model.get(model)
        if not entry:
            return None

        months = entry["months"]
        while months and _count(months[-1]) == 0:
            months = months[:-1]
        window = months[-WINDOW_MONTHS:]

        total = sum(_count(m) for m in window)
        active = sum(1 for m in window if _count(m) > 0)
        if total < MIN_TOTAL_COMMENTS or active < MIN_ACTIVE_MONTHS:
            return None

        return {"model": model, "total": total, "months": window}


_store: TimelineStore | None = None


def get_store() -> TimelineStore:
    global _store
    if _store is None:
        _store = TimelineStore()
    return _store
