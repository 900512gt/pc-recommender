"""
aspects.py
查詢某型號的五大面向口碑分數（效能／溫控／噪音／保固／CP值），給前端畫雷達圖比較。

資料來源是 GA 那邊既有的 part_sentiment.json——由 fine-tune 過的中研院 BERT 跑完
全部 50,593 則評論產生，161 個型號跟 RAG chunk 的型號 100% 對得上。所以這個功能
不需要重跑蒸餾、不花任何 API 費用，直接讀現成的分數。

刻意讀原始 JSON 而不是透過 GA 的 SentimentScorer：那支對樣本數不足的面向會補上
0.5，對加權計算是合理的預設值，但畫進雷達圖就變成無中生有的「中立評價」。這裡
把真實樣本數一起回傳，讓前端自己決定要不要畫。
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SENTIMENT_FILE = ROOT / "code" / "ga_pc_builder" / "data" / "part_sentiment.json"

DIMENSIONS = ("效能", "溫控", "噪音", "保固", "CP值")

# 與 GA 的 data/sentiment.py MIN_SAMPLES 一致。低於這個樣本數的面向分數波動太大
# ——三則評論算出「溫控 0.33」看起來很篤定，其實只是有兩個人抱怨。
MIN_SAMPLES = 10

# 雷達圖至少要三個軸才構成多邊形，兩個軸只會畫出一條線。
MIN_SUFFICIENT_DIMENSIONS = 3


class AspectStore:
    def __init__(self, path: Path = SENTIMENT_FILE):
        self._by_model: dict[str, dict] = {}
        if not path.exists():
            print(f"[RAG] 找不到 {path.name}，面向雷達圖停用")
            return

        raw = json.loads(path.read_text(encoding="utf-8"))
        for key, item in raw.items():
            if "|" not in key:
                continue
            # key 形如 "GPU|RTX4070"；型號名在 161 個條目裡是唯一的，所以直接用型號當索引
            _, model = key.split("|", 1)
            self._by_model[model] = item
        print(f"[RAG] 載入 {len(self._by_model)} 個型號的面向口碑分數")

    def get(self, model: str) -> dict | None:
        """回傳該型號的五大面向分數；查無此型號、或資料充足的面向不到三個時回 None。"""
        item = self._by_model.get(model)
        if not item:
            return None

        scores = item.get("dimensions", {})
        counts = item.get("counts", {})

        aspects = []
        sufficient = 0
        for name in DIMENSIONS:
            c = counts.get(name, {})
            # 原始資料的鍵是簡體的「负」
            positive, negative = int(c.get("正", 0)), int(c.get("负", 0))
            samples = positive + negative
            is_sufficient = samples >= MIN_SAMPLES
            if is_sufficient:
                sufficient += 1
            aspects.append({
                "name": name,
                "score": round(float(scores.get(name, 0.5)), 3),
                "positive": positive,
                "negative": negative,
                "samples": samples,
                "sufficient": is_sufficient,
            })

        if sufficient < MIN_SUFFICIENT_DIMENSIONS:
            return None

        return {
            "model": model,
            "review_count": int(item.get("review_count", 0)),
            "aspects": aspects,
        }


_store: AspectStore | None = None


def get_store() -> AspectStore:
    global _store
    if _store is None:
        _store = AspectStore()
    return _store
