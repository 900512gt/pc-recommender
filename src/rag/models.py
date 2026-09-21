"""
models.py
零件型號的索引與詳情，給 /parts 列表頁與 /parts/[型號] 詳情頁使用。

把散在各處的資料湊成一個型號的完整樣貌：
  rag_chunks_v2.jsonl   摘要、面向敘述、優缺點、比較（蒸餾出的文字）
  ga_database_v2.json   目前售價、跑分、規格（只有還在賣的型號才有）

逐月走勢與面向分數各自有獨立端點（timeline.py / aspects.py），不重複塞進這裡
——詳情頁分開抓，列表頁根本不需要那些細節。
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CHUNKS_FILE = ROOT / "data" / "rag_chunks_v2.jsonl"
GA_DB_FILE = ROOT / "data" / "ga_database_v2.json"

# chunk 的 category 欄位混用了中英文（CASE 與 機殼、MB 與 主機板 並存），
# 統一成中文再給前端，否則同一類零件會在篩選器裡出現兩次。
CATEGORY_ALIASES = {
    "CASE": "機殼",
    "MB": "主機板",
    "PSU": "電源",
    "AIR_COOLER": "風冷",
    "WATER_COOLER": "水冷",
    "RAM": "記憶體",
}


class ModelStore:
    def __init__(self, chunks_file: Path = CHUNKS_FILE, ga_db_file: Path = GA_DB_FILE):
        self._by_model: dict[str, dict] = {}
        self._listing: dict[str, dict] = {}

        if ga_db_file.exists():
            db = json.loads(ga_db_file.read_text(encoding="utf-8"))
            for items in db.values():
                for item in items:
                    model = item.get("ptt_model")
                    # 同型號會有多張卡/多家品牌，取最便宜的當代表售價
                    if model and (model not in self._listing
                                  or item.get("price", 0) < self._listing[model].get("price", 0)):
                        self._listing[model] = item

        if not chunks_file.exists():
            print(f"[RAG] 找不到 {chunks_file.name}，型號頁停用")
            return

        with chunks_file.open(encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                model = chunk["model"]
                entry = self._by_model.setdefault(model, {
                    "model": model,
                    "category": None,
                    "summary": None,
                    "confidence": None,
                    "review_count": 0,
                    "label_distribution": {},
                    "source_distribution": {},
                    "data_start_date": None,
                    "data_end_date": None,
                    "aspects": [],
                    "pros_cons": [],
                    "comparisons": [],
                })

                category = chunk.get("category")
                if category:
                    entry["category"] = CATEGORY_ALIASES.get(category, category)

                kind = chunk.get("chunk_type")
                if kind == "summary":
                    entry["summary"] = chunk.get("text")
                    entry["confidence"] = chunk.get("confidence")
                    entry["review_count"] = chunk.get("total_valid_comments") or 0
                    entry["label_distribution"] = chunk.get("label_distribution") or {}
                    entry["source_distribution"] = chunk.get("source_distribution") or {}
                    entry["data_start_date"] = chunk.get("data_start_date")
                    entry["data_end_date"] = chunk.get("data_end_date")
                elif kind == "aspect":
                    entry["aspects"].append({
                        "aspect": chunk.get("aspect"),
                        "text": chunk.get("text"),
                    })
                elif kind == "pros_cons":
                    entry["pros_cons"].append(chunk.get("text"))
                elif kind == "comparison":
                    entry["comparisons"].append(chunk.get("text"))

        print(f"[RAG] 載入 {len(self._by_model)} 個型號的詳情資料")

    def index(self) -> list[dict]:
        """型號列表，只帶列表頁要顯示與篩選用的欄位。"""
        rows = []
        for model, entry in self._by_model.items():
            listing = self._listing.get(model)
            rows.append({
                "model": model,
                "category": entry["category"],
                "review_count": entry["review_count"],
                "label_distribution": entry["label_distribution"],
                "confidence": entry["confidence"],
                "price": listing.get("price") if listing else None,
                "benchmark": listing.get("benchmark") if listing else None,
            })
        rows.sort(key=lambda r: -r["review_count"])
        return rows

    def detail(self, model: str) -> dict | None:
        entry = self._by_model.get(model)
        if not entry:
            return None

        listing = self._listing.get(model)
        return {
            **entry,
            # listing 為 None 只代表這個型號不在原價屋報價單快照裡，不能據此推論停產
            # ——這份快照連 RTX40 系列都沒有，但那顯然還買得到。口碑資料照樣顯示，
            # 前端會把價格顯示成「—」並加註說明。
            "listing": {
                "name": listing.get("name"),
                "price": listing.get("price"),
                "benchmark": listing.get("benchmark"),
                "brand": listing.get("brand"),
            } if listing else None,
        }


_store: ModelStore | None = None


def get_store() -> ModelStore:
    global _store
    if _store is None:
        _store = ModelStore()
    return _store
