"""
retriever_fulltext.py
第三種檢索實作：不做相似度排序，找到類別後把該類別「全部」chunk 丟給 LLM，
讓 LLM 自己讀完所有型號的完整評價再做推薦/比較判斷。

動機：chunk 數量少（161 筆，全部約 36000 tokens，單一類別最多約 9000 tokens），
完全塞得進 GPT-4o 的 context window。相似度排序（TF-IDF / embedding）在「這個查詢
主題是什麼」上表現不錯，但在「這個查詢的最佳答案是什麼」這種需要判斷適合度、
理解正負面語意的任務上，容易被字面上主題相關但實際結論相反的內容誤導
（例如「這張卡遊戲效能不足」跟「我要打電動用的顯卡」字面上都圍繞「遊戲」，
相似度反而會偏高）。把判斷交給 LLM 讀完整段落，而不是交給向量距離排序，
可以避開這個問題。

跟 retriever.py / retriever_chroma.py 完全獨立，方便三方比較（見 compare_retrievers.py）。
不需要 embedding API 呼叫，也不需要 Chroma。

檢索策略：
  1. 從使用者查詢中抽取型號名稱 → 直接回傳對應 chunk（精確比對，跟其他兩版相同邏輯，
     不受「目前是否還在賣」限制——查骨董卡的評價還是查得到）
  2. 找不到型號 → 從查詢辨識類別關鍵字 → 回傳該類別「全部」chunk，但只保留
     data/ga_database_v2.json（目前 CoolPC 實際在賣的商品庫）裡還買得到的型號。
     這不是額外塞給 LLM 的主觀偏好，只是把「推薦範圍」限制在使用者真的能買到的東西，
     跟電商系統過濾缺貨商品是同一種邏輯。PTT/巴哈評論裡留著很多已經停產型號的舊討論
     （例如 RTX30 系列大多已下市，只剩 RTX3050 還在賣），沒有這層過濾，LLM 讀到舊評論
     就可能推薦出已經買不到的型號。
  3. 完全辨識不出類別 → 回傳空 list，交給 LLM 用背景知識回答（見 chat.py SYSTEM_PROMPT）
"""

import json
import re
from pathlib import Path

from src.rag.chunk_text import chunk_to_text, chunk_to_context, CATEGORY_KEYWORDS  # noqa: F401

ROOT           = Path(__file__).parent.parent.parent
CHUNKS_FILE    = ROOT / "data" / "rag_chunks.jsonl"
GA_DB_FILE     = ROOT / "data" / "ga_database_v2.json"


def _load_sellable_models(ga_db_file: Path) -> set[str]:
    """讀取目前實際在賣的商品庫，回傳所有型號名稱（小寫）的集合。"""
    with open(ga_db_file, encoding="utf-8") as f:
        db = json.load(f)
    models: set[str] = set()
    for items in db.values():
        for item in items:
            model = item.get("ptt_model")
            if model:
                models.add(model.lower())
    return models


class RetrieverFulltext:
    def __init__(self, chunks_file: Path = CHUNKS_FILE, ga_db_file: Path = GA_DB_FILE):
        self.chunks: list[dict] = []
        with open(chunks_file, encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        self.model_index: dict[str, dict] = {
            c["model"].lower(): c for c in self.chunks
        }

        self.aliases: dict[str, str] = {}
        for model in self.model_index:
            short = re.sub(r"^(rtx|rx|amd r\d |intel |amd )", "", model).strip()
            if short and short != model:
                self.aliases[short] = model

        self._sellable_models = _load_sellable_models(ga_db_file)

    # ── 公開 API（跟 retriever.Retriever 相同介面）──────────────

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        """
        top_k 只影響型號精確比對的回傳筆數（跟其他兩版一致）。
        類別分類這條路徑刻意忽略 top_k，回傳整個類別，讓 LLM 自己判斷取捨。
        """
        matched = self._match_models(query)
        if matched:
            return matched[:top_k]
        return self._category_dump(query)

    def get_by_model(self, model: str) -> dict | None:
        key = model.lower()
        return self.model_index.get(key) or self.model_index.get(self.aliases.get(key, ""))

    def _match_models(self, query: str) -> list[dict]:
        q = query.lower()
        found: list[dict] = []
        seen: set[str] = set()

        candidates = sorted(self.model_index.keys(), key=len, reverse=True)
        for model_key in candidates:
            if model_key in q and model_key not in seen:
                found.append(self.model_index[model_key])
                seen.add(model_key)

        for alias, full in sorted(self.aliases.items(), key=lambda x: len(x[0]), reverse=True):
            if alias in q and full not in seen:
                chunk = self.model_index.get(full)
                if chunk:
                    found.append(chunk)
                    seen.add(full)

        return found

    def _category_dump(self, query: str) -> list[dict]:
        """
        辨識查詢裡的類別關鍵字，回傳該類別「目前還買得到」的全部 chunk
        （不辨識到類別就回傳空；目前買不到的型號會被排除，避免推薦到停產骨董）。
        """
        q_lower = query.lower()
        target_cats: set[str] = set()
        for kw, cats in CATEGORY_KEYWORDS.items():
            if kw in q_lower:
                target_cats |= cats

        if not target_cats:
            return []

        return [
            c for c in self.chunks
            if c["category"] in target_cats and c["model"].lower() in self._sellable_models
        ]
