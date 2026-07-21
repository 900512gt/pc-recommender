"""
retriever_chroma.py
跟 retriever.py 對外介面相同的另一套檢索實作，語意搜尋用 OpenAI embedding + Chroma
向量資料庫取代 TF-IDF。跟 retriever.py 完全獨立、互不影響，方便日後 A/B 比較
（見 compare_retrievers.py）。

檢索策略：
  1. 從使用者查詢中抽取型號名稱 → 直接回傳對應 chunk（精確比對，跟 retriever.py 相同邏輯）
  2. 找不到型號 → 用 OpenAI embedding 查詢 Chroma 做語意搜尋

前置作業：
  python src/rag/embed_chunks.py   # 建立 data/chroma_db/
"""

import json
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from src.rag.chunk_text import chunk_to_text, chunk_to_context, CATEGORY_KEYWORDS  # noqa: F401

ROOT        = Path(__file__).parent.parent.parent
CHUNKS_FILE = ROOT / "data" / "rag_chunks.jsonl"
CHROMA_DIR  = ROOT / "data" / "chroma_db"
COLLECTION  = "parts"
EMBED_MODEL = "text-embedding-3-small"

# cosine distance 門檻（越小越相似）：
#   MAX_DISTANCE 擋完全離題的查詢（連最像的都差很遠）
#   MIN_GAP      擋「第一名沒有明顯贏過第二名」的模糊查詢（資料鑑別力不足，寧可不答）
#   兩個數字是拿 18 條測試查詢實測校準出來的折衷值，之後有更多實際使用資料可以再調整。
MAX_DISTANCE = 0.6
MIN_GAP      = 0.005

load_dotenv(ROOT / ".env")


class RetrieverChroma:
    def __init__(self, chunks_file: Path = CHUNKS_FILE, chroma_dir: Path = CHROMA_DIR):
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

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("請在 .env 設定 OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)

        if not chroma_dir.exists():
            raise RuntimeError(
                f"找不到 {chroma_dir}，請先執行 python src/rag/embed_chunks.py 建立向量索引"
            )
        db = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = db.get_collection(COLLECTION)

    # ── 公開 API（跟 retriever.Retriever 相同介面）──────────────

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        matched = self._match_models(query)
        if matched:
            return matched[:top_k]
        return self._semantic_search(query, top_k)

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

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """OpenAI embedding + Chroma 向量搜尋，回傳 chunk list。"""
        q_lower = query.lower()
        target_cats: set[str] = set()
        for kw, cats in CATEGORY_KEYWORDS.items():
            if kw in q_lower:
                target_cats |= cats

        where = {"category": {"$in": sorted(target_cats)}} if target_cats else None

        query_vec = self._client.embeddings.create(
            model=EMBED_MODEL, input=[query]
        ).data[0].embedding

        # 多拿一筆，用來判斷「第一名贏第二名多少」
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=max(top_k, 2),
            where=where,
        )

        ids = result["ids"][0]
        distances = result["distances"][0]  # cosine distance，越小越相似

        if not ids:
            return []

        # 完全離題（例如問天氣）：連最相似的都差很遠，直接不回答
        if distances[0] > MAX_DISTANCE:
            return []

        # 第一名沒有明顯贏過第二名：代表這批資料對這個問題沒有清楚的鑑別力
        # （例如「哪張顯卡比較耐用」這種問題，每個型號摘要都用差不多的字眼描述穩定性，
        # 分數幾乎打平）。與其硬選兩個看似自信、實則接近雜訊的答案，
        # 不如老實回傳空結果，讓 LLM 改用背景知識回答並註明「非論壇評價」（見 chat.py SYSTEM_PROMPT）。
        if len(distances) >= 2 and (distances[1] - distances[0]) < MIN_GAP:
            return []

        matched_ids = ids[:top_k]
        return [self.model_index[i.lower()] for i in matched_ids]
