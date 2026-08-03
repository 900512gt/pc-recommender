"""
retriever_chroma_v2.py
跟 retriever_v2.py 對外介面相同的另一套 v2 檢索實作，語意搜尋用 OpenAI
embedding + Chroma 向量資料庫取代 TF-IDF（跟 retriever.py / retriever_chroma.py
的關係一樣：v1 的 tfidf/chroma 是同一份 chunk、不同排序方式的兩種嘗試，
v2 的 tfidf/chroma 也是同一份 rag_chunks_v2.jsonl、不同排序方式）。

跟 retriever_v2.py 完全獨立、互不影響。

檢索策略：
  1. 精確比對型號 → 回傳該型號「全部」chunk（跟 retriever_v2.py 相同邏輯，不受 top_k 限制）
  2. 找不到型號 → 用 OpenAI embedding 查詢 Chroma 的 parts_v2 collection 做語意搜尋，
     最小單位是單一 chunk（跟 retriever_v2.py 的 TF-IDF fallback 一樣，但换成向量相似度）

前置作業：
  python src/rag/embed_chunks_v2.py   # 建立 data/chroma_db/ 裡的 parts_v2 collection
"""

import json
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from src.rag.chunk_text import CATEGORY_KEYWORDS

ROOT        = Path(__file__).parent.parent.parent
CHUNKS_FILE = ROOT / "data" / "rag_chunks_v2.jsonl"
CHROMA_DIR  = ROOT / "data" / "chroma_db"
COLLECTION  = "parts_v2"
EMBED_MODEL = "text-embedding-3-small"

# cosine distance 門檻（越小越相似），沿用 retriever_chroma.py（v1）校準過的起始值：
#   MAX_DISTANCE 擋完全離題的查詢；MIN_GAP 擋「第一名沒有明顯贏過第二名」的模糊查詢
# v2 是單一 chunk（比 v1 整型號的合併文字短很多），距離分布可能不太一樣，
# 這兩個數字之後有更多 v2 的實際使用資料再重新校準。
MAX_DISTANCE = 0.6
MIN_GAP      = 0.005

load_dotenv(ROOT / ".env")


class RetrieverChromaV2:
    def __init__(self, chunks_file: Path = CHUNKS_FILE, chroma_dir: Path = CHROMA_DIR):
        self.chunks: list[dict] = []
        with open(chunks_file, encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        # chunk_id → chunk，Chroma 查詢只會回傳 id，實際內容從這裡查回來
        self.chunk_index: dict[str, dict] = {c["chunk_id"]: c for c in self.chunks}

        # 依型號分組（小寫 key），供精確比對用
        self.chunks_by_model: dict[str, list[dict]] = {}
        for c in self.chunks:
            self.chunks_by_model.setdefault(c["model"].lower(), []).append(c)

        self.aliases: dict[str, str] = {}
        for model in self.chunks_by_model:
            short = re.sub(r"^(rtx|rx|amd r\d |intel |amd )", "", model).strip()
            if short and short != model:
                self.aliases[short] = model

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("請在 .env 設定 OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)

        if not chroma_dir.exists():
            raise RuntimeError(
                f"找不到 {chroma_dir}，請先執行 python src/rag/embed_chunks_v2.py 建立向量索引"
            )
        db = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = db.get_collection(COLLECTION)

    # ── 公開 API（跟 retriever_v2.RetrieverV2 相同介面）──────────────

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        matched = self._match_models(query)
        if matched:
            return matched
        return self._semantic_search(query, top_k)

    def get_by_model(self, model: str) -> list[dict]:
        key = model.lower()
        return self.chunks_by_model.get(key) or self.chunks_by_model.get(self.aliases.get(key, ""), [])

    def _match_models(self, query: str) -> list[dict]:
        q = query.lower()
        found: list[dict] = []
        seen_models: set[str] = set()

        candidates = sorted(self.chunks_by_model.keys(), key=len, reverse=True)
        for model_key in candidates:
            if model_key in q and model_key not in seen_models:
                found.extend(self.chunks_by_model[model_key])
                seen_models.add(model_key)

        for alias, full in sorted(self.aliases.items(), key=lambda x: len(x[0]), reverse=True):
            if alias in q and full not in seen_models:
                found.extend(self.chunks_by_model.get(full, []))
                seen_models.add(full)

        return found

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """OpenAI embedding + Chroma 向量搜尋，最小單位是單一 chunk。"""
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

        # 第一名沒有明顯贏過第二名：這批資料對這個問題沒有清楚的鑑別力，
        # 與其硬選看似自信、實則接近雜訊的答案，不如老實回傳空結果，
        # 讓 LLM 改用背景知識回答並註明「非論壇評價」（見 chat.py SYSTEM_PROMPT）。
        if len(distances) >= 2 and (distances[1] - distances[0]) < MIN_GAP:
            return []

        matched_ids = ids[:top_k]
        return [self.chunk_index[i] for i in matched_ids if i in self.chunk_index]
