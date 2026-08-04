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
import time
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
QUERY_EMBED_RETRIES = 2  # 查詢時是即時回應使用者，重試次數/等待時間比批次建索引短

# cosine distance 門檻（越小越相似）。拿約 15 條涵蓋「明確查詢」「模糊但主題內」
# 「完全離題」「主題邊緣（滑鼠/鍵盤等資料庫沒有的 3C 週邊）」四類的測試查詢，實測
# v2（單一 chunk，比 v1 整型號合併文字短很多）的距離分布後校準：
#   - 明確查詢的 top1 distance 落在 0.31~0.43
#   - 模糊但主題內的落在 0.53~0.61
#   - 完全離題的落在 0.65~0.70
#   - 滑鼠/鍵盤這類主題邊緣的落在 0.60~0.61，剛好跟「模糊但主題內」重疊，
#     沒有一個閾值能同時完美分開兩者；寧可保守一點讓少數模糊查詢落回背景知識
#     回答（chat.py 現在會誠實標注「非來自論壇評價」），也不要冒著把滑鼠/鍵盤
#     查詢誤配到機殼/顯卡評論、講得煞有介事的風險，所以維持在偏低的 0.6。
MAX_DISTANCE = 0.6

# v1 另外有 MIN_GAP（擋「第一名沒有明顯贏過第二名」），v2 這裡刻意不沿用：
# v2 chunk 的顆粒度是「單一面向」，查「性價比高的主機板」這種問題，本來就會有
# 好幾個不同型號的「性價比」chunk 分數非常接近（每個型號都在講性價比，這是正常
# 現象，不是資料沒有鑑別力）。實測「有沒有性價比高的主機板」這條查詢，top1/top2
# 距離只差 0.0022，但 top1（mb__a520m__aspect__性價比）本身是完全合理的答案，
# MIN_GAP=0.005 會誤傷這種案例。而且這裡本來就回傳 top_k（預設 2）個候選給 LLM，
# 不是強迫選一個「唯一正確答案」，「評價相近」的情況已經交給 chat.py 的
# SYSTEM_PROMPT 規則處理（如實告知使用者這幾款評價相近），不需要在 retriever 這層
# 又用距離差距擋一次。

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
        self._check_index_freshness()

    def _check_index_freshness(self) -> None:
        """檢查 Chroma 索引的 chunk_id 集合是不是跟目前的 rag_chunks_v2.jsonl 完全一致。
        如果之後重新蒸餾但忘記重跑 embed_chunks_v2.py，索引會悄悄過期、检索到錯的/舊的
        chunk 而不會有任何警告，所以這裡在啟動時就大聲印出來，方便從 log 發現問題。"""
        indexed_ids = set(self._collection.get(include=[])["ids"])
        local_ids = set(self.chunk_index.keys())
        missing_from_index = local_ids - indexed_ids
        stale_in_index = indexed_ids - local_ids
        if missing_from_index or stale_in_index:
            print(
                f"[WARNING] RetrieverChromaV2 索引與 {CHUNKS_FILE.name} 不同步："
                f"jsonl 有但索引沒有 {len(missing_from_index)} 筆、"
                f"索引有但 jsonl 已刪除 {len(stale_in_index)} 筆。"
                f"請重跑 python src/rag/embed_chunks_v2.py 重建索引。"
            )

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

    def _embed_query(self, query: str) -> list[float]:
        """查詢時即時呼叫 embedding API，暫時性錯誤（rate limit/網路）重試幾次，
        避免使用者的單一次訊息因為 OpenAI 短暫抖動就直接失敗。"""
        for attempt in range(QUERY_EMBED_RETRIES):
            try:
                return self._client.embeddings.create(
                    model=EMBED_MODEL, input=[query]
                ).data[0].embedding
            except Exception:
                if attempt == QUERY_EMBED_RETRIES - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """OpenAI embedding + Chroma 向量搜尋，最小單位是單一 chunk。"""
        q_lower = query.lower()
        target_cats: set[str] = set()
        for kw, cats in CATEGORY_KEYWORDS.items():
            if kw in q_lower:
                target_cats |= cats

        where = {"category": {"$in": sorted(target_cats)}} if target_cats else None

        query_vec = self._embed_query(query)

        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
        )

        ids = result["ids"][0]
        distances = result["distances"][0]  # cosine distance，越小越相似

        if not ids:
            return []

        # 完全離題（例如問天氣）：連最相似的都差很遠，直接不回答，
        # 讓 LLM 改用背景知識回答並註明「非論壇評價」（見 chat.py SYSTEM_PROMPT）。
        if distances[0] > MAX_DISTANCE:
            return []

        matched_ids = ids[:top_k]
        return [self.chunk_index[i] for i in matched_ids if i in self.chunk_index]
