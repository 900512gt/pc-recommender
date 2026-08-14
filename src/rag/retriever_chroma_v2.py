"""
retriever_chroma_v2.py
跟 retriever_v2.py 對外介面相同的另一套 v2 檢索實作，語意搜尋用 OpenAI
embedding + Chroma 向量資料庫取代 TF-IDF（跟 retriever.py / retriever_chroma.py
的關係一樣：v1 的 tfidf/chroma 是同一份 chunk、不同排序方式的兩種嘗試，
v2 的 tfidf/chroma 也是同一份 rag_chunks_v2.jsonl、不同排序方式）。

跟 retriever_v2.py 完全獨立、互不影響。

檢索策略：
  1. 精確比對型號 → 回傳該型號「全部」chunk（跟 retriever_v2.py 相同邏輯，不受 top_k 限制）。
     這裡不篩「還在賣」，指名問特定型號是合理的口碑查詢，就算已停產也一樣回答。
  2. 找不到型號 → 用 OpenAI embedding 查詢 Chroma 的 parts_v2 collection 做語意搜尋，
     只保留「目前還買得到」的型號（比對 data/ga_database_v2.json，做法照抄
     retriever_fulltext.py 的 _load_sellable_models()，白名單直接放進查詢的
     where 條件，不是查回來後才用 Python 過濾）——避免模糊/推薦類查詢（例如
     「中階顯卡推薦」）撈到已停產的舊卡：舊卡討論多、社群共識穩定，語意上反而常常
     比新卡更像「推薦」用詞，沒有這層過濾就可能把停產商品講得像現行選項。
     依型號去重、湊滿 top_k 個不同型號後，各自回傳完整 chunk 組（見
     _semantic_search 的說明），不是回傳 top_k 則零散的單一 chunk。

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
GA_DB_FILE  = ROOT / "data" / "ga_database_v2.json"
CHROMA_DIR  = ROOT / "data" / "chroma_db"
COLLECTION  = "parts_v2"
EMBED_MODEL = "text-embedding-3-small"
QUERY_EMBED_RETRIES = 2  # 查詢時是即時回應使用者，重試次數/等待時間比批次建索引短
SEARCH_POOL_SIZE = 20  # 語意搜尋的起始候選池大小，湊不滿 top_k 個不同型號時會倍增
MAX_SEARCH_POOL_SIZE = 200  # 候選池倍增的上限，避免湊不滿時無限擴大、無限呼叫 API

# 語意搜尋 fallback（開放式推薦問題，沒指名型號）預設要湊到幾個不同型號。
# 依 category 統計 data/ga_database_v2.json 目前「還在賣」的型號數量（2026-08）：
#   CPU 33、機殼 29、SSD 19、GPU 16、主機板 15 ← 現行型號數量多，2 個候選明顯
#   只能呈現一小部分選擇範圍；電源 9、風冷 9、記憶體 7、水冷 5 ← 型號本來就少，
#   2 個候選已經接近資料庫上限，不太需要調高。
# 沒有依 category 動態調整（例如電源用 2、CPU 用 5），是刻意的取捨：這類問題
# 的 category 本來就是用關鍵字比對猜的（見 CATEGORY_KEYWORDS），不一定準確，
# 而且每個型號的完整 chunk 組本身就有好幾則（summary + 多則 aspect + pros_cons +
# comparison），調高 top_k 會讓要塞進 prompt 的 chunk 數量、token 用量、延遲
# 等比例增加，所以只從 2 小幅調到 3（而不是取上限 5），在「大類別候選太少」
# 跟「查詢成本」之間取一個保守的折衷；「小類別本來就沒必要調高」不會因為
# 調成統一值而變糟，3 個候選對電源、水冷這類本來就少的類別也還在合理範圍內。
# _match_models（指名型號）不受這個值影響，永遠回傳完整資料。
DEFAULT_SEMANTIC_TOP_K = 3

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


class RetrieverChromaV2:
    def __init__(
        self, chunks_file: Path = CHUNKS_FILE, chroma_dir: Path = CHROMA_DIR,
        ga_db_file: Path = GA_DB_FILE,
    ):
        self.chunks: list[dict] = []
        with open(chunks_file, encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        self._sellable_models = _load_sellable_models(ga_db_file)

        # chunk_id → chunk，Chroma 查詢只會回傳 id，實際內容從這裡查回來
        self.chunk_index: dict[str, dict] = {c["chunk_id"]: c for c in self.chunks}

        # Chroma metadata 的 "model" 欄位存的是原始大小寫（見 embed_chunks_v2.py），
        # 但 self._sellable_models 存的是小寫，語意搜尋要把「還在賣」白名單放進
        # where 條件時，需要用原始大小寫的型號字串才能匹配 Chroma metadata。
        self._sellable_models_original: set[str] = {
            c["model"] for c in self.chunks if c["model"].lower() in self._sellable_models
        }

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

    def retrieve(self, query: str, top_k: int = DEFAULT_SEMANTIC_TOP_K) -> list[dict]:
        """
        top_k 的語意依路徑而不同：
          - 精確比對型號（_match_models）：不受 top_k 限制，回傳比對到的每個
            型號「全部」chunk。
          - 語意搜尋 fallback（_semantic_search）：top_k 代表要湊到「幾個不同
            型號」，不是幾則 chunk。湊到的每個型號都會回傳該型號完整的 chunk
            組（summary/aspect/pros_cons/comparison），所以實際回傳的 chunk
            筆數通常會大於 top_k，且不是固定值。預設值見 DEFAULT_SEMANTIC_TOP_K
            的說明（依現行型號數量的 category 分布統計校準過，不是隨便選的）。
        """
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
        """OpenAI embedding + Chroma 向量搜尋。

        top_k 是「要湊到幾個不同型號」的目標值，不是「取原始結果前 N 筆」。
        做法：把 Chroma 依 distance 排序回傳的候選從頭掃描，每遇到一個還沒
        收錄過的型號就收錄，同型號的其他 chunk 直接跳過（不提前停止），直到
        湊滿 top_k 個不同型號為止。如果目前候選池掃完仍不夠，就擴大池子
        （n_results 倍增，上限 MAX_SEARCH_POOL_SIZE）重新查詢再掃一次；如果
        候選池已經沒有 distance 落在 MAX_DISTANCE 內的項目，或池子已經到
        上限，就如實回傳目前湊到的數量（可能小於 top_k），不放寬距離門檻湊數。
        湊到的每個型號都用 get_by_model() 取完整 chunk 組，攤平後回傳。
        """
        q_lower = query.lower()
        target_cats: set[str] = set()
        for kw, cats in CATEGORY_KEYWORDS.items():
            if kw in q_lower:
                target_cats |= cats

        # 「還在賣」白名單直接放進 where 條件（不是查詢後再用 Python 過濾），
        # 避免候選名額被注定會被丟棄的停產型號佔掉。跟類別條件用 $and 合併。
        conditions: list[dict] = [
            {"model": {"$in": sorted(self._sellable_models_original)}}
        ]
        if target_cats:
            conditions.append({"category": {"$in": sorted(target_cats)}})
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        query_vec = self._embed_query(query)

        pool_size = SEARCH_POOL_SIZE
        checked_top1_distance = False
        matched_models: list[str] = []

        while True:
            result = self._collection.query(
                query_embeddings=[query_vec],
                n_results=pool_size,
                where=where,
            )
            ids = result["ids"][0]
            distances = result["distances"][0]  # cosine distance，越小越相似

            if not ids:
                return []

            # 完全離題（例如問天氣）：連最相似的都差很遠，直接不回答，
            # 讓 LLM 改用背景知識回答並註明「非論壇評價」（見 chat.py SYSTEM_PROMPT）。
            # 只在第一輪、看整批裡最相似的一筆判斷一次。
            if not checked_top1_distance:
                if distances[0] > MAX_DISTANCE:
                    return []
                checked_top1_distance = True

            # 依型號去重＋湊滿數量是同一個掃描過程：從頭逐筆看，型號第一次
            # 出現才收錄，同型號的後續 chunk 跳過但不停止掃描，直到湊滿
            # top_k 個不同型號，或遇到超出 MAX_DISTANCE 的項目才停。
            matched_models = []
            seen_models: set[str] = set()
            hit_distance_limit = False
            for chunk_id, dist in zip(ids, distances):
                if dist > MAX_DISTANCE:
                    hit_distance_limit = True
                    break
                chunk = self.chunk_index.get(chunk_id)
                if chunk is None:
                    continue
                model = chunk["model"]
                if model not in seen_models:
                    seen_models.add(model)
                    matched_models.append(model)
                    if len(matched_models) >= top_k:
                        break

            if len(matched_models) >= top_k:
                break
            if hit_distance_limit:
                # 池子裡剩下的都超出距離門檻，擴大池子只會拿到更遠的結果，
                # 不可能湊到更多合格型號，如實回傳目前湊到的數量。
                break
            if len(ids) < pool_size or pool_size >= MAX_SEARCH_POOL_SIZE:
                # Chroma 回傳筆數少於要求，代表 where 條件下已經沒有更多候選；
                # 或池子已經到上限，兩種情況都不再擴大，如實回傳目前湊到的數量。
                break
            pool_size *= 2

        matched: list[dict] = []
        for model in matched_models:
            matched.extend(self.get_by_model(model))
        return matched
