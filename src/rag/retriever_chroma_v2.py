"""
retriever_chroma_v2.py
跟 retriever_v2.py 對外介面相同的另一套 v2 檢索實作，語意搜尋用 OpenAI
embedding + Chroma 向量資料庫取代 TF-IDF（跟 retriever.py / retriever_chroma.py
的關係一樣：v1 的 tfidf/chroma 是同一份 chunk、不同排序方式的兩種嘗試，
v2 的 tfidf/chroma 也是同一份 rag_chunks_v2.jsonl、不同排序方式）。

跟 retriever_v2.py 完全獨立、互不影響。

檢索策略（三步驟）：
  1. 呼叫 _extract_model_intents() 用輕量 LLM 從使用者問題裡抽取「提到的
     型號＋極性」（include=在問/在意這個型號、exclude=明確表示不要/排除
     這個型號，例如「除了 X 以外」「不要 X」「X 我已經有了」）。這一步取代
     舊版純字串比對（`model_key in query`）——字串比對只看「提到了沒」，
     看不出使用者是想要還是排除，「請推薦除了 5070 以外的顯卡」用字串比對
     會被誤判成「使用者在問 RTX5070」而回傳它的完整資料，跟使用者真實意圖
     相反。抽取只能從資料庫已知型號清單（用 JSON schema 的 enum 強制）裡
     辨認，避免 LLM 生成資料庫沒有的型號字串。
     LLM 抽取失敗（重試後仍失敗，例如網路問題）時，fallback 回舊版的純
     字串比對（_match_models 保留下來當安全網，沒有被刪除），寧可退回舊
     行為、也不讓 retrieve() 直接掛掉。
  2. 有 include 型號 → 對每個型號呼叫 get_by_model() 取得「全部」chunk
     （不受 top_k 限制，這裡不篩「還在賣」，指名問特定型號是合理的口碑
     查詢，就算已停產也一樣回答）。沒有 include 型號（不管有沒有
     exclude）→ 走 _semantic_search 語意搜尋。
  3. exclude 型號清單往下傳給 _semantic_search：語意搜尋只保留「目前還
     買得到」的型號（比對 data/ga_database_v2.json，白名單直接放進查詢的
     where 條件），依型號去重、湊滿 top_k 個不同型號後各自回傳完整 chunk
     組；掃描候選時，遇到在 exclude 清單裡的型號直接跳過不收錄，確保使用者
     明確排除的型號不會透過語意相似度又混進結果。

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
INTENT_MODEL = "gpt-4o-mini"  # 只做型號＋極性抽取的前處理，不需要用到主要回答用的 gpt-5.5
INTENT_EXTRACT_RETRIES = 2
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

        # 小寫 key → 原始大小寫的代表型號字串（跟 chunk 裡存的 "model" 欄位一致），
        # 供 LLM 型號抽取的 enum 清單使用，讓 LLM 只能從資料庫已知型號裡辨認，
        # 不會生成清單外、查不到資料的字串。
        self._canonical_model_names: dict[str, str] = {
            key: chunks[0]["model"] for key, chunks in self.chunks_by_model.items()
        }

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
        三步驟流程（見檔頭「檢索策略」說明）：
          1. _extract_model_intents() 用 LLM 抽取問題裡提到的型號＋極性
             （include/exclude）。抽取失敗（回傳 None）時 fallback 回
             _match_models 的純字串比對安全網。
          2. 有 include 型號 → 對每個型號呼叫 get_by_model() 取得「全部」
             chunk，不受 top_k 限制。沒有 include 型號 → 走 _semantic_search。
          3. exclude 型號清單傳給 _semantic_search，掃描候選時直接跳過。

        top_k 只影響語意搜尋路徑：代表要湊到「幾個不同型號」，不是幾則
        chunk。湊到的每個型號都會回傳完整 chunk 組，所以實際回傳的 chunk
        筆數通常會大於 top_k，且不是固定值。預設值見 DEFAULT_SEMANTIC_TOP_K
        的說明（依現行型號數量的 category 分布統計校準過，不是隨便選的）。
        """
        intents = self._extract_model_intents(query)

        if intents is None:
            # LLM 抽取失敗（重試後仍失敗），退回舊版純字串比對安全網
            matched = self._match_models(query)
            if matched:
                return matched
            return self._semantic_search(query, top_k)

        include_models = [m["model"] for m in intents if m.get("polarity") == "include"]
        exclude_models = [m["model"] for m in intents if m.get("polarity") == "exclude"]

        if include_models:
            found: list[dict] = []
            seen: set[str] = set()
            for model in include_models:
                key = model.lower()
                if key in seen:
                    continue
                seen.add(key)
                found.extend(self.get_by_model(model))
            return found

        return self._semantic_search(query, top_k, exclude_models=exclude_models)

    def get_by_model(self, model: str) -> list[dict]:
        key = model.lower()
        return self.chunks_by_model.get(key) or self.chunks_by_model.get(self.aliases.get(key, ""), [])

    def _match_models(self, query: str) -> list[dict]:
        """純字串比對，只看型號字串有沒有出現在問題裡，看不出使用者是
        想要還是排除這個型號。保留作為 _extract_model_intents() 呼叫 LLM
        失敗時的安全網，正常情況下 retrieve() 不會走到這裡。"""
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

    def _extract_model_intents(self, query: str) -> list[dict] | None:
        """呼叫輕量 LLM，從使用者問題裡抽取「提到的型號＋極性」，取代純
        字串比對，才能處理「除了 X 以外」「不要 X」這類否定語意（字串比對
        只看提到了沒，看不出使用者是想要還是排除）。

        回傳格式：[{"model": "RTX5070", "polarity": "include" | "exclude"}, ...]
        只能從資料庫已知型號清單（用 JSON schema 的 enum 強制）裡辨認，避免
        LLM 生成清單外、查不到資料的字串。

        重試後仍失敗（網路問題等）回傳 None，呼叫端（retrieve()）要 fallback
        回 _match_models 的純字串比對安全網，不能讓整個檢索直接掛掉。
        """
        model_names = sorted(self._canonical_model_names.values())

        # 把別名（例如 "4070" → "rtx4070"）併進清單顯示成「正式名稱（簡稱）」，
        # 不然使用者用簡稱講（"4070 適合打電動嗎"）LLM 可能找不到對應的正式
        # enum 值——實測過純列正式名稱清單會漏掉這種簡稱案例。
        aliases_by_full: dict[str, list[str]] = {}
        for alias, full in self.aliases.items():
            aliases_by_full.setdefault(full, []).append(alias)
        model_list_display = []
        for key in sorted(self._canonical_model_names):
            canonical = self._canonical_model_names[key]
            alias_list = aliases_by_full.get(key)
            if alias_list:
                model_list_display.append(f"{canonical}（簡稱：{'/'.join(sorted(alias_list))}）")
            else:
                model_list_display.append(canonical)

        schema = {
            "type": "object",
            "properties": {
                "mentioned_models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string", "enum": model_names},
                            "polarity": {"type": "string", "enum": ["include", "exclude"]},
                        },
                        "required": ["model", "polarity"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["mentioned_models"],
            "additionalProperties": False,
        }

        system_prompt = (
            "你是一個型號抽取器。從使用者的問題裡找出有沒有提到下面清單裡的"
            "零件型號（清單裡的簡稱也算，要正規化回正式名稱），並判斷使用者"
            "對這個型號的態度：\n"
            "- include：使用者在詢問、比較、考慮、想要這個型號。**兩個型號"
            "放在一起比較、或問『A 跟 B 選哪個』，A 和 B 都算 include**——"
            "使用者是想要這兩個型號的資訊來幫助決定，不是要排除其中一個。\n"
            "- exclude：使用者用明確的排除語氣講到這個型號，且不需要它的"
            "資訊，例如「除了 X 以外」「不要 X」「X 我已經有了」「X 先不看」"
            "「不考慮 X」。單純把兩個型號放在一起比較不算 exclude。\n"
            "只能使用下面清單裡出現過的正式名稱（不能用簡稱當作 model 的值），"
            "不要自己生成清單外的名稱。使用者沒提到型號、或提到的不在清單裡，"
            "就回傳空陣列。\n\n"
            "範例：\n"
            "問「RTX4070 跟 RTX5070 該選哪個」→ "
            '[{"model":"RTX4070","polarity":"include"},{"model":"RTX5070","polarity":"include"}]\n'
            "問「請推薦除了 RTX5070 以外的顯卡」→ "
            '[{"model":"RTX5070","polarity":"exclude"}]\n'
            "問「4070 適合拿來打電動嗎」（簡稱要正規化）→ "
            '[{"model":"RTX4070","polarity":"include"}]\n'
            "問「RTX4070 跟 RTX5070 選一個，但不要跟我推薦 RTX3050」"
            "（比較的兩個都是 include，句尾的排除語氣只影響 RTX3050，"
            "不會影響前面已經判斷為 include 的型號）→ "
            '[{"model":"RTX4070","polarity":"include"},'
            '{"model":"RTX5070","polarity":"include"},'
            '{"model":"RTX3050","polarity":"exclude"}]\n\n'
            f"已知型號清單：{', '.join(model_list_display)}"
        )

        for attempt in range(INTENT_EXTRACT_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=INTENT_MODEL,
                    temperature=0,  # 結構化抽取任務，要的是穩定一致，不要創意變化
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "model_intents",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                )
                data = json.loads(resp.choices[0].message.content)
                return data.get("mentioned_models", [])
            except Exception:
                if attempt == INTENT_EXTRACT_RETRIES - 1:
                    return None
                time.sleep(1.0 * (attempt + 1))
        return None

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

    def _semantic_search(
        self, query: str, top_k: int, exclude_models: list[str] | None = None
    ) -> list[dict]:
        """OpenAI embedding + Chroma 向量搜尋。

        top_k 是「要湊到幾個不同型號」的目標值，不是「取原始結果前 N 筆」。
        做法：把 Chroma 依 distance 排序回傳的候選從頭掃描，每遇到一個還沒
        收錄過的型號就收錄，同型號的其他 chunk 直接跳過（不提前停止），直到
        湊滿 top_k 個不同型號為止。如果目前候選池掃完仍不夠，就擴大池子
        （n_results 倍增，上限 MAX_SEARCH_POOL_SIZE）重新查詢再掃一次；如果
        候選池已經沒有 distance 落在 MAX_DISTANCE 內的項目，或池子已經到
        上限，就如實回傳目前湊到的數量（可能小於 top_k），不放寬距離門檻湊數。
        湊到的每個型號都用 get_by_model() 取完整 chunk 組，攤平後回傳。

        exclude_models：使用者明確排除的型號（來自 _extract_model_intents()
        的 polarity=exclude），掃描候選時遇到就跳過、不收錄進候選型號清單，
        確保這些型號不會透過語意相似度又混進結果——整句話（包含被排除的
        型號名稱）還是會拿去 embedding，候選池本來就可能撈到它的 chunk，
        單靠上層擋掉還不夠。
        """
        exclude_set = {m.lower() for m in (exclude_models or [])}
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
                if model.lower() in exclude_set:
                    continue
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
