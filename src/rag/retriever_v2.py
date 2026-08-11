"""
retriever_v2.py
讀 v2 蒸餾管線輸出的 data/rag_chunks_v2.jsonl（一個型號拆成多個語意 chunk：
summary/aspect/pros_cons/comparison），檢索邏輯照抄 retriever.py 的兩段式架構，
但語意搜尋的最小單位改成「單一 chunk」而不是「整個型號」，讓 TF-IDF 相似度
計算可以聚焦在特定面向/摘要/比較上，而不是被整型號合併後的大段文字稀釋。

跟 retriever.py / retriever_chroma.py / retriever_fulltext.py 完全獨立，
讀不同的檔案、互不影響。切換方式見 server.py 的 RAG_RETRIEVER 環境變數
（本版對應 RAG_RETRIEVER=v2）。

已知限制（不是 bug，是這個方法的天花板）：字元 n-gram TF-IDF 對短、口語化
的查詢，沒有能力可靠分辨「主題內」跟「純粹字面剛好重疊」。實測「晚餐吃什麼
比較好」（離題）的相似度比「推薦一張適合玩遊戲的顯卡」（明確在主題內）還高
（0.1861 vs 0.0728）。用完整測試組驗證過：明確查詢最低分（0.0728）本來就
低於離題查詢最高分（0.1861），數學上不存在一個門檻能同時擋掉全部離題、放行
全部明確查詢——**實測過調高 MIN_RELEVANCE_SCORE（試過 0.08）並不會減少誤判，
只會多誤傷合理查詢，不要嘗試用調高門檻解決**。這正是正式環境選 chroma_v2
（embedding）當主要 backend、不是這一版的原因：embedding 校準過的門檻
（見 retriever_chroma_v2.py 的 MAX_DISTANCE）能正確分辨這類查詢，字元比對
做不到。

檢索策略：
  1. 精確比對型號 → 回傳該型號「全部」chunk（不受 top_k 限制），因為使用者
     指名問特定型號時，應該給完整資訊，而不是像語意搜尋那樣只挑幾個片段。
     這裡不篩「還在賣」，指名問特定型號是合理的口碑查詢，就算已停產也一樣回答。
  2. 找不到型號 → TF-IDF 語意搜尋，在所有 chunk 的 text 逐一比對 cosine
     similarity，只保留「目前還買得到」的型號（比對 data/ga_database_v2.json），
     回傳 top_k 個最相關的「單一 chunk」（可能來自不同型號/面向）。
     這層過濾是為了避免模糊/推薦類查詢（例如「中階顯卡推薦」）撈到已停產的
     舊卡——舊卡討論多、社群共識穩定，語意上反而常常比新卡更像「推薦」用詞，
     沒有這層過濾就可能把停產商品講得像現行選項（做法照抄 retriever_fulltext.py
     的 _load_sellable_models()）。
"""

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.chunk_text import CATEGORY_KEYWORDS

ROOT        = Path(__file__).parent.parent.parent
CHUNKS_FILE = ROOT / "data" / "rag_chunks_v2.jsonl"
GA_DB_FILE  = ROOT / "data" / "ga_database_v2.json"

# 判斷「查詢是否離題」的最低相似度分數。實測過調高到 0.08 想減少誤判，結果
# 反而更糟：「推薦一張適合玩遊戲的顯卡」這種合理查詢只有 0.0728 分，會被 0.08
# 的門檻誤擋，但「晚餐吃什麼比較好」這種離題查詢卻有 0.1861 分，比 0.08 還高，
# 照樣蒙混過關——用完整測試組的 min/max 驗證過，明確查詢最低分（0.0728）本來
# 就低於離題查詢最高分（0.1861），數學上不存在一個門檻能同時擋掉全部離題、
# 放行全部明確查詢。維持在原本的 0.05，不要為了「感覺應該更嚴格」去調高，
# 調高只會誤傷合理查詢，擋不住真正的離題案例。見檔頭「已知限制」的完整說明。
MIN_RELEVANCE_SCORE = 0.05

# 信心不足時，在 context 文字後面加提醒，跟 v1 chunk_to_context() 的 low_confidence 提示同精神
LOW_CONFIDENCE_LEVELS = ("insufficient", "low")


def chunk_to_context_v2(chunk: dict) -> str:
    """v2 chunk 的 text 已經是組好的一段話，這裡只補上信心不足的提醒。"""
    text = chunk["text"]
    if chunk.get("confidence") in LOW_CONFIDENCE_LEVELS:
        text += "\n（注意：此型號評論數量較少，摘要可信度有限）"
    return text


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


class RetrieverV2:
    def __init__(self, chunks_file: Path = CHUNKS_FILE, ga_db_file: Path = GA_DB_FILE):
        self.chunks: list[dict] = []
        with open(chunks_file, encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        self._sellable_models = _load_sellable_models(ga_db_file)

        # 依型號分組（小寫 key），供精確比對用；跟 v1 不同，這裡一個型號對應多個 chunk
        self.chunks_by_model: dict[str, list[dict]] = {}
        for c in self.chunks:
            self.chunks_by_model.setdefault(c["model"].lower(), []).append(c)

        # 建立縮寫別名，跟其他 retriever 相同邏輯：「rtx5070」→「5070」、「amd r7 7800x3d」→「7800x3d」
        self.aliases: dict[str, str] = {}
        for model in self.chunks_by_model:
            short = re.sub(r"^(rtx|rx|amd r\d |intel |amd )", "", model).strip()
            if short and short != model:
                self.aliases[short] = model

        # 建立 TF-IDF 索引：一個 chunk 一列（而不是像 v1 一個型號一列）
        texts = [c["text"] for c in self.chunks]
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    # ── 公開 API（跟 retriever.Retriever 介面相容，但 get_by_model 回傳 list）──

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        """從查詢中找最相關的 chunks。無命中時回傳空 list。"""
        matched = self._match_models(query)
        if matched:
            return matched
        return self._semantic_search(query, top_k)

    def get_by_model(self, model: str) -> list[dict]:
        """直接依型號取該型號全部 chunk（找不到回傳空 list）。"""
        key = model.lower()
        return self.chunks_by_model.get(key) or self.chunks_by_model.get(self.aliases.get(key, ""), [])

    def _match_models(self, query: str) -> list[dict]:
        """在查詢字串中尋找已知型號名稱（含縮寫）。純子字串比對，命中就回傳該型號全部 chunk。"""
        q = query.lower()
        found: list[dict] = []
        seen_models: set[str] = set()

        # 先比對完整型號名（較長的優先，避免「4070」比「4070Ti」早匹配）
        candidates = sorted(self.chunks_by_model.keys(), key=len, reverse=True)
        for model_key in candidates:
            if model_key in q and model_key not in seen_models:
                found.extend(self.chunks_by_model[model_key])
                seen_models.add(model_key)

        # 再比對縮寫別名
        for alias, full in sorted(self.aliases.items(), key=lambda x: len(x[0]), reverse=True):
            if alias in q and full not in seen_models:
                found.extend(self.chunks_by_model.get(full, []))
                seen_models.add(full)

        return found

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """TF-IDF 語意搜尋，最小單位是單一 chunk（可能來自不同型號/面向）。"""
        q_lower = query.lower()
        target_cats: set[str] = set()
        for kw, cats in CATEGORY_KEYWORDS.items():
            if kw in q_lower:
                target_cats |= cats

        if target_cats:
            candidates = [(i, c) for i, c in enumerate(self.chunks) if c["category"] in target_cats]
        else:
            candidates = list(enumerate(self.chunks))

        if not candidates:
            candidates = list(enumerate(self.chunks))

        # 只保留目前還買得到的型號，避免模糊/推薦類查詢撈到已停產的舊卡（見檔頭說明）
        candidates = [(i, c) for i, c in candidates if c["model"].lower() in self._sellable_models]
        if not candidates:
            return []

        idxs       = [i for i, _ in candidates]
        sub_matrix = self._tfidf_matrix[idxs]

        query_vec = self._vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, sub_matrix).flatten()

        if scores.max() < MIN_RELEVANCE_SCORE:
            return []

        top_local = scores.argsort()[::-1][:top_k]
        return [candidates[local_idx][1] for local_idx in top_local]
