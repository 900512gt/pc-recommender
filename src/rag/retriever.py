"""
retriever.py
從 rag_chunks.jsonl 檢索相關 chunk，供 chat.py 組裝 prompt 用。

檢索策略：
  1. 從使用者查詢中抽取型號名稱 → 直接回傳對應 chunk（精確比對）
  2. 找不到型號 → TF-IDF 語意搜尋（category 關鍵字 + 查詢內容）
"""

import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT        = Path(__file__).parent.parent.parent
CHUNKS_FILE = ROOT / "data" / "rag_chunks.jsonl"

# category 中英對照（統一用中文）
CATEGORY_ZH = {
    "GPU": "顯示卡", "CPU": "處理器", "SSD": "固態硬碟",
    "MB": "主機板",  "PSU": "電源",   "RAM": "記憶體",
    "HDD": "硬碟",   "AIR_COOLER": "風冷", "WATER_COOLER": "水冷",
    "CASE": "機殼",
    # 已是中文的直接保留
    "主機板": "主機板", "記憶體": "記憶體", "電源": "電源",
    "機殼": "機殼",   "水冷": "水冷",   "風冷": "風冷",
}

# 使用者可能輸入的 category 關鍵字 → 對應的 category 集合
CATEGORY_KEYWORDS = {
    "顯示卡": {"GPU", "顯示卡"},
    "顯卡":   {"GPU", "顯示卡"},
    "gpu":    {"GPU", "顯示卡"},
    "處理器": {"CPU", "處理器"},
    "cpu":    {"CPU", "處理器"},
    "ssd":    {"SSD", "固態硬碟"},
    "固態":   {"SSD", "固態硬碟"},
    "主機板": {"MB", "主機板"},
    "mb":     {"MB", "主機板"},
    "記憶體": {"記憶體", "RAM"},
    "ram":    {"記憶體", "RAM"},
    "電源":   {"電源", "PSU"},
    "機殼":   {"機殼", "CASE"},
    "水冷":   {"水冷", "WATER_COOLER"},
    "風冷":   {"風冷", "AIR_COOLER"},
    "散熱":   {"水冷", "風冷", "AIR_COOLER", "WATER_COOLER"},
}


def _chunk_to_text(chunk: dict) -> str:
    """把 chunk 轉成可供 TF-IDF 索引的純文字。"""
    parts = [
        chunk["model"],
        CATEGORY_ZH.get(chunk["category"], chunk["category"]),
        chunk.get("summary", ""),
        " ".join(chunk.get("pros", [])),
        " ".join(chunk.get("cons", [])),
        " ".join(chunk.get("comparisons", [])),
    ]
    return " ".join(p for p in parts if p)


def chunk_to_context(chunk: dict) -> str:
    """把 chunk 轉成給 LLM 讀的結構化文字（省 token 版）。"""
    cat = CATEGORY_ZH.get(chunk["category"], chunk["category"])
    lines = [f"【{chunk['model']} {cat} 社群評價】（共 {chunk['comment_count']} 則評論）"]

    if chunk.get("pros"):
        lines.append("優點：" + "、".join(chunk["pros"]))
    if chunk.get("cons"):
        lines.append("缺點：" + "、".join(chunk["cons"]))

    if chunk.get("aspects"):
        for k, v in chunk["aspects"].items():
            lines.append(f"{k}：{v}")

    if chunk.get("comparisons"):
        lines.append("常被比較：" + "、".join(chunk["comparisons"]))

    if chunk.get("summary"):
        lines.append(f"整體評價：{chunk['summary']}")

    if chunk.get("low_confidence"):
        lines.append("（注意：此型號評論數量較少，摘要可信度有限）")

    return "\n".join(lines)


class Retriever:
    def __init__(self, chunks_file: Path = CHUNKS_FILE):
        self.chunks: list[dict] = []
        with open(chunks_file, encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        # 建立型號 → chunk 的索引（小寫 key 方便比對）
        self.model_index: dict[str, dict] = {
            c["model"].lower(): c for c in self.chunks
        }

        # 建立縮寫別名：「5070」→「RTX5070」、「7800X3D」→「AMD R7 7800X3D」等
        self.aliases: dict[str, str] = {}
        for model in self.model_index:
            short = re.sub(r"^(rtx|rx|amd r\d |intel |amd )", "", model).strip()
            if short and short != model:
                self.aliases[short] = model

        # 建立 TF-IDF 索引（精確比對驗證 + 語意搜尋共用）
        texts = [_chunk_to_text(c) for c in self.chunks]
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    # ── 公開 API ──────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        """從查詢中找最相關的 chunks。無命中時回傳空 list。"""
        matched = self._match_models(query)
        if matched:
            return matched[:top_k]
        return self._semantic_search(query, top_k)

    def get_by_model(self, model: str) -> dict | None:
        """直接依型號取 chunk。"""
        key = model.lower()
        return self.model_index.get(key) or self.model_index.get(self.aliases.get(key, ""))

    def _match_models(self, query: str) -> list[dict]:
        """在查詢字串中尋找已知型號名稱（含縮寫）。純子字串比對，不做相關性判斷。"""
        q = query.lower()
        found: list[dict] = []
        seen: set[str] = set()

        # 先比對完整型號名（較長的優先，避免「4070」比「4070Ti」早匹配）
        candidates = sorted(self.model_index.keys(), key=len, reverse=True)
        for model_key in candidates:
            if model_key in q and model_key not in seen:
                found.append(self.model_index[model_key])
                seen.add(model_key)

        # 再比對縮寫別名
        for alias, full in sorted(self.aliases.items(), key=lambda x: len(x[0]), reverse=True):
            if alias in q and full not in seen:
                chunk = self.model_index.get(full)
                if chunk:
                    found.append(chunk)
                    seen.add(full)

        return found

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """TF-IDF 語意搜尋，回傳帶真實 score 的 chunk list。"""
        q_lower = query.lower()
        target_cats: set[str] = set()
        for kw, cats in CATEGORY_KEYWORDS.items():
            if kw in q_lower:
                target_cats |= cats

        if target_cats:
            candidates = [(i, c) for i, c in enumerate(self.chunks)
                          if c["category"] in target_cats]
        else:
            candidates = list(enumerate(self.chunks))

        if not candidates:
            candidates = list(enumerate(self.chunks))

        idxs       = [i for i, _ in candidates]
        sub_matrix = self._tfidf_matrix[idxs]

        query_vec = self._vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, sub_matrix).flatten()

        if scores.max() < 0.05:
            return []

        top_local = scores.argsort()[::-1][:top_k]
        return [candidates[local_idx][1] for local_idx in top_local]
