"""
chunk_text.py
給 embed_chunks.py / retriever_chroma.py 共用的 chunk → 純文字轉換。
跟 retriever.py 裡的 _chunk_to_text 邏輯一致，但獨立一份，
確保 Chroma 那條路徑跟原本的 TF-IDF 路徑完全不互相依賴，方便日後比較。
"""

CATEGORY_ZH = {
    "GPU": "顯示卡", "CPU": "處理器", "SSD": "固態硬碟",
    "MB": "主機板",  "PSU": "電源",   "RAM": "記憶體",
    "HDD": "硬碟",   "AIR_COOLER": "風冷", "WATER_COOLER": "水冷",
    "CASE": "機殼",
    "主機板": "主機板", "記憶體": "記憶體", "電源": "電源",
    "機殼": "機殼",   "水冷": "水冷",   "風冷": "風冷",
}

# 使用者可能輸入的 category 關鍵字 → 對應的 category 集合（跟 retriever.py 相同）
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


def chunk_to_text(chunk: dict) -> str:
    """把 chunk 轉成可供 embedding 的純文字。"""
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
    """把 chunk 轉成給 LLM 讀的結構化文字（跟 retriever.py 的版本相同）。"""
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
