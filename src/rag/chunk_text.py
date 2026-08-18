"""
chunk_text.py
零件類別的中英文對照表跟關鍵字分類器。

CATEGORY_KEYWORDS 現在只在 retriever_chroma_v2.py 的
_guess_category_variants_by_keyword() 當 LLM 類別分類失敗時的 fallback
安全網使用（正式路徑已經改用 LLM 判斷，準確率量化見 category_keyword_eval.py）；
retriever_v2.py（TF-IDF 版本）則是主要路徑就在用它做類別過濾。
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
