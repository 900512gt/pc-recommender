"""
設定檔：路徑、權重、參數
"""
from pathlib import Path

# ─────────────────────────────────────────────
# 路徑設定
# ─────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent

# 朋友提供的整合資料庫（PTT口碑 + 原價屋規格價格）
DB_PATH = Path(__file__).parent / "data" / "ga_database_v2.json"

# 你的巴哈評論標注資料
MATCHED_FILES = [
    _REPO_ROOT / "match_data" / "matched_part1.jsonl",
    _REPO_ROOT / "match_data" / "matched_part2.jsonl",
    _REPO_ROOT / "match_data" / "matched_part3.jsonl",
]

# ─────────────────────────────────────────────
# 類別對應表（ga_database_v2.json → 系統內部名稱）
# ─────────────────────────────────────────────
CAT_MAP = {
    "CPU":          "CPU",
    "GPU":          "GPU",
    "RAM":          "記憶體",
    "MB":           "主機板",
    "SSD":          "SSD",
    "HDD":          "HDD",
    "AIR_COOLER":   "風冷",
    "WATER_COOLER": "水冷",
    "CASE":         "機殼",
    "PSU":          "電源",
}

# ─────────────────────────────────────────────
# 三種用途的 Fitness 權重
# ─────────────────────────────────────────────
USAGE_WEIGHTS = {
    "遊戲": {
        "w_perf":   0.35,
        "w_sent":   0.20,
        "w_cp":     0.15,
        "w_budget": 0.20,
        "w_compat": 0.10,
        "perf_weights": {
            "CPU": 0.20, "GPU": 0.40, "記憶體": 0.10,
            "主機板": 0.05, "SSD": 0.10, "HDD": 0.02,
            "散熱": 0.05, "機殼": 0.02, "電源": 0.06,
        }
    },
    "工作": {
        "w_perf":   0.30,
        "w_sent":   0.20,
        "w_cp":     0.20,
        "w_budget": 0.20,
        "w_compat": 0.10,
        "perf_weights": {
            "CPU": 0.35, "GPU": 0.15, "記憶體": 0.20,
            "主機板": 0.08, "SSD": 0.12, "HDD": 0.02,
            "散熱": 0.04, "機殼": 0.02, "電源": 0.02,
        }
    },
    "一般文書": {
        "w_perf":   0.15,
        "w_sent":   0.20,
        "w_cp":     0.35,
        "w_budget": 0.20,
        "w_compat": 0.10,
        "perf_weights": {
            "CPU": 0.30, "GPU": 0.05, "記憶體": 0.20,
            "主機板": 0.10, "SSD": 0.15, "HDD": 0.05,
            "散熱": 0.05, "機殼": 0.05, "電源": 0.05,
        }
    }
}

# ─────────────────────────────────────────────
# 註：PRICE_CEILING（無 benchmark 零件的價格正規化天花板）
# 已統一定義於 core/ga_engine.py，此處不再重複定義，避免兩份數值不一致。
# ─────────────────────────────────────────────
