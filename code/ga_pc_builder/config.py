"""
設定檔：路徑、權重、參數
"""
from pathlib import Path

# ─────────────────────────────────────────────
# 路徑設定
# ─────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent

# 整合資料庫（PTT口碑 + 原價屋規格價格），由 src/benchmark_collection/ 的
# merge_benchmark.py + merge_tgp.py 產出，直接讀資料管線的輸出而非另存複本。
DB_PATH = _REPO_ROOT / "data" / "ga_database_v2.json"

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

# 水冷開關。2026-10 決定不推薦水冷：
#   - 評論太少：目錄 22 款水冷只有 5 款有評論（每款 5~17 則），BERT 在水冷上的 test 樣本只有 84 格
#   - 資料庫 22 款水冷的 max_tdp 全部是空的，壓制力只能從冷排尺寸推估
# 設成 False 時目錄不載入水冷，GA 只會配風冷，API 也不接受 cooling_prefer="水冷"。
# GA、相容性檢查裡處理水冷的程式碼都保留，改回 True 就恢復。
ENABLE_WATER_COOLING = False
if not ENABLE_WATER_COOLING:
    CAT_MAP.pop("WATER_COOLER", None)

# ─────────────────────────────────────────────
# 效能與 CP 值的算法（GARecommender 的預設值）
#
# PERF_BASIS：沒有跑分的零件用什麼當效能
#   "hedonic" → hedonic 品質指數（data/hedonic.py：規格換算的價值，品牌、白色、RGB 溢價不算品質）
#               與散熱器解熱瓦數；規格解釋不了價格的類別（機殼）給中性值，不再「越貴越好」
#   "price"   → 價格（舊做法）
# CP_MODE：CP 值的算法
#   "dea"     → Farrell 成本效率：同品質的最低可能價格 / 實際價格（各類別 BCC 價格—品質前緣），
#               少花錢本身不加分，只有「同樣的品質花得比較少」才加分
#   "ratio"   → 效能 / (總價/預算)（舊做法，少花錢 CP 就變高）
# DEA_FRONTIER："vrs"（BCC 變動規模報酬）或 "fdh"（敏感度分析用）
# 實驗依據見 results/hedonic_compare_遊戲*.csv 與專案文件「Hedonic與CP_嚴謹版實驗報告」。
# ─────────────────────────────────────────────
PERF_BASIS = "hedonic"
CP_MODE = "dea"
DEA_FRONTIER = "vrs"

# ─────────────────────────────────────────────
# 三種用途的 Fitness 權重
#
# 效能／口碑／CP值 三項預設等權，不依用途預設「使用者比較重視哪一項」——
# 那是使用者的偏好，由前端的三個滑桿調整（GARecommender._apply_custom_weights）。
# 用途只決定 perf_weights：哪些零件的效能對這個用途比較重要（遊戲看顯卡、工作看 CPU）。
# 改這裡的 w_perf / w_sent / w_cp 時，web/app/builder/page.tsx 的
# USAGE_DEFAULT_WEIGHTS 要一起改。
# ─────────────────────────────────────────────
_EQUAL = 0.70 / 3   # 扣掉 w_budget 0.20、w_compat 0.10 之後三項平分

USAGE_WEIGHTS = {
    "遊戲": {
        "w_perf":   _EQUAL,
        "w_sent":   _EQUAL,
        "w_cp":     _EQUAL,
        "w_budget": 0.20,
        "w_compat": 0.10,
        "perf_weights": {
            "CPU": 0.20, "GPU": 0.40, "記憶體": 0.10,
            "主機板": 0.05, "SSD": 0.10, "HDD": 0.02,
            "散熱": 0.05, "機殼": 0.02, "電源": 0.06,
        }
    },
    "工作": {
        "w_perf":   _EQUAL,
        "w_sent":   _EQUAL,
        "w_cp":     _EQUAL,
        "w_budget": 0.20,
        "w_compat": 0.10,
        "perf_weights": {
            "CPU": 0.35, "GPU": 0.15, "記憶體": 0.20,
            "主機板": 0.08, "SSD": 0.12, "HDD": 0.02,
            "散熱": 0.04, "機殼": 0.02, "電源": 0.02,
        }
    },
    "一般文書": {
        "w_perf":   _EQUAL,
        "w_sent":   _EQUAL,
        "w_cp":     _EQUAL,
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
# 使用者指定口碑面向時，口碑權重分給「指定面向」這一項的比例；剩下的仍看一般口碑。
# 0.5：只指定一個面向時，該面向最好與最差的型號在總分上相差約 0.12，
# 與 fitness 裡的規則扣分（0.05~0.15）同一個量級。
# 實驗依據見 scripts/sentiment_ablation.py（--pref-term-share 可比較不同比例）。
# ─────────────────────────────────────────────
PREF_TERM_SHARE = 0.5

# ─────────────────────────────────────────────
# 註：PRICE_CEILING（無 benchmark 零件的價格正規化天花板）
# 已統一定義於 core/ga_engine.py，此處不再重複定義，避免兩份數值不一致。
# ─────────────────────────────────────────────
