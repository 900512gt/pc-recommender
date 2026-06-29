"""
散熱器壓制力推斷

問題：原價屋散熱器規格沒有「支援 TDP」欄位（資料庫 0/224 有此資料）。
解法：從散熱器類型、塔數、冷排尺寸推斷壓制力（可壓多少瓦）。

推斷依據（業界經驗值）：
  下吹式薄型（AXP90, L9）        : ~95W
  下吹式標準                     : ~125W
  單塔單扇風冷                   : ~150W
  單塔雙扇 / 大型單塔            : ~180W
  雙塔風冷（AK620, Assassin）    : ~250W
  120/240 水冷                   : ~200W
  280 水冷                       : ~250W
  360 水冷                       : ~300W
  420 水冷                       : ~350W
"""


def estimate_cooler_capacity(cooler_name: str, specs: dict) -> float:
    """
    推斷散熱器能壓制的最大 CPU 功耗（瓦特）。
    """
    name = cooler_name
    cooler_type = specs.get("cooler_type", "")

    # ── 水冷：依冷排尺寸 ──
    if cooler_type == "liquid" or "水冷" in name or "AIO" in name:
        radiator = specs.get("radiator_mm", 0)
        # 從名稱補抓冷排尺寸
        if not radiator:
            for size in [420, 360, 280, 240, 120]:
                if str(size) in name:
                    radiator = size
                    break
        if radiator >= 420:   return 350
        if radiator >= 360:   return 300
        if radiator >= 280:   return 250
        if radiator >= 240:   return 200
        if radiator >= 120:   return 150
        return 200   # 水冷預設（抓不到尺寸時保守給 200）

    # ── 風冷：依形式判斷 ──
    # 下吹式薄型（ITX 用）
    if any(k in name for k in ["AXP90", "AXP", "L9", "薄型", "LP"]):
        return 95
    # 下吹式標準
    if "下吹" in name:
        return 125

    # 雙塔風冷（高階）
    if any(k in name for k in ["雙塔", "AK620", "Assassin", "阿薩辛", "AK-620",
                                "NH-D15", "Dark Rock Pro", "PA120", "FrostFlow",
                                "雙塔反葉", "雙塔雙扇"]):
        return 250

    # 大型單塔 / 單塔雙扇
    if any(k in name for k in ["單塔雙扇", "無限6", "AK400", "PA120 SE",
                                "Frost", "Hyper 212", "6導管", "7導管"]):
        return 180

    # 一般塔散（預設）
    if cooler_type == "air" or "風冷" in name or "塔" in name:
        return 150

    # 抓不到 → 保守給 130
    return 130
