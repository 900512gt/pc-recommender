"""
散熱器壓制力

主要來源：資料庫的 max_tdp（廠商標示的解熱瓦數）。風冷 28 款全部都有這個欄位；
水冷 22 款全部是空的（水冷目前也由 config.ENABLE_WATER_COOLING 停用）。
資料庫 max_tdp 有已知錯誤的型號，用下面的 _MAX_TDP_OVERRIDES 校正（不改資料庫檔案）。
沒有 max_tdp 時才用 estimate_cooler_capacity() 從類型、塔數、冷排尺寸推估。

推估依據（業界經驗值，只在沒有 max_tdp 時使用）：

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


# 資料庫 max_tdp 的已知錯誤：(商品名必須同時包含的字串, 校正後瓦數, 理由)
_MAX_TDP_OVERRIDES = [
    (("Peerless Assassin 120 MINI", "黑化"), 265,
     "同型號（ptt_model 相同）的白化版標 265W，黑化版只標 130W，只差顏色，判定為資料錯誤"),
    (("NH-D15S",), 250,
     "貓頭鷹官方不標 TDP，資料庫 180W 比單塔的 Hyper 212 3DHP（200W）還低；"
     "NH-D15S 是高階雙塔，改用同級雙塔（利民 PA120 系列 265~280W）偏保守的 250W"),
]


def get_cooler_capacity(cooler_name: str, specs: dict) -> float:
    """散熱器能壓制的最大 CPU 功耗（瓦特）。順序：校正表 → 資料庫 max_tdp → 推估。"""
    for keys, watts, _reason in _MAX_TDP_OVERRIDES:
        if all(k in cooler_name for k in keys):
            return float(watts)
    try:
        tdp = float(specs.get("max_tdp") or 0)
    except (TypeError, ValueError):
        tdp = 0.0
    if tdp > 0:
        return tdp
    return estimate_cooler_capacity(cooler_name, specs)


def estimate_cooler_capacity(cooler_name: str, specs: dict) -> float:
    """
    推斷散熱器能壓制的最大 CPU 功耗（瓦特）。只在資料庫沒有 max_tdp 時使用，
    一般請呼叫 get_cooler_capacity()。
    """
    name = cooler_name
    cooler_type = specs.get("cooler_type", "")

    # ── 水冷：依冷排尺寸 ──
    if cooler_type == "liquid" or "水冷" in name or "AIO" in name:
        # 資料庫的欄位名稱是 radiator_size；舊程式讀 radiator_mm 永遠讀不到
        try:
            radiator = float(specs.get("radiator_size") or specs.get("radiator_mm") or 0)
        except (TypeError, ValueError):
            radiator = 0
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
