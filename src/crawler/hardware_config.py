"""
hardware_config.py
各硬體類別的型號清單與搜尋策略設定檔

每筆型號資料格式：
  model      : 顯示用的完整型號名稱
  brand      : 品牌
  series     : 系列（用於統計分類）
  search_kw  : PTT 搜尋關鍵字（第一輪）
  fallback   : 搜不到時改用的備援關鍵字（第二輪）
"""

# ══════════════════════════════════════════════════════
#  支援的類別清單（命令列參數對應名稱）
# ══════════════════════════════════════════════════════

VALID_CATEGORIES = [
    "CPU", "GPU", "RAM", "MB", "SSD", "HDD",
    "AIR_COOLER", "WATER_COOLER", "CASE", "PSU",
]

CATEGORY_DISPLAY = {
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

# ══════════════════════════════════════════════════════
#  CPU
# ══════════════════════════════════════════════════════

CPU_LIST = [
    {"model": "Intel i3-14100",           "brand": "Intel", "series": "i3",           "search_kw": "14100",     "fallback": "14代 i3"},
    {"model": "Intel i5-12400",           "brand": "Intel", "series": "i5",           "search_kw": "12400",     "fallback": "12代 i5"},
    {"model": "Intel i5-14400F",          "brand": "Intel", "series": "i5",           "search_kw": "14400F",    "fallback": "14代 i5"},
    {"model": "Intel i5-14400",           "brand": "Intel", "series": "i5",           "search_kw": "14400",     "fallback": "14代 i5"},
    {"model": "Intel i7-14700F",          "brand": "Intel", "series": "i7",           "search_kw": "14700F",    "fallback": "14代 i7"},
    {"model": "Intel i7-14700",           "brand": "Intel", "series": "i7",           "search_kw": "14700",     "fallback": "14代 i7"},
    {"model": "Intel Core Ultra 5 225F",  "brand": "Intel", "series": "Core Ultra 5", "search_kw": "225F",      "fallback": "Core Ultra 5"},
    {"model": "Intel Core Ultra 5 225",   "brand": "Intel", "series": "Core Ultra 5", "search_kw": "Ultra 225", "fallback": "Core Ultra 5"},
    {"model": "Intel Core Ultra 5 235",   "brand": "Intel", "series": "Core Ultra 5", "search_kw": "Ultra 235", "fallback": "Core Ultra 5"},
    {"model": "Intel Core Ultra 5 245KF", "brand": "Intel", "series": "Core Ultra 5", "search_kw": "245KF",     "fallback": "Arrow Lake"},
    {"model": "Intel Core Ultra 5 245K",  "brand": "Intel", "series": "Core Ultra 5", "search_kw": "245K",      "fallback": "Arrow Lake"},
    {"model": "Intel Core Ultra 7 265KF", "brand": "Intel", "series": "Core Ultra 7", "search_kw": "265KF",     "fallback": "Arrow Lake"},
    {"model": "Intel Core Ultra 7 265K",  "brand": "Intel", "series": "Core Ultra 7", "search_kw": "265K",      "fallback": "Arrow Lake"},
    {"model": "Intel Core Ultra 9 285K",  "brand": "Intel", "series": "Core Ultra 9", "search_kw": "285K",      "fallback": "Arrow Lake"},
    {"model": "AMD R5 3400G",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "3400G",     "fallback": "Ryzen 5 3000"},
    {"model": "AMD R5 5500GT",            "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "5500GT",    "fallback": "Ryzen 5 5000"},
    {"model": "AMD R5 5600XT",            "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "5600XT",    "fallback": "Ryzen 5 5000"},
    {"model": "AMD R5 5600GT",            "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "5600GT",    "fallback": "Ryzen 5 5000"},
    {"model": "AMD R5 5500X3D",           "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "5500X3D",   "fallback": "X3D AM4"},
    {"model": "AMD R5 7500F",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "7500F",     "fallback": "Ryzen 5 7000"},
    {"model": "AMD R5 8400F",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "8400F",     "fallback": "Ryzen 5 8000"},
    {"model": "AMD R5 8500G",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "8500G",     "fallback": "Ryzen 5 8000"},
    {"model": "AMD R5 8600G",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "8600G",     "fallback": "Ryzen 5 8000"},
    {"model": "AMD R5 9500F",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "9500F",     "fallback": "Ryzen 9000"},
    {"model": "AMD R5 9600X",             "brand": "AMD",   "series": "Ryzen 5",      "search_kw": "9600X",     "fallback": "Ryzen 9000"},
    {"model": "AMD R7 7700",              "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "R7 7700",   "fallback": "Ryzen 7 7000"},
    {"model": "AMD R7 7800X3D",           "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "7800X3D",   "fallback": "X3D AM5"},
    {"model": "AMD R7 8700G",             "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "8700G",     "fallback": "Ryzen 7 8000"},
    {"model": "AMD R7 9700X",             "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "9700X",     "fallback": "Ryzen 9000"},
    {"model": "AMD R7 9400X3D",           "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "9400X3D",   "fallback": "X3D AM5"},
    {"model": "AMD R7 9450X3D",           "brand": "AMD",   "series": "Ryzen 7",      "search_kw": "9450X3D",   "fallback": "X3D AM5"},
    {"model": "AMD R9 9900X",             "brand": "AMD",   "series": "Ryzen 9",      "search_kw": "9900X",     "fallback": "Ryzen 9000"},
    {"model": "AMD R9 9900X3D",           "brand": "AMD",   "series": "Ryzen 9",      "search_kw": "9900X3D",   "fallback": "X3D AM5"},
    {"model": "AMD R9 9950X",             "brand": "AMD",   "series": "Ryzen 9",      "search_kw": "9950X",     "fallback": "Ryzen 9000"},
    {"model": "AMD R9 9950X3D",           "brand": "AMD",   "series": "Ryzen 9",      "search_kw": "9950X3D",   "fallback": "X3D AM5"},
]

# ══════════════════════════════════════════════════════
#  其他類別（第二階段以後補上）
# ══════════════════════════════════════════════════════

GPU_LIST = [
    # ── NVIDIA 舊款入門 ──────────────────────────────
    {"model": "GT710",     "brand": "NVIDIA", "series": "GT",    "search_kw": "GT710",   "fallback": "GT 710"},
    {"model": "GT730",     "brand": "NVIDIA", "series": "GT",    "search_kw": "GT730",   "fallback": "GT 730"},
    {"model": "GT1030",    "brand": "NVIDIA", "series": "GT",    "search_kw": "GT1030",  "fallback": "GT 1030"},

    # ── NVIDIA RTX30 ─────────────────────────────────
    {"model": "RTX3050",   "brand": "NVIDIA", "series": "RTX30", "search_kw": "RTX3050", "fallback": "3050 6G"},

    # ── NVIDIA RTX50 ─────────────────────────────────
    {"model": "RTX5050",   "brand": "NVIDIA", "series": "RTX50", "search_kw": "RTX5050", "fallback": "5050"},
    {"model": "RTX5060",   "brand": "NVIDIA", "series": "RTX50", "search_kw": "RTX5060", "fallback": "5060"},
    {"model": "RTX5060Ti", "brand": "NVIDIA", "series": "RTX50", "search_kw": "5060Ti",  "fallback": "5060 Ti"},
    {"model": "RTX5070",   "brand": "NVIDIA", "series": "RTX50", "search_kw": "RTX5070", "fallback": "5070"},
    {"model": "RTX5070Ti", "brand": "NVIDIA", "series": "RTX50", "search_kw": "5070Ti",  "fallback": "5070 Ti"},
    {"model": "RTX5080",   "brand": "NVIDIA", "series": "RTX50", "search_kw": "RTX5080", "fallback": "5080"},

    # ── AMD ──────────────────────────────────────────
    {"model": "RX7650GRE", "brand": "AMD",    "series": "RX7000","search_kw": "7650GRE", "fallback": "RX7650"},
    {"model": "RX9060XT",  "brand": "AMD",    "series": "RX9000","search_kw": "9060XT",  "fallback": "RX9060"},
    {"model": "RX9070GRE", "brand": "AMD",    "series": "RX9000","search_kw": "9070GRE", "fallback": "RX9070GRE"},
    {"model": "RX9070",    "brand": "AMD",    "series": "RX9000","search_kw": "RX9070",  "fallback": "9070"},
    {"model": "RX9070XT",  "brand": "AMD",    "series": "RX9000","search_kw": "9070XT",  "fallback": "RX9070XT"},

    # ── Intel Arc ────────────────────────────────────
    {"model": "Arc B580",  "brand": "Intel",  "series": "Arc",   "search_kw": "B580",    "fallback": "Arc B580"},
]
RAM_LIST = [
    # ── DDR4 ─────────────────────────────────────────
    {"model": "DDR4-3200", "brand": "通用", "series": "DDR4",
     "search_kw": "DDR4 3200", "fallback": "D4 3200"},
    {"model": "DDR4-3600", "brand": "通用", "series": "DDR4",
     "search_kw": "DDR4 3600", "fallback": "D4 3600"},

    # ── DDR5 入門 ─────────────────────────────────────
    {"model": "DDR5-4800", "brand": "通用", "series": "DDR5",
     "search_kw": "DDR5 4800", "fallback": "D5 4800"},
    {"model": "DDR5-5600", "brand": "通用", "series": "DDR5",
     "search_kw": "DDR5 5600", "fallback": "D5 5600"},

    # ── DDR5 甜蜜點 ───────────────────────────────────
    {"model": "DDR5-6000", "brand": "通用", "series": "DDR5",
     "search_kw": "DDR5 6000", "fallback": "D5 6000"},
    {"model": "DDR5-6400", "brand": "通用", "series": "DDR5",
     "search_kw": "DDR5 6400", "fallback": "D5 6400"},

    # ── DDR5 高階 ─────────────────────────────────────
    {"model": "DDR5-6800", "brand": "通用", "series": "DDR5",
     "search_kw": "DDR5 6800", "fallback": "D5 6800"},
    {"model": "DDR5-7200", "brand": "通用", "series": "DDR5",
     "search_kw": "DDR5 7200", "fallback": "D5 7200"},
    {"model": "DDR5-8000", "brand": "通用", "series": "DDR5",
     "search_kw": "DDR5 8000", "fallback": "D5 8000"},
]

MB_LIST = [
    # ── Intel LGA1700（第12/13/14代）────────────────
    {"model": "H610M", "brand": "Intel", "series": "H610",  "search_kw": "H610M",  "fallback": "H610 主機板"},
    {"model": "B760M", "brand": "Intel", "series": "B760",  "search_kw": "B760M",  "fallback": "B760 主機板"},
    {"model": "B760",  "brand": "Intel", "series": "B760",  "search_kw": "B760 主機板", "fallback": "B760 ATX"},
    {"model": "Z790",  "brand": "Intel", "series": "Z790",  "search_kw": "Z790",   "fallback": "Z790 主機板"},

    # ── Intel LGA1851（第15代 Arrow Lake）───────────
    {"model": "H810M", "brand": "Intel", "series": "H810",  "search_kw": "H810M",  "fallback": "H810 主機板"},
    {"model": "H810",  "brand": "Intel", "series": "H810",  "search_kw": "H810 主機板", "fallback": "H810 ATX"},
    {"model": "B840",  "brand": "Intel", "series": "B840",  "search_kw": "B840",   "fallback": "B840 主機板"},
    {"model": "B860M", "brand": "Intel", "series": "B860",  "search_kw": "B860M",  "fallback": "B860 主機板"},
    {"model": "B860",  "brand": "Intel", "series": "B860",  "search_kw": "B860 主機板", "fallback": "B860 ATX"},
    {"model": "Z890",  "brand": "Intel", "series": "Z890",  "search_kw": "Z890",   "fallback": "Z890 主機板"},

    # ── AMD AM4（R5000 / R3000） ─────────────────────
    {"model": "A520M", "brand": "AMD",   "series": "A520",  "search_kw": "A520M",  "fallback": "A520 主機板"},
    {"model": "B550M", "brand": "AMD",   "series": "B550",  "search_kw": "B550M",  "fallback": "B550 主機板"},
    {"model": "B550",  "brand": "AMD",   "series": "B550",  "search_kw": "B550 主機板", "fallback": "B550 ATX"},

    # ── AMD AM5（R7000 / R8000 / R9000） ────────────
    {"model": "A620",  "brand": "AMD",   "series": "A620",  "search_kw": "A620",   "fallback": "A620 主機板"},
    {"model": "B650M", "brand": "AMD",   "series": "B650",  "search_kw": "B650M",  "fallback": "B650 主機板"},
    {"model": "B650",  "brand": "AMD",   "series": "B650",  "search_kw": "B650 主機板", "fallback": "B650 ATX"},
    {"model": "B850M", "brand": "AMD",   "series": "B850",  "search_kw": "B850M",  "fallback": "B850 主機板"},
    {"model": "B850",  "brand": "AMD",   "series": "B850",  "search_kw": "B850 主機板", "fallback": "B850 ATX"},
    {"model": "X870",  "brand": "AMD",   "series": "X870",  "search_kw": "X870 主機板", "fallback": "X870"},
    {"model": "X870E", "brand": "AMD",   "series": "X870E", "search_kw": "X870E",  "fallback": "X870E 主機板"},
]
SSD_LIST = [
    # ── SATA ─────────────────────────────────────────
    {"model": "Samsung 870 EVO", "brand": "Samsung", "series": "SATA",
     "search_kw": "870 EVO",   "fallback": "870EVO"},
    {"model": "Kingston SKC600", "brand": "金士頓",  "series": "SATA",
     "search_kw": "SKC600",    "fallback": "金士頓 SATA"},

    # ── NVMe Gen3 ─────────────────────────────────────
    {"model": "Kingston NV3",    "brand": "金士頓",  "series": "NVMe Gen4",
     "search_kw": "NV3",        "fallback": "金士頓 NV3"},

    # ── NVMe Gen4 ─────────────────────────────────────
    {"model": "Samsung 990 PRO", "brand": "Samsung", "series": "NVMe Gen4",
     "search_kw": "990 PRO",    "fallback": "Samsung 990"},
    {"model": "Samsung 990 EVO", "brand": "Samsung", "series": "NVMe Gen4",
     "search_kw": "990 EVO",    "fallback": "Samsung 990"},
    {"model": "WD SN850X",       "brand": "WD",      "series": "NVMe Gen4",
     "search_kw": "SN850X",     "fallback": "WD 黑標 SSD"},
    {"model": "WD SN5100",       "brand": "WD",      "series": "NVMe Gen4",
     "search_kw": "SN5100",     "fallback": "WD 藍標 SSD"},
    {"model": "Kingston KC3000", "brand": "金士頓",  "series": "NVMe Gen4",
     "search_kw": "KC3000",     "fallback": "金士頓 Gen4"},
    {"model": "Kingston FURY Renegade", "brand": "金士頓", "series": "NVMe Gen4",
     "search_kw": "FURY Renegade", "fallback": "Kingston FURY"},
    {"model": "Micron T500",     "brand": "美光",    "series": "NVMe Gen4",
     "search_kw": "T500",       "fallback": "Micron T500"},
    {"model": "Micron T700",     "brand": "美光",    "series": "NVMe Gen4",
     "search_kw": "T700",       "fallback": "Micron T700"},
    {"model": "Micron P310",     "brand": "美光",    "series": "NVMe Gen4",
     "search_kw": "P310",       "fallback": "Micron P310"},
    {"model": "ADATA LEGEND 860","brand": "威剛",    "series": "NVMe Gen4",
     "search_kw": "LEGEND 860", "fallback": "威剛 LEGEND"},
    {"model": "ADATA LEGEND 900","brand": "威剛",    "series": "NVMe Gen4",
     "search_kw": "LEGEND 900", "fallback": "威剛 LEGEND"},
    {"model": "ADATA XPG S70",   "brand": "威剛",    "series": "NVMe Gen4",
     "search_kw": "XPG S70",    "fallback": "威剛 S70"},
    {"model": "Corsair MP600",   "brand": "海盜船",  "series": "NVMe Gen4",
     "search_kw": "MP600",      "fallback": "海盜船 SSD"},
    {"model": "Acer GM7000",     "brand": "宏碁",    "series": "NVMe Gen4",
     "search_kw": "GM7000",     "fallback": "Acer SSD"},
    {"model": "Acer FA200",      "brand": "宏碁",    "series": "NVMe Gen4",
     "search_kw": "FA200",      "fallback": "Acer FA200"},
    {"model": "ZhiTai TiPlus",   "brand": "致態",    "series": "NVMe Gen4",
     "search_kw": "TiPlus",     "fallback": "致態 SSD"},

    # ── NVMe Gen5 ─────────────────────────────────────
    {"model": "Samsung 9100 PRO","brand": "Samsung", "series": "NVMe Gen5",
     "search_kw": "9100 PRO",   "fallback": "Samsung Gen5"},
    {"model": "Micron T705",     "brand": "美光",    "series": "NVMe Gen5",
     "search_kw": "T705",       "fallback": "Micron T705"},
    {"model": "Micron T710",     "brand": "美光",    "series": "NVMe Gen5",
     "search_kw": "T710",       "fallback": "Micron T710"},
    {"model": "WD SN8100",       "brand": "WD",      "series": "NVMe Gen5",
     "search_kw": "SN8100",     "fallback": "WD Gen5"},
    {"model": "Corsair MP700",   "brand": "海盜船",  "series": "NVMe Gen5",
     "search_kw": "MP700",      "fallback": "海盜船 Gen5"},
    {"model": "ADATA XPG MARS",  "brand": "威剛",    "series": "NVMe Gen5",
     "search_kw": "XPG MARS",   "fallback": "威剛 Gen5"},
    {"model": "ZhiTai TiPro 9000","brand": "致態",   "series": "NVMe Gen5",
     "search_kw": "TiPro 9000", "fallback": "致態 Gen5"},
    {"model": "MSI SPATIUM M580","brand": "MSI",     "series": "NVMe Gen5",
     "search_kw": "M580",       "fallback": "MSI SSD Gen5"},
    {"model": "Acer GM9000",     "brand": "宏碁",    "series": "NVMe Gen5",
     "search_kw": "GM9000",     "fallback": "Acer Gen5"},
]

HDD_LIST = [
    # ── Seagate ───────────────────────────────────────
    {"model": "Seagate 新梭魚 1TB",  "brand": "Seagate", "series": "新梭魚",
     "search_kw": "新梭魚 1T",   "fallback": "Seagate 1TB"},
    {"model": "Seagate 新梭魚 2TB",  "brand": "Seagate", "series": "新梭魚",
     "search_kw": "新梭魚 2T",   "fallback": "Seagate 2TB"},
    {"model": "Seagate 新梭魚 4TB",  "brand": "Seagate", "series": "新梭魚",
     "search_kw": "新梭魚 4T",   "fallback": "Seagate HDD"},
    {"model": "Seagate 新梭魚 8TB",  "brand": "Seagate", "series": "新梭魚",
     "search_kw": "新梭魚 8T",   "fallback": "Seagate HDD"},

    # ── WD ───────────────────────────────────────────
    {"model": "WD 藍標 1TB",   "brand": "WD", "series": "藍標",
     "search_kw": "WD 藍標 1T", "fallback": "WD Blue 1TB"},
    {"model": "WD 藍標 2TB",   "brand": "WD", "series": "藍標",
     "search_kw": "WD 藍標 2T", "fallback": "WD Blue 2TB"},
    {"model": "WD 藍標 4TB",   "brand": "WD", "series": "藍標",
     "search_kw": "WD 藍標 4T", "fallback": "WD Blue 4TB"},
    {"model": "WD 紅標 2TB",   "brand": "WD", "series": "紅標",
     "search_kw": "WD 紅標 2T", "fallback": "WD Red 2TB"},
    {"model": "WD 紅標 4TB",   "brand": "WD", "series": "紅標",
     "search_kw": "WD 紅標 4T", "fallback": "WD Red 4TB"},
    {"model": "WD 黑標 4TB",   "brand": "WD", "series": "黑標",
     "search_kw": "WD 黑標 4T", "fallback": "WD Black 4TB"},

    # ── Toshiba ───────────────────────────────────────
    {"model": "Toshiba P300 2TB", "brand": "Toshiba", "series": "P300",
     "search_kw": "Toshiba P300", "fallback": "東芝 P300"},
    {"model": "Toshiba X300 4TB", "brand": "Toshiba", "series": "X300",
     "search_kw": "Toshiba X300", "fallback": "東芝 X300"},
]
PSU_LIST = [
    # ── 銅牌入門 ──────────────────────────────────────
    {"model": "全漢 聖武士",      "brand": "全漢",   "series": "銅牌",
     "search_kw": "聖武士",      "fallback": "全漢 電源 銅牌"},
    {"model": "九州風神 PK650D",  "brand": "九州風神","series": "銅牌",
     "search_kw": "PK650D",     "fallback": "九州風神 電源"},
    {"model": "微星 MAG A650BN",  "brand": "微星",   "series": "銅牌",
     "search_kw": "MAG A650BN", "fallback": "微星 電源 銅牌"},

    # ── 海韻 Seasonic ─────────────────────────────────
    {"model": "海韻 FOCUS GX",    "brand": "海韻",   "series": "金牌",
     "search_kw": "FOCUS GX",    "fallback": "海韻 電源"},
    {"model": "海韻 CORE GX",     "brand": "海韻",   "series": "金牌",
     "search_kw": "CORE GX",     "fallback": "海韻 ATX3"},
    {"model": "海韻 VERTEX GX",   "brand": "海韻",   "series": "白金",
     "search_kw": "VERTEX GX",   "fallback": "海韻 1200W"},

    # ── 振華 Superflower ──────────────────────────────
    {"model": "振華 COMBAT SG",   "brand": "振華",   "series": "金牌",
     "search_kw": "COMBAT SG",   "fallback": "振華 電源"},
    {"model": "振華 COMBAT FG",   "brand": "振華",   "series": "金牌",
     "search_kw": "COMBAT FG",   "fallback": "振華 電源"},
    {"model": "振華 LEADEX III",  "brand": "振華",   "series": "金牌",
     "search_kw": "LEADEX III",  "fallback": "振華 LEADEX"},
    {"model": "振華 LEADEX VII",  "brand": "振華",   "series": "白金",
     "search_kw": "LEADEX VII",  "fallback": "振華 白金"},

    # ── 酷碼 Cooler Master ────────────────────────────
    {"model": "酷碼 MWE GOLD V3", "brand": "酷碼",   "series": "金牌",
     "search_kw": "MWE GOLD V3", "fallback": "酷碼 電源"},
    {"model": "酷碼 GX II GOLD",  "brand": "酷碼",   "series": "金牌",
     "search_kw": "GX II GOLD",  "fallback": "酷碼 電源"},

    # ── 微星 MSI ──────────────────────────────────────
    {"model": "微星 MAG A650GL",  "brand": "微星",   "series": "金牌",
     "search_kw": "MAG A650GL",  "fallback": "微星 電源 金牌"},
    {"model": "微星 MPG A850GS",  "brand": "微星",   "series": "金牌",
     "search_kw": "MPG A850GS",  "fallback": "微星 電源 金牌"},

    # ── 全漢 FSP ──────────────────────────────────────
    {"model": "全漢 VITA GM",     "brand": "全漢",   "series": "金牌",
     "search_kw": "VITA GM",     "fallback": "全漢 電源 金牌"},
    {"model": "全漢 金鋼彈",       "brand": "全漢",   "series": "SFX",
     "search_kw": "金鋼彈",      "fallback": "全漢 SFX"},

    # ── 華碩 ASUS ─────────────────────────────────────
    {"model": "華碩 TUF GAMING 電源", "brand": "華碩","series": "金牌",
     "search_kw": "TUF GAMING 650W", "fallback": "華碩 TUF 電源"},
    {"model": "華碩 ROG STRIX 電源",  "brand": "華碩","series": "白金",
     "search_kw": "ROG STRIX 850W",  "fallback": "ROG STRIX 電源"},
    {"model": "華碩 ROG LOKI",    "brand": "華碩",   "series": "SFX",
     "search_kw": "ROG LOKI",    "fallback": "ROG LOKI SFX"},
]

AIR_COOLER_LIST = [
    # ── 利民 Thermalright ─────────────────────────────
    {"model": "利民 Peerless Assassin 120", "brand": "利民", "series": "雙塔",
     "search_kw": "Peerless Assassin",  "fallback": "利民 PA120"},
    {"model": "利民 Phantom Spirit 120",   "brand": "利民", "series": "雙塔",
     "search_kw": "Phantom Spirit",     "fallback": "利民 散熱器"},
    {"model": "利民 Frost Commander 140",  "brand": "利民", "series": "雙塔",
     "search_kw": "Frost Commander",    "fallback": "利民 140mm"},
    {"model": "利民 AXP90",               "brand": "利民", "series": "下吹",
     "search_kw": "AXP90",              "fallback": "利民 下吹"},
    {"model": "利民 AXP120",              "brand": "利民", "series": "下吹",
     "search_kw": "AXP120",             "fallback": "利民 下吹"},

    # ── 貓頭鷹 Noctua ─────────────────────────────────
    {"model": "Noctua NH-D15",  "brand": "貓頭鷹", "series": "雙塔",
     "search_kw": "NH-D15",    "fallback": "Noctua 散熱"},
    {"model": "Noctua NH-D15S", "brand": "貓頭鷹", "series": "雙塔",
     "search_kw": "NH-D15S",   "fallback": "Noctua 散熱"},
    {"model": "Noctua NH-U12A", "brand": "貓頭鷹", "series": "單塔",
     "search_kw": "NH-U12A",   "fallback": "Noctua 散熱"},
    {"model": "Noctua NH-U12S", "brand": "貓頭鷹", "series": "單塔",
     "search_kw": "NH-U12S",   "fallback": "Noctua 散熱"},
    {"model": "Noctua NH-L9i",  "brand": "貓頭鷹", "series": "下吹",
     "search_kw": "NH-L9i",    "fallback": "Noctua 下吹"},
    {"model": "Noctua NH-L9a",  "brand": "貓頭鷹", "series": "下吹",
     "search_kw": "NH-L9a",    "fallback": "Noctua 下吹"},

    # ── 九州風神 DEEPCOOL ─────────────────────────────
    {"model": "DEEPCOOL AG400",     "brand": "九州風神", "series": "單塔",
     "search_kw": "AG400",          "fallback": "九州風神 散熱"},
    {"model": "DEEPCOOL AG620",     "brand": "九州風神", "series": "雙塔",
     "search_kw": "AG620",          "fallback": "九州風神 散熱"},
    {"model": "DEEPCOOL ASSASSIN IV","brand": "九州風神", "series": "雙塔",
     "search_kw": "ASSASSIN IV",    "fallback": "阿薩辛 散熱"},

    # ── 酷碼 Cooler Master ────────────────────────────
    {"model": "酷碼 Hyper 212",  "brand": "酷碼", "series": "單塔",
     "search_kw": "Hyper 212",   "fallback": "酷碼 散熱"},
    {"model": "酷碼 Hyper 620S", "brand": "酷碼", "series": "雙塔",
     "search_kw": "Hyper 620S",  "fallback": "酷碼 散熱"},
    {"model": "酷碼 Hyper 622",  "brand": "酷碼", "series": "雙塔",
     "search_kw": "Hyper 622",   "fallback": "酷碼 散熱"},

    # ── 喬思伯 Jonsbo ─────────────────────────────────
    {"model": "喬思伯 CR1400",  "brand": "喬思伯", "series": "單塔",
     "search_kw": "CR1400",     "fallback": "喬思伯 散熱"},
    {"model": "喬思伯 CR3000",  "brand": "喬思伯", "series": "雙塔",
     "search_kw": "CR3000",     "fallback": "喬思伯 散熱"},

    # ── ID-COOLING ────────────────────────────────────
    {"model": "ID-COOLING FROZN A620", "brand": "ID-COOLING", "series": "雙塔",
     "search_kw": "FROZN A620",        "fallback": "ID-COOLING 散熱"},

    # ── Scythe ────────────────────────────────────────
    {"model": "Scythe 無限6", "brand": "Scythe", "series": "單塔",
     "search_kw": "無限6",    "fallback": "Scythe 散熱"},
    {"model": "Scythe 虎徹3", "brand": "Scythe", "series": "單塔",
     "search_kw": "虎徹3",    "fallback": "Scythe 散熱"},
]

WATER_COOLER_LIST = [
    # ── 利民 Thermalright ─────────────────────────────
    {"model": "利民 Frozen Warframe",    "brand": "利民",   "series": "Warframe",
     "search_kw": "Warframe",           "fallback": "利民 水冷"},
    {"model": "利民 Grand Vision 360",   "brand": "利民",   "series": "Grand Vision",
     "search_kw": "Grand Vision",       "fallback": "利民 水冷"},

    # ── 九州風神 DEEPCOOL ─────────────────────────────
    {"model": "九州風神 LT520",          "brand": "九州風神","series": "LT",
     "search_kw": "LT520",             "fallback": "九州風神 水冷"},
    {"model": "九州風神 LT360",          "brand": "九州風神","series": "LT",
     "search_kw": "LT360",             "fallback": "九州風神 水冷"},
    {"model": "九州風神 LM360",          "brand": "九州風神","series": "LM",
     "search_kw": "LM360",             "fallback": "九州風神 水冷"},
    {"model": "九州風神 LQ360",          "brand": "九州風神","series": "LQ",
     "search_kw": "LQ360",             "fallback": "九州風神 水冷"},
    {"model": "九州風神 LE360",          "brand": "九州風神","series": "LE",
     "search_kw": "LE360",             "fallback": "九州風神 水冷"},

    # ── 酷碼 Cooler Master ────────────────────────────
    {"model": "酷碼 MasterLiquid Nex",  "brand": "酷碼",   "series": "Nex",
     "search_kw": "MasterLiquid Nex",  "fallback": "酷碼 水冷"},
    {"model": "酷碼 MasterLiquid Atmos","brand": "酷碼",   "series": "Atmos",
     "search_kw": "MasterLiquid Atmos","fallback": "酷碼 水冷"},

    # ── Montech ───────────────────────────────────────
    {"model": "Montech HyperFlow",      "brand": "Montech", "series": "HyperFlow",
     "search_kw": "HyperFlow",         "fallback": "Montech 水冷"},

    # ── ID-COOLING ────────────────────────────────────
    {"model": "ID-COOLING SL360",       "brand": "ID-COOLING","series": "SL",
     "search_kw": "SL360",             "fallback": "ID-COOLING 水冷"},
    {"model": "ID-COOLING FX360",       "brand": "ID-COOLING","series": "FX",
     "search_kw": "FX360",             "fallback": "ID-COOLING 水冷"},

    # ── 華碩 ASUS ─────────────────────────────────────
    {"model": "華碩 ROG STRIX LC III",  "brand": "華碩",   "series": "LC III",
     "search_kw": "飛龍三代",           "fallback": "LC III 水冷"},
    {"model": "華碩 TUF GAMING LC III", "brand": "華碩",   "series": "TUF LC",
     "search_kw": "TUF LC",            "fallback": "華碩 水冷"},
    {"model": "華碩 ROG RYUO IV",       "brand": "華碩",   "series": "RYUO",
     "search_kw": "RYUO IV",           "fallback": "龍王 水冷"},
    {"model": "華碩 ProArt LC",         "brand": "華碩",   "series": "ProArt LC",
     "search_kw": "ProArt LC",         "fallback": "華碩 水冷"},

    # ── 其他 ──────────────────────────────────────────
    {"model": "COUGAR POSEIDON-GT",     "brand": "COUGAR",  "series": "POSEIDON",
     "search_kw": "POSEIDON-GT",       "fallback": "COUGAR 水冷"},
    {"model": "TRYX Panorama",          "brand": "TRYX",    "series": "Panorama",
     "search_kw": "TRYX Panorama",     "fallback": "TRYX 水冷"},
    {"model": "快睿 CRYO",              "brand": "快睿",    "series": "CRYO",
     "search_kw": "CRYO 水冷",         "fallback": "快睿 水冷"},
]

CASE_LIST = [
    # ── Montech ───────────────────────────────────────
    {"model": "Montech X1",          "brand": "Montech", "series": "ATX",
     "search_kw": "Montech X1",      "fallback": "Montech 機殼"},
    {"model": "Montech X2 PLUS",     "brand": "Montech", "series": "ATX",
     "search_kw": "Montech X2",      "fallback": "Montech 機殼"},
    {"model": "Montech XR",          "brand": "Montech", "series": "ATX",
     "search_kw": "Montech XR",      "fallback": "Montech 機殼"},
    {"model": "Montech Air 100",     "brand": "Montech", "series": "M-ATX",
     "search_kw": "Montech Air 100", "fallback": "Montech Air"},
    {"model": "Montech Air 1000",    "brand": "Montech", "series": "ATX",
     "search_kw": "Montech Air 1000","fallback": "Montech Air"},
    {"model": "Montech SKY ONE",     "brand": "Montech", "series": "ATX",
     "search_kw": "SKY ONE",         "fallback": "Montech SKY"},
    {"model": "Montech SKY TWO",     "brand": "Montech", "series": "ATX",
     "search_kw": "SKY TWO",         "fallback": "Montech SKY"},
    {"model": "Montech KING 65",     "brand": "Montech", "series": "ATX",
     "search_kw": "KING 65",         "fallback": "Montech KING"},
    {"model": "Montech KING 95",     "brand": "Montech", "series": "ATX",
     "search_kw": "KING 95",         "fallback": "Montech KING"},
    {"model": "Montech HS01",        "brand": "Montech", "series": "ATX",
     "search_kw": "Montech HS01",    "fallback": "Montech 背插"},
    {"model": "Montech HS02",        "brand": "Montech", "series": "ATX",
     "search_kw": "Montech HS02",    "fallback": "Montech 背插"},

    # ── 酷碼 Cooler Master ────────────────────────────
    {"model": "酷碼 NR200",          "brand": "酷碼", "series": "ITX",
     "search_kw": "NR200",           "fallback": "酷碼 ITX"},
    {"model": "酷碼 NR200P",         "brand": "酷碼", "series": "ITX",
     "search_kw": "NR200P",          "fallback": "酷碼 ITX"},
    {"model": "酷碼 Q300L",          "brand": "酷碼", "series": "M-ATX",
     "search_kw": "Q300L",           "fallback": "酷碼 機殼"},
    {"model": "酷碼 MasterBox TD500","brand": "酷碼", "series": "ATX",
     "search_kw": "TD500",           "fallback": "酷碼 機殼"},
    {"model": "酷碼 Qube 500",       "brand": "酷碼", "series": "ATX",
     "search_kw": "Qube 500",        "fallback": "酷碼 機殼"},
    {"model": "酷碼 MasterBox 600",  "brand": "酷碼", "series": "ATX",
     "search_kw": "MasterBox 600",   "fallback": "酷碼 機殼"},

    # ── 喬思伯 Jonsbo ─────────────────────────────────
    {"model": "喬思伯 D31",   "brand": "喬思伯", "series": "M-ATX",
     "search_kw": "喬思伯 D31",  "fallback": "Jonsbo D31"},
    {"model": "喬思伯 D41",   "brand": "喬思伯", "series": "ATX",
     "search_kw": "喬思伯 D41",  "fallback": "Jonsbo D41"},
    {"model": "喬思伯 D200",  "brand": "喬思伯", "series": "M-ATX",
     "search_kw": "喬思伯 D200", "fallback": "Jonsbo D200"},
    {"model": "喬思伯 D300",  "brand": "喬思伯", "series": "M-ATX",
     "search_kw": "喬思伯 D300", "fallback": "Jonsbo D300"},
    {"model": "喬思伯 D400",  "brand": "喬思伯", "series": "ATX",
     "search_kw": "喬思伯 D400", "fallback": "Jonsbo D400"},
    {"model": "喬思伯 Z20",   "brand": "喬思伯", "series": "M-ATX",
     "search_kw": "喬思伯 Z20",  "fallback": "Jonsbo Z20"},
    {"model": "喬思伯 N4",    "brand": "喬思伯", "series": "NAS",
     "search_kw": "喬思伯 N4",   "fallback": "Jonsbo NAS"},

    # ── Fractal Design ────────────────────────────────
    {"model": "Fractal North",       "brand": "Fractal", "series": "ATX",
     "search_kw": "Fractal North",   "fallback": "Fractal 機殼"},
    {"model": "Fractal North XL",    "brand": "Fractal", "series": "E-ATX",
     "search_kw": "North XL",        "fallback": "Fractal 機殼"},
    {"model": "Fractal Meshify 3",   "brand": "Fractal", "series": "E-ATX",
     "search_kw": "Meshify 3",       "fallback": "Fractal 機殼"},
    {"model": "Fractal Torrent Compact","brand": "Fractal","series": "E-ATX",
     "search_kw": "Torrent Compact", "fallback": "Fractal 機殼"},
    {"model": "Fractal Define 7",    "brand": "Fractal", "series": "靜音",
     "search_kw": "Define 7",        "fallback": "Fractal 靜音"},
    {"model": "Fractal Terra",       "brand": "Fractal", "series": "ITX",
     "search_kw": "Fractal Terra",   "fallback": "Fractal ITX"},
    {"model": "Fractal Era 2",       "brand": "Fractal", "series": "ITX",
     "search_kw": "Fractal Era",     "fallback": "Fractal ITX"},

    # ── 華碩 ASUS ─────────────────────────────────────
    {"model": "華碩 AP201",  "brand": "華碩", "series": "M-ATX",
     "search_kw": "AP201",   "fallback": "華碩 機殼"},
    {"model": "華碩 AP303",  "brand": "華碩", "series": "ATX",
     "search_kw": "AP303",   "fallback": "華碩 機殼"},
    {"model": "華碩 A21",    "brand": "華碩", "series": "M-ATX",
     "search_kw": "華碩 A21","fallback": "華碩 機殼"},
    {"model": "華碩 A23",    "brand": "華碩", "series": "M-ATX",
     "search_kw": "華碩 A23","fallback": "華碩 機殼"},
    {"model": "華碩 A31",    "brand": "華碩", "series": "ATX",
     "search_kw": "華碩 A31","fallback": "華碩 機殼"},
    {"model": "華碩 ProArt PA401","brand": "華碩","series": "ATX",
     "search_kw": "PA401",   "fallback": "華碩 ProArt 機殼"},
    {"model": "華碩 ROG Helios II","brand": "華碩","series": "E-ATX",
     "search_kw": "Helios II","fallback": "ROG 機殼"},

    # ── 聯力 Lian Li ──────────────────────────────────
    {"model": "聯力 O11 Dynamic Mini V2","brand": "聯力","series": "ATX",
     "search_kw": "O11 Mini V2",    "fallback": "聯力 O11"},
    {"model": "聯力 O11 Dynamic EVO","brand": "聯力","series": "E-ATX",
     "search_kw": "O11 EVO",        "fallback": "聯力 O11"},
    {"model": "聯力 LANCOOL 216",   "brand": "聯力","series": "E-ATX",
     "search_kw": "LANCOOL 216",    "fallback": "聯力 LANCOOL"},
    {"model": "聯力 Vector V100R",  "brand": "聯力","series": "ATX",
     "search_kw": "Vector V100R",   "fallback": "聯力 機殼"},

    # ── 微星 MSI ──────────────────────────────────────
    {"model": "微星 MAG FORGE M100R","brand": "微星","series": "M-ATX",
     "search_kw": "FORGE M100R",    "fallback": "微星 機殼"},
    {"model": "微星 MAG PANO 130R", "brand": "微星","series": "ATX",
     "search_kw": "PANO 130R",      "fallback": "微星 機殼"},
    {"model": "微星 MAG PANO 100R", "brand": "微星","series": "ATX",
     "search_kw": "PANO 100R",      "fallback": "微星 機殼"},

    # ── NZXT ─────────────────────────────────────────
    {"model": "NZXT H3 Flow", "brand": "NZXT", "series": "M-ATX",
     "search_kw": "NZXT H3",  "fallback": "NZXT 機殼"},
    {"model": "NZXT H5 Flow", "brand": "NZXT", "series": "E-ATX",
     "search_kw": "NZXT H5",  "fallback": "NZXT 機殼"},
    {"model": "NZXT H6 Flow", "brand": "NZXT", "series": "ATX",
     "search_kw": "NZXT H6",  "fallback": "NZXT 機殼"},
    {"model": "NZXT H7 Flow", "brand": "NZXT", "series": "E-ATX",
     "search_kw": "NZXT H7",  "fallback": "NZXT 機殼"},
    {"model": "NZXT H9 Flow", "brand": "NZXT", "series": "ATX",
     "search_kw": "NZXT H9",  "fallback": "NZXT 機殼"},

    # ── Phanteks ─────────────────────────────────────
    {"model": "Phanteks XT M3",      "brand": "Phanteks","series": "M-ATX",
     "search_kw": "Phanteks XT M3",  "fallback": "Phanteks 機殼"},
    {"model": "Phanteks XT Pro Ultra","brand": "Phanteks","series": "ATX",
     "search_kw": "XT Pro Ultra",    "fallback": "Phanteks 機殼"},
    {"model": "Phanteks NV5",        "brand": "Phanteks","series": "ATX",
     "search_kw": "Phanteks NV5",    "fallback": "Phanteks 機殼"},
    {"model": "Phanteks Evolv X2",   "brand": "Phanteks","series": "ATX",
     "search_kw": "Evolv X2",        "fallback": "Phanteks 機殼"},

    # ── 曜越 Thermaltake ──────────────────────────────
    {"model": "曜越 The Tower 300",  "brand": "曜越","series": "M-ATX",
     "search_kw": "The Tower 300",   "fallback": "曜越 Tower"},
    {"model": "曜越 The Tower 600",  "brand": "曜越","series": "ATX",
     "search_kw": "The Tower 600",   "fallback": "曜越 Tower"},
    {"model": "曜越 View 290",       "brand": "曜越","series": "ATX",
     "search_kw": "View 290",        "fallback": "曜越 機殼"},

    # ── HYTE ──────────────────────────────────────────
    {"model": "HYTE Y70",   "brand": "HYTE","series": "E-ATX",
     "search_kw": "HYTE Y70","fallback": "HYTE 機殼"},
    {"model": "HYTE X50",   "brand": "HYTE","series": "ATX",
     "search_kw": "HYTE X50","fallback": "HYTE 機殼"},

    # ── 銀欣 SilverStone ──────────────────────────────
    {"model": "銀欣 SUGO 16",  "brand": "銀欣","series": "ITX",
     "search_kw": "SUGO 16",   "fallback": "銀欣 ITX"},
    {"model": "銀欣 LD03",     "brand": "銀欣","series": "ITX",
     "search_kw": "銀欣 LD03", "fallback": "銀欣 ITX"},
]

# ══════════════════════════════════════════════════════
#  類別 → 型號清單 對應表
# ══════════════════════════════════════════════════════

CATEGORY_ITEMS = {
    "CPU":          CPU_LIST,
    "GPU":          GPU_LIST,
    "RAM":          RAM_LIST,
    "MB":           MB_LIST,
    "SSD":          SSD_LIST,
    "HDD":          HDD_LIST,
    "AIR_COOLER":   AIR_COOLER_LIST,
    "WATER_COOLER": WATER_COOLER_LIST,
    "CASE":         CASE_LIST,
    "PSU":          PSU_LIST,
}