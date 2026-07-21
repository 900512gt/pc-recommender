"""
CPU 實際功耗對照表

問題：原價屋/Intel 標稱 TDP 嚴重低估實際功耗。
  例：i7-14700F 標稱 65W，但實際 PL2 可達 219W。
  這會導致散熱相容性檢查被「假 TDP」騙過，推薦壓不住的散熱器。

解法：用 Intel ARK / AMD 官方的實際功耗（PL2 / PPT）當散熱與電源估算依據。

資料來源：
  Intel ARK (ark.intel.com) - Maximum Turbo Power (PL2)
  AMD 官網 - Package Power Tracking (PPT)
"""

# 型號關鍵字 → 實際最大功耗（PL2 / PPT，瓦特）
# 用於散熱器壓制力檢查與 PSU 瓦數估算
CPU_ACTUAL_POWER = {
    # ── Intel Core Ultra 200S (Arrow Lake) ──
    "285K":  250,
    "265K":  250,
    "265KF": 250,
    "245K":  159,
    "245KF": 159,
    "225":   121,
    "225F":  121,

    # ── Intel 14th Gen (Raptor Lake Refresh) ──
    "14900K":  253,
    "14900KF": 253,
    "14900":   219,
    "14700K":  253,
    "14700KF": 253,
    "14700":   219,   # 含 14700F，標稱 65W 但實際 219W
    "14600K":  181,
    "14600KF": 181,
    "14400":   148,
    "14400F":  148,

    # ── Intel 13th Gen (Raptor Lake) ──
    "13900K":  253,
    "13900KF": 253,
    "13900":   219,
    "13700K":  253,
    "13700KF": 253,
    "13700":   219,
    "13600K":  181,
    "13600KF": 181,
    "13400":   148,
    "13400F":  148,

    # ── Intel 12th Gen (Alder Lake) ──
    "12900K": 241,
    "12700K": 190,
    "12600K": 150,
    "12400":  117,
    "12400F": 117,

    # ── AMD Ryzen 9000 (Zen 5) ──
    "9950X3D": 200,
    "9950X":   200,
    "9900X3D": 162,
    "9900X":   162,
    "9800X3D": 162,
    "9700X":   88,
    "9600X":   88,
    "9500F":   65,

    # ── AMD Ryzen 7000 (Zen 4) ──
    "7950X3D": 162,
    "7950X":   230,
    "7900X3D": 162,
    "7900X":   230,
    "7900":    88,
    "7800X3D": 162,
    "7700X":   142,
    "7700":    88,
    "7600X":   142,
    "7600":    88,
    "7500F":   88,

    # ── AMD Ryzen 8000 (Zen 4 Phoenix) ──
    "8700G": 88,
    "8600G": 88,
    "8500G": 88,
    "8400F": 88,

    # ── AMD Ryzen 5000 (Zen 3) ──
    "5950X":   142,
    "5900X":   142,
    "5800X3D": 142,
    "5800X":   142,
    "5700X":   88,
    "5700G":   88,
    "5600X":   88,
    "5600G":   88,
    "5600":    88,

    # ── AMD Threadripper (sTR5) ──
    "9980X":  350,
    "9970X":  350,
    "9960X":  350,
    "9995WX": 350,
    "9985WX": 350,
    "9975WX": 350,
    "9965WX": 350,
    "9955WX": 350,
}


def get_actual_power(cpu_name: str, fallback_tdp: float = 65) -> float:
    """
    從 CPU 名稱查實際最大功耗（PL2/PPT）。
    找不到時回傳標稱 TDP（fallback）。

    用於：
      1. 散熱器壓制力檢查（散熱器要壓得住實際功耗，不是標稱）
      2. PSU 瓦數估算
    """
    # 較長的關鍵字優先比對（避免 "14700" 誤匹配 "14700K"）
    for keyword in sorted(CPU_ACTUAL_POWER, key=len, reverse=True):
        if keyword in cpu_name:
            return float(CPU_ACTUAL_POWER[keyword])
    return float(fallback_tdp)


def needs_high_end_cooler(cpu_name: str) -> bool:
    """
    判斷這顆 CPU 是否需要高階散熱（雙塔風冷 / 240+ 水冷）。
    實際功耗 > 150W 視為需要。
    """
    return get_actual_power(cpu_name) > 150
