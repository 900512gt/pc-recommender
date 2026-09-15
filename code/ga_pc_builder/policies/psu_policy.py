"""
電源評分：夠用即可，過度配置扣分（機會成本）

設計原則：
  1. 不足 → 重懲罰（安全問題）
  2. 夠用區間 → 高分（plateau）
  3. 過度配置 → 漸進扣分（多花的錢是機會成本，本可用於效能零件）
  4. psu_tier 保留升級彈性：想留升級空間者，需求上調，允許大電源

修正紀錄（v2）：
  - 降低 psu_floor 高估：原本 250W GPU 就要求 750W floor，導致 650W 被誤殺。
    實際 490W 系統，650W 綽綽有餘，不應被判不足。
  - 收窄 plateau 並加重過度配置懲罰：原本 estimated×1.3 太寬且懲罰係數僅 0.15，
    導致 850W 與 750W 同分，GA 選了貴 4500 元的冗餘電源。
"""


def _estimate_watt(cpu, gpu) -> float:
    """估算系統所需電供瓦數。"""
    cpu_tdp    = float(cpu.specs.get("tdp", 65) or 65) if cpu else 65
    system_pwr = 80

    # K/KF 系列實際功耗修正
    if cpu:
        cpu_name = cpu.name
        if "265K" in cpu_name or "265KF" in cpu_name:
            cpu_tdp = 180
        elif "245K" in cpu_name or "245KF" in cpu_name:
            cpu_tdp = 160

    if gpu:
        if gpu.price > 40000:   gpu_tdp = 500
        elif gpu.price > 28000: gpu_tdp = 350
        elif gpu.price > 18000: gpu_tdp = 250
        elif gpu.price > 12000: gpu_tdp = 200
        elif gpu.price > 8000:  gpu_tdp = 180
        elif gpu.price > 5000:  gpu_tdp = 150
        elif gpu.price > 4000:  gpu_tdp = 120
        else:                   gpu_tdp = 90
    else:
        gpu_tdp = 150

    # GPU tier 對應的安全底線（修正：降低過度保守的 floor）
    if gpu_tdp >= 450:   psu_floor = 850
    elif gpu_tdp >= 300: psu_floor = 750
    elif gpu_tdp >= 200: psu_floor = 600   # 原 750，過度保守
    elif gpu_tdp >= 150: psu_floor = 550   # 原 650
    else:                psu_floor = 450

    # sustained + transient 尖峰
    base            = cpu_tdp * 0.6 + gpu_tdp * 0.6 + system_pwr
    transient       = gpu_tdp * 0.6
    sustained_floor = (base + transient) * 1.1

    return max(psu_floor, sustained_floor)


def psu_score(psu, cpu, gpu, psu_tier: str = "standard") -> float:
    if not psu:
        return 0.0

    estimated = _estimate_watt(cpu, gpu)

    # psu_tier：想留升級空間 → 需求上調，允許更大電源不被罰
    if psu_tier == "upgrade":
        estimated *= 1.15
    elif psu_tier == "flagship":
        estimated *= 1.30

    try:
        psu_watt = float(psu.specs.get("wattage", 0) or 0)
    except (ValueError, TypeError):
        return 0.0

    if psu_watt <= 0:
        return 0.0

    # 銅牌用料扣分
    rating = str(psu.specs.get("rating", "")).strip()
    copper_pen = -0.05 if "銅" in rating else 0.0

    # 分段評分
    if psu_watt < estimated * 0.95:
        # 不足：安全問題，重懲罰
        return -0.5 + copper_pen
    elif psu_watt <= estimated * 1.25:
        # 夠用區間：plateau 高分（1.25 讓 650→750 這種小升級不被罰）
        return 0.02 + copper_pen
    else:
        # 過度配置：漸進扣分，力道加重（0.15→0.5），讓機會成本被 GA 感知
        excess_ratio = (psu_watt - estimated * 1.25) / estimated
        return 0.02 - excess_ratio * 0.5 + copper_pen
