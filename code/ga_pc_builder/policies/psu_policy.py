"""
理論上來說夠用即可
"""


def _estimate_watt(cpu, gpu) -> float:
    """估算電供"""
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

    # GPU tier 
    if gpu_tdp >= 450:   psu_floor = 1000
    elif gpu_tdp >= 300: psu_floor = 850
    elif gpu_tdp >= 200: psu_floor = 750
    elif gpu_tdp >= 150: psu_floor = 650
    else:                psu_floor = 550

    # sustained + transient 
    base            = cpu_tdp * 0.6 + gpu_tdp * 0.6 + system_pwr
    transient       = gpu_tdp * 0.6
    sustained_floor = (base + transient) * 1.1

    return max(psu_floor, sustained_floor)


def psu_score(psu, cpu, gpu, psu_tier: str = "standard") -> float:

    if not psu:
        return 0.0

    estimated = _estimate_watt(cpu, gpu)

    # psu_tier scaling
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

    # 銅牌
    rating = str(psu.specs.get("rating", "")).strip()
    copper_pen = -0.05 if "銅" in rating else 0.0

    # Plateau model
    if psu_watt < estimated:
        # 不足重懲罰
        return -0.5 + copper_pen

    elif psu_watt <= estimated * 1.3:
        # 合理區間plateau，給固定小加分
        return 0.02 + copper_pen

    else:
        # 過度冗餘漸進懲罰
        excess_ratio = (psu_watt - estimated * 1.3) / estimated
        return 0.02 - excess_ratio * 0.15 + copper_pen
