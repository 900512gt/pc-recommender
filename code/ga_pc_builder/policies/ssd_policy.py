"""
SSD 評分政策：多層行為模型，針對不同用途拆分容量、頻寬、延遲、耐久與價格。
"""


def _normalize_price_score(price_score: float) -> float:
    return max(min(price_score, 0.05), -0.05)


def ssd_score(ssd, usage: str = "遊戲") -> float:
    """計算 SSD 評分。

    模型結構：
      1. capacity_score    -> 容量是否滿足需求
      2. bandwidth_score   -> 介面頻寬天花板 / 讀取能力
      3. latency_score     -> NAND 類型代理延遲
      4. endurance_score   -> NAND 耐久 + 工作負載懲罰
      5. price_score       -> 用途導向的 CP 值

    最終分數用權重歸一化，避免單一 factor 對系統造成尺度偏移。
    """

    if not ssd:
        return 0.0

    capacity = float(ssd.specs.get("capacity_gb", 0) or 0)
    interface = str(ssd.specs.get("interface", "") or "").strip().upper()
    read_mbs = float(ssd.specs.get("read_mbs", 0) or 0)
    write_mbs = float(ssd.specs.get("write_mbs", 0) or 0)
    nand = str(ssd.specs.get("nand", "") or "").strip().upper()
    price = float(ssd.price or 0)

    # ── 1. Capacity Score ──────────────────────────────
    if capacity < 1000:
        capacity_score = -0.15
    elif capacity >= 2000:
        capacity_score = 0.03 if usage == "工作" else 0.01
    else:
        capacity_score = 0.0

    # ── 2. Bandwidth Score（介面頻寬天花板）──────────────
    if "SATA" in interface:
        bandwidth_score = -0.15
    else:
        bandwidth_score = min(read_mbs / 7000.0, 1.0) * 0.02
        if "GEN5" in interface:
            bandwidth_score += 0.01 if usage == "工作" else 0.005
        elif "GEN4" in interface:
            bandwidth_score += 0.02
        elif "PCI" in interface:
            bandwidth_score += 0.01

    # ── 3. Latency Score（NAND 類型代理）─────────────────
    if nand == "TLC":
        latency_score = 0.03
    elif nand == "QLC":
        latency_score = -0.02
    else:
        latency_score = 0.0

    # ── 4. Endurance Score（耐久 + 工作負載）─────────────
    endurance_score = 0.0
    if nand == "QLC":
        endurance_score -= 0.02
        if usage == "工作":
            endurance_score -= 0.02
    elif nand == "TLC" and usage == "工作" and write_mbs < 2500:
        endurance_score -= 0.01

    # ── 5. Price Score（用途導向 piecewise）──────────────
    if usage == "遊戲":
        if price <= 5000:
            price_score = 0.05
        elif price <= 5500:
            price_score = 0.02
        elif price <= 6500:
            price_score = 0.0
        else:
            price_score = -0.03
    elif usage == "工作":
        if nand == "TLC" and write_mbs >= 6000:
            price_score = 0.02
        elif write_mbs >= 4000:
            price_score = 0.01
        elif price <= 4500:
            price_score = 0.01
        else:
            price_score = 0.0
    else:
        if "GEN4" in interface and price <= 5000:
            price_score = 0.03
        elif price <= 4000:
            price_score = 0.01
        else:
            price_score = 0.0

    price_score = _normalize_price_score(price_score)

    # ── calibration / normalization layer ─────────────────
    return (
        capacity_score * 0.25
        + bandwidth_score * 0.20
        + latency_score * 0.20
        + endurance_score * 0.20
        + price_score * 0.15
    )
