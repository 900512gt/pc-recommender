
import random
import math


def get_build_tier(cpu, gpu) -> float:
    cpu_bench = float(cpu.specs.get("benchmark", 0) or 0) if cpu else 0
    gpu_bench = float(gpu.specs.get("benchmark", 0) or 0) if gpu else 0
    cpu_tier  = min(cpu_bench / 58000, 1.0) * 10
    gpu_tier  = min(gpu_bench / 50000, 1.0) * 10
    return cpu_tier * 0.4 + gpu_tier * 0.6


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


#  Feasibility Layer 
RAM_BUDGET_RATIO = 0.18  # RAM 最多佔預算 18%

def ram_feasible(part, budget: int) -> bool:
    """超過預算比例直接排除"""
    return part.price <= budget * RAM_BUDGET_RATIO


#  Preference Layer 
def pick_ram(catalog, build, budget: int):
    """
     feasibility filter（硬性約束）
     feasible set 內做 preference optimization
    """
    cpu  = build.parts.get("CPU")
    gpu  = build.parts.get("GPU")
    pool = catalog.get("記憶體")
    if not pool:
        return None


    # 容量至少 32GB
    feasible = [p for p in pool
                if ram_feasible(p, budget)
                and (p.specs.get("capacity_gb") or 0) >= 32]

    # 如果真的沒得選才會到16GB
    if not feasible:
        feasible = [p for p in pool if ram_feasible(p, budget)]

    # 再不行就選最便宜的
    if not feasible:
        feasible = [min(pool, key=lambda p: p.price)]

 
    tier = get_build_tier(cpu, gpu)

    dual_feasible   = [p for p in feasible if "雙通" in p.name]
    single_feasible = [p for p in feasible if "雙通" not in p.name]

    # CPU+GPU組合等級越高則是偏好雙通道
    if dual_feasible:
        dual_prob = _sigmoid((tier - 7) * 1.5)  # 設Tier 7為 中性點
    else:
        dual_prob = 0.0  # 沒有 feasible 雙通選項

    if dual_feasible and random.random() < dual_prob:
        candidates = dual_feasible
    else:
        candidates = single_feasible if single_feasible else dual_feasible


    def ram_value(p):
        cl_bonus = 200 if "CL30" in p.name else 100 if "CL36" in p.name else 0
        return p.price - cl_bonus

    #  保留隨機性
    k = min(3, len(candidates))
    sample = random.sample(candidates, k)
    return min(sample, key=ram_value)


#  Fitness Correction Layer 
def ram_score(ram, budget: int) -> float:

    if not ram:
        return 0.0

    cap   = float(ram.specs.get("capacity_gb", 0) or 0)
    name  = ram.name
    price = ram.price

    if cap < 16:   return -0.20
    elif cap < 32: return -0.10

    score = +0.02  # 32GB baseline


    is_dual     = "雙通" in name
    sweet_spot  = 12500 if is_dual else 10300

    if price <= sweet_spot:
        score += 0.01
    else:
        score -= (price - sweet_spot) / budget * 0.2

    return score
