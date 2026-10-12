"""
預算下限：GA 實際組得出來的最便宜相容組合（審查 #8）

API 原本接受最低 10,000 元的預算，但目錄裡最便宜的相容組合要一萬三千多，
低於這個預算 GA 只能回傳超支 20% 以上的配置（fitness -10）。
這裡從目錄算出「GA 組得出來、不超支、也不會被硬約束淘汰的最便宜組合」，API 和文字介面用它當預算下限。
價格每次重建資料庫都會變，所以不寫死，啟動時從目錄算。

條件：
  - 必選零件和散熱器都有
  - CPU-主機板腳位相同、記憶體世代符合主機板、散熱器支援 CPU 腳位、散熱器壓制力 ≥ CPU 實際功耗
    （以上對應 GARecommender.fitness 的硬約束與修補）
  - 機殼放得下主機板板型、顯卡長度、風冷高度（相容性檢查不會有任何提示）
  - 記憶體要是 GA 選得到的：GA 只用 pick_ram 挑記憶體，預算買得起 16GB 就不會選 8GB，
    所以不能直接拿最便宜的記憶體算（那樣算出來的下限 GA 自己組不出來）
"""
from advisor.compatibility import form_factor_fits
from data.cooler_capacity import get_cooler_capacity
from data.cpu_power import get_actual_power
from policies.ram_policy import ram_feasible

_SIMPLE_CATS = ["GPU", "SSD", "電源"]
_COOLING_CATS = ["風冷", "水冷"]


def _cheapest(parts):
    return min(parts, key=lambda p: p.price) if parts else None


def _num(part, key):
    try:
        return float(part.specs.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _sockets(part) -> list:
    s = part.specs.get("supported_sockets") or []
    if isinstance(s, str):
        import json
        try:
            s = json.loads(s)
        except Exception:
            s = [s]
    return list(s)


def _reachable_ram(pool, budget: int) -> list:
    """pick_ram 在這個預算下可能挑到的記憶體（照它的分層：先 32GB、再 16GB、再全部，都在預算比例內）"""
    for min_gb in (32, 16, 0):
        tier = [p for p in pool if ram_feasible(p, budget) and (p.specs.get("capacity_gb") or 0) >= min_gb]
        if tier:
            return tier
    return [_cheapest(pool)] if pool else []


def _case_fits(case, mb, gpu, cooler) -> bool:
    if not form_factor_fits(mb, case):
        return False
    gl, mg = _num(gpu, "length_mm"), _num(case, "max_gpu_mm")
    if gl and mg and gl > mg:
        return False
    if cooler.category == "風冷":
        ch, mc = _num(cooler, "height_mm"), _num(case, "max_cooler_mm")
        if ch and mc and ch > mc:
            return False
    else:
        rs, mw = _num(cooler, "radiator_size"), _num(case, "max_water_mm")
        if rs and mw and rs > mw:
            return False
    return True


def cheapest_build(catalog, budget: int | None = None):
    """最便宜的相容組合。budget 給定時，記憶體只考慮 pick_ram 在該預算下挑得到的；
    不給就不限制記憶體。回傳 (總價, {類別: 零件})，湊不出來回傳 (None, None)。"""
    simple = {c: _cheapest(catalog.get(c)) for c in _SIMPLE_CATS}
    if any(p is None for p in simple.values()):
        return None, None
    gpu = simple["GPU"]
    coolers = sorted((p for c in _COOLING_CATS for p in catalog.get(c)), key=lambda p: p.price)
    cases = sorted(catalog.get("機殼"), key=lambda p: p.price)
    rams = catalog.get("記憶體")

    # 主機板只有腳位、記憶體世代、板型會影響相容性，同一組只需要看最便宜的那張
    mb_groups: dict[tuple, object] = {}
    for mb in catalog.get("主機板"):
        key = (str(mb.specs.get("socket", "")).strip(), str(mb.specs.get("memory_type", "")).strip(),
               str(mb.specs.get("form_factor", "")).strip())
        if key not in mb_groups or mb.price < mb_groups[key].price:
            mb_groups[key] = mb

    case_cache: dict[tuple, object] = {}

    def cheapest_case(mb, cooler):
        key = (mb.specs.get("form_factor"), id(cooler))
        if key not in case_cache:
            case_cache[key] = next((c for c in cases if _case_fits(c, mb, gpu, cooler)), None)
        return case_cache[key]

    ram_cache: dict[str, object] = {}

    def cheapest_ram(mem):
        if mem not in ram_cache:
            pool = [r for r in rams if str(r.specs.get("type", "")).strip()
                    and str(r.specs.get("type", "")).strip() in mem]
            ram_cache[mem] = _cheapest(_reachable_ram(pool, budget) if budget else pool)
        return ram_cache[mem]

    best_price, best = None, None
    for cpu in catalog.get("CPU"):
        sock = str(cpu.specs.get("socket", "")).strip()
        power = get_actual_power(cpu.name, fallback_tdp=float(cpu.specs.get("tdp") or 65))
        ok_coolers = [c for c in coolers
                      if (not _sockets(c) or sock in _sockets(c))
                      and get_cooler_capacity(c.name, c.specs) >= power]
        if not ok_coolers:
            continue
        for (mb_sock, mem, _form), mb in mb_groups.items():
            if mb_sock != sock:
                continue
            ram = cheapest_ram(mem)
            if ram is None:
                continue
            combo = None
            for cooler in ok_coolers:
                case = cheapest_case(mb, cooler)
                if case and (combo is None or cooler.price + case.price < combo[0]):
                    combo = (cooler.price + case.price, cooler, case)
            if combo is None:
                continue
            parts = {"CPU": cpu, "主機板": mb, "記憶體": ram, "機殼": combo[2], combo[1].category: combo[1], **simple}
            price = sum(p.price for p in parts.values())
            if best_price is None or price < best_price:
                best_price, best = price, parts
    return best_price, best


def min_budget(catalog, max_rounds: int = 20):
    """GA 組得出不超支配置的最低預算。

    記憶體能挑什麼取決於預算（預算越高，pick_ram 越會要求 16GB、32GB），所以反覆計算：
    先用不限記憶體的最便宜組合當預算，算出在這個預算下 GA 組得出的最便宜組合，
    比預算貴就把預算提高到那個價格，直到組合價格 ≤ 預算。
    回傳 (預算下限, 那組配置)；湊不出來回傳 (None, None)。
    """
    budget, parts = cheapest_build(catalog)
    if budget is None:
        return None, None
    for _ in range(max_rounds):
        price, parts = cheapest_build(catalog, budget)
        if price is None:
            return None, None
        if price <= budget:
            return budget, parts
        budget = price
    return budget, parts
