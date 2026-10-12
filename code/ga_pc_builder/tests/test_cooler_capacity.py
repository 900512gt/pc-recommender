#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
散熱器壓制力測試（審查 #3）。

壓制力以資料庫 max_tdp 為準，已知錯誤的兩筆用校正表修正，沒有 max_tdp 才用名稱推估。

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_cooler_capacity.py -v
"""
from config import DB_PATH
from data.catalog import Part, PartCatalog
from data.cooler_capacity import get_cooler_capacity, estimate_cooler_capacity, _MAX_TDP_OVERRIDES
from data.cpu_power import get_actual_power

_catalog = None


def _setup():
    global _catalog
    if _catalog is None:
        _catalog = PartCatalog(DB_PATH)
    return _catalog


def _cpu_power(cpu):
    return get_actual_power(cpu.name, fallback_tdp=float(cpu.specs.get("tdp") or 65))


def test_uses_db_max_tdp_except_overrides():
    """沒被校正的風冷，壓制力就是資料庫的 max_tdp"""
    overridden = []
    for a in _setup().get("風冷"):
        cap = get_cooler_capacity(a.name, a.specs)
        if any(all(k in a.name for k in keys) for keys, _, _ in _MAX_TDP_OVERRIDES):
            overridden.append((a.name, cap))
        else:
            assert cap == float(a.specs["max_tdp"]), a.name
    # 校正只套用在 PA120 MINI 黑化版 1 款、NH-D15S 2 款（一般版、黑化版）
    assert len(overridden) == 3, overridden


def test_overrides_fix_known_data_errors():
    catalog = _setup()
    minis = [a for a in catalog.get("風冷") if "Peerless Assassin 120 MINI" in a.name]
    assert len(minis) == 2
    assert len({get_cooler_capacity(a.name, a.specs) for a in minis}) == 1, "同型號黑白兩色壓制力應該相同"
    for a in catalog.get("風冷"):
        if "NH-D15S" in a.name:
            assert get_cooler_capacity(a.name, a.specs) == 250


def test_falls_back_to_estimate_without_max_tdp():
    for bad in (None, 0, "", "abc"):
        specs = {"max_tdp": bad, "cooler_type": "下吹式"}
        assert get_cooler_capacity("某下吹式散熱器", specs) == estimate_cooler_capacity("某下吹式散熱器", specs)


def test_water_reads_radiator_size():
    """舊程式讀 radiator_mm，資料庫的欄位是 radiator_size，所以水冷全部被估成預設值"""
    assert estimate_cooler_capacity("某水冷 AIO", {"radiator_size": 360}) == 300
    assert estimate_cooler_capacity("某水冷 AIO", {"radiator_size": "420"}) == 350
    assert estimate_cooler_capacity("某水冷 AIO", {"radiator_mm": 240}) == 200     # 舊欄位名稱仍可用


def test_every_cpu_has_an_air_cooler_with_headroom():
    """每顆 CPU 至少有一款風冷壓得住、而且不會被扣「散熱餘裕不足」（功耗 ≤ 壓制力 90%）"""
    catalog = _setup()
    best = max(get_cooler_capacity(a.name, a.specs) for a in catalog.get("風冷"))
    for cpu in catalog.get("CPU"):
        assert _cpu_power(cpu) <= best * 0.9, (cpu.name, _cpu_power(cpu), best)


def test_fitness_and_checker_use_same_capacity():
    """fitness 的硬約束和相容性檢查的提示，用的是同一個壓制力"""
    from data.sentiment import SentimentScorer
    from advisor.compatibility import CompatibilityChecker
    from core.ga_engine import GARecommender, Build
    catalog = _setup()
    ga = GARecommender(catalog, SentimentScorer(), CompatibilityChecker(), budget=80000,
                       pop_size=10, generations=1)
    hot = max(catalog.get("CPU"), key=_cpu_power)
    weak = min(catalog.get("風冷"), key=lambda a: get_cooler_capacity(a.name, a.specs))
    import random
    random.seed(0)
    b = ga._random_build()
    b.parts["CPU"] = hot
    b.parts["風冷"] = weak
    ga._repair(b)
    assert ga.fitness(b) <= -5                       # 壓不住：硬約束
    _, issues = ga.checker.check(b)
    assert any("散熱不足" in i for i in issues)
