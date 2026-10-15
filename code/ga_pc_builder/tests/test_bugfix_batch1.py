#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
審查 #1（共用 scorer 平行請求）、#8（預算下限）、#10（散熱器價格規則用實際功耗）的測試。

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_bugfix_batch1.py -v
"""
import random
import threading

import pytest

from config import DB_PATH
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer
from data.cpu_power import get_actual_power
from advisor.compatibility import CompatibilityChecker
from core.ga_engine import GARecommender, Build, HARD_FAIL, _PoolView
from core.feasibility import cheapest_build, min_budget, _reachable_ram
from policies.ram_policy import pick_ram

_catalog = _scorer = None


def _setup():
    global _catalog, _scorer
    if _catalog is None:
        _catalog = PartCatalog(DB_PATH)
        _scorer = SentimentScorer()
    return _catalog, _scorer


# ── #8 預算下限 ─────────────────────────────────────────────

def test_min_budget_build_is_fully_compatible():
    """下限那組配置：相容性檢查沒有任何提示、沒被硬約束淘汰、總價不超過下限"""
    catalog, scorer = _setup()
    budget, parts = min_budget(catalog)
    b = Build(dict(parts))
    assert b.total_price <= budget
    penalty, issues = CompatibilityChecker().check(b)
    assert penalty == 0 and issues == [], issues
    ga = GARecommender(catalog, scorer, CompatibilityChecker(), budget=budget, pop_size=2, generations=0)
    assert ga.fitness(b) > HARD_FAIL


def test_min_budget_ram_is_one_ga_can_pick():
    """GA 只用 pick_ram 挑記憶體。下限那組的記憶體必須是 pick_ram 在這個預算下挑得到的
    （舊版直接拿最便宜的 8GB 算出 13,447，但 GA 在這個預算下一定挑 16GB，自己組不出來）"""
    catalog, _ = _setup()
    budget, parts = min_budget(catalog)
    ram, mb = parts["記憶體"], parts["主機板"]
    pool = [r for r in catalog.get("記憶體") if r.specs.get("type") and r.specs["type"] in mb.specs["memory_type"]]
    reachable = _reachable_ram(pool, budget)
    assert any(r is ram for r in reachable)
    random.seed(0)
    picks = [pick_ram(_PoolView({"記憶體": pool}), Build(dict(parts)), budget) for _ in range(500)]
    reachable_ids = {id(r) for r in reachable}
    assert all(id(p) in reachable_ids for p in picks), "pick_ram 挑到分層以外的記憶體，_reachable_ram 跟 pick_ram 的規則不一致"
    assert min(p.price for p in picks) == ram.price, "pick_ram 挑得到的最便宜記憶體，應該就是下限用的那條"


def test_min_budget_is_the_lowest():
    """少 1 元就組不出來；而且不會比「不管記憶體規則」的最便宜組合還低"""
    catalog, _ = _setup()
    budget, _ = min_budget(catalog)
    assert cheapest_build(catalog, budget)[0] <= budget
    assert cheapest_build(catalog, budget - 1)[0] > budget - 1
    assert budget >= cheapest_build(catalog)[0]


def test_api_budget_messages():
    """低於下限一律回同一段中文訊息（原本 < 10,000 是英文的 422，10,000 以上才是中文）"""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import api
    api.limiter.enabled = False
    with TestClient(api.app) as c:
        floor = api.res.min_budget
        for b in (1, 9999, 10000, floor - 1):
            r = c.post("/api/recommend", json={"budget": b, "usage": "一般文書"})
            assert r.status_code == 400 and f"{floor:,}" in r.json()["detail"], (b, r.text)
        assert c.post("/api/recommend", json={"budget": 0, "usage": "一般文書"}).status_code == 422


# ── #10 散熱器價格規則 ───────────────────────────────────────

def _cpu(catalog, key):
    return next(c for c in catalog.get("CPU") if key in c.name)


def test_hot_cpu_with_low_nominal_tdp_not_penalized_for_good_cooler():
    """i7-14700F 標稱 65W、實際 219W：配 3,000 元以內的散熱器不該被當成「低功耗 CPU 配太貴的散熱器」扣分"""
    catalog, _ = _setup()
    cpu = _cpu(catalog, "14700F")
    assert float(cpu.specs["tdp"]) <= 65 and get_actual_power(cpu.name) > 200
    for cooler in catalog.get("風冷"):
        if cooler.price <= 3000:
            assert GARecommender._cooler_price_penalty(cpu, cooler) == 0, cooler.name


def test_low_power_cpu_still_penalized_for_expensive_cooler():
    """規則本身要保留：65W 級的 CPU 配超過 1,200 元的散熱器照樣扣"""
    catalog, _ = _setup()
    cpu = _cpu(catalog, "3400G")
    pricey = max(catalog.get("風冷"), key=lambda a: a.price)
    cheap = min(catalog.get("風冷"), key=lambda a: a.price)
    assert GARecommender._cooler_price_penalty(cpu, pricey) > 0
    assert GARecommender._cooler_price_penalty(cpu, cheap) == 0


def test_amd_tiers_unchanged():
    """改用實際功耗後，AMD 每顆 CPU 的級距都跟原本用標稱 TDP 一樣"""
    catalog, _ = _setup()

    def tier(x, a, b):
        return 0 if x <= a else (1 if x <= b else 2)
    for cpu in catalog.get("CPU"):
        if "AMD" in cpu.name:
            tdp = float(cpu.specs.get("tdp") or 65)
            assert tier(tdp, 65, 125) == tier(get_actual_power(cpu.name, fallback_tdp=tdp), 90, 165), cpu.name


# ── #1 共用 scorer，平行請求 ────────────────────────────────

def test_parallel_requests_share_scorer_safely():
    """api.py 讓所有請求共用同一個 SentimentScorer。不同用途、不同口碑偏好的請求同時跑，
    每個結果的分數都要跟事後單獨重算的一致（舊版 set_usage 會互相蓋掉，還會 KeyError）"""
    catalog, scorer = _setup()
    jobs = [("遊戲", None), ("工作", {"GPU": ["穩定"]}), ("一般文書", {"CPU": ["效能"]}), ("遊戲", {"主機板": ["用料"]})]
    out, errors = [None] * len(jobs), []

    def work(i, usage, prefs):
        try:
            ga = GARecommender(catalog, scorer, CompatibilityChecker(), usage=usage, budget=50000,
                               pop_size=30, generations=10, aspect_prefs=prefs)
            best = ga.run(verbose=False)[0]
            out[i] = (best, ga.fitness(best))
        except Exception as e:          # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i, u, p)) for i, (u, p) in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for (usage, prefs), (best, fit) in zip(jobs, out):
        ga = GARecommender(catalog, scorer, CompatibilityChecker(), usage=usage, budget=50000,
                           pop_size=2, generations=0, aspect_prefs=prefs)
        assert abs(ga.fitness(best) - fit) < 1e-12
