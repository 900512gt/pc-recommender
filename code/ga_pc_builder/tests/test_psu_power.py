#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
電源相關測試：審查 #19（16-pin 接頭）與電源瓦數估算。

- 16-pin 顯卡配到沒有原生 16-pin 的電源：由使用者決定接不接受轉接線（allow_16pin_adapter）
  不接受（預設）→ 相容性扣分，修補時換成有原生接頭、瓦數不低於原本的電源
  接受 → 只提示、不扣分
- 電源需求瓦數：顯卡改用資料庫 tgp_watts、CPU 改用實際功耗（原本顯卡依價格推估）

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_psu_power.py -v
"""
import random

from config import DB_PATH
from data.catalog import Part, PartCatalog
from data.sentiment import SentimentScorer
from advisor.compatibility import CompatibilityChecker, needs_16pin, has_native_16pin
from advisor.upgrade_advisor import UpgradeAdvisor
from core.ga_engine import GARecommender, Build
from policies.psu_policy import required_watt, _gpu_watt_from_price

_catalog = _scorer = None


def _setup():
    global _catalog, _scorer
    if _catalog is None:
        _catalog = PartCatalog(DB_PATH)
        _scorer = SentimentScorer()
    return _catalog, _scorer


def _first(catalog, cat, cond):
    return next(p for p in catalog.get(cat) if cond(p))


def _watt(p):
    return float(p.specs["wattage"])


# ── 資料 ───────────────────────────────────────────────────

def test_connector_data_is_readable():
    catalog, _ = _setup()
    n16 = sum(needs_16pin(g) for g in catalog.get("GPU"))
    native = sum(has_native_16pin(p) for p in catalog.get("電源"))
    assert n16 > 0 and native > 0
    assert native < len(catalog.get("電源")), "要有沒有原生 16-pin 的電源，測試才有意義"


# ── 相容性檢查 ─────────────────────────────────────────────

def _pair(catalog, gpu16: bool, native: bool):
    gpu = _first(catalog, "GPU", lambda g: needs_16pin(g) == gpu16)
    psu = _first(catalog, "電源", lambda p: has_native_16pin(p) == native)
    return Build({"GPU": gpu, "電源": psu})


def test_checker_penalizes_adapter_when_not_allowed():
    catalog, _ = _setup()
    pen, issues = CompatibilityChecker().check(_pair(catalog, True, False))
    assert pen > 0 and any("16-pin" in i for i in issues)


def test_checker_only_notes_adapter_when_allowed():
    catalog, _ = _setup()
    pen, issues = CompatibilityChecker().check(_pair(catalog, True, False), allow_16pin_adapter=True)
    assert pen == 0 and any("轉接線" in i for i in issues)


def test_checker_silent_when_no_adapter_needed():
    catalog, _ = _setup()
    for gpu16, native in ((True, True), (False, False), (False, True)):
        pen, issues = CompatibilityChecker().check(_pair(catalog, gpu16, native))
        assert pen == 0 and not any("16-pin" in i for i in issues), (gpu16, native)


# ── GA 修補 ───────────────────────────────────────────────

def _ga(**kw):
    catalog, scorer = _setup()
    kw.setdefault("budget", 80000)
    return GARecommender(catalog, scorer, CompatibilityChecker(), pop_size=10, generations=1, **kw)


def test_repair_swaps_to_native_psu_with_enough_watts():
    catalog, _ = _setup()
    ga = _ga()
    gpu16 = [g for g in catalog.get("GPU") if needs_16pin(g)]
    old_psus = [p for p in catalog.get("電源") if not has_native_16pin(p)]
    natives = [p for p in catalog.get("電源") if has_native_16pin(p)]
    random.seed(0)
    for _ in range(100):
        b = ga._random_build()
        b.parts["GPU"] = random.choice(gpu16)
        old = random.choice(old_psus)
        b.parts["電源"] = old
        ga._repair(b)
        new = b.parts["電源"]
        assert has_native_16pin(new)
        enough = [p for p in natives if _watt(p) >= _watt(old)]
        if enough:
            assert _watt(new) == min(_watt(p) for p in enough), "應該挑瓦數不低於原本的最小一級，不多花錢"


def test_repair_leaves_psu_when_adapter_allowed():
    catalog, _ = _setup()
    ga = _ga(allow_16pin_adapter=True)
    random.seed(1)
    b = ga._random_build()
    b.parts["GPU"] = _first(catalog, "GPU", needs_16pin)
    old = _first(catalog, "電源", lambda p: not has_native_16pin(p))
    b.parts["電源"] = old
    ga._repair(b)
    assert b.parts["電源"] is old


def test_ga_never_recommends_adapter_by_default():
    catalog, scorer = _setup()
    for budget, usage in ((50000, "遊戲"), (80000, "遊戲"), (80000, "工作")):
        random.seed(budget)
        ga = GARecommender(catalog, scorer, CompatibilityChecker(), usage=usage, budget=budget,
                           pop_size=80, generations=40)
        for b in ga.run(verbose=False):
            assert not (needs_16pin(b.parts["GPU"]) and not has_native_16pin(b.parts["電源"]))


# ── 瓦數估算 ───────────────────────────────────────────────

def test_required_watt_uses_gpu_tgp_not_price():
    """同樣的 CPU，換兩張價格同級但 TGP 不同的顯卡，需求瓦數要跟著 TGP 走"""
    catalog, _ = _setup()
    cpu = _first(catalog, "CPU", lambda c: "7700" in c.name)
    gpus = catalog.get("GPU")
    low = min(gpus, key=lambda g: g.specs["tgp_watts"])
    high = max(gpus, key=lambda g: g.specs["tgp_watts"])
    assert required_watt(cpu, high) > required_watt(cpu, low)
    # 把 TGP 拿掉，退回依價格推估（舊做法）
    fake = Part(category="GPU", name="無 TGP 的顯卡", price=high.price, specs={})
    assert _gpu_watt_from_price(high.price) != high.specs["tgp_watts"]
    assert required_watt(cpu, fake) != required_watt(cpu, high)


def test_rtx5080_not_treated_as_500w():
    catalog, _ = _setup()
    cpu = _first(catalog, "CPU", lambda c: "9950X3D" in c.name)
    gpu = _first(catalog, "GPU", lambda g: "RTX5080" in g.name)
    assert gpu.specs["tgp_watts"] == 360
    assert required_watt(cpu, gpu) < 850      # 舊做法依價格當成 500W，floor 直接 850W


def test_cpu_power_uses_actual_value():
    """CPU 改用實際功耗（i7-14700F 標稱 65W、實際 219W）。
    注意：目前公式裡顯卡等級的最低瓦數（450/550/600/750/850W）幾乎永遠比 CPU+顯卡的估算高，
    所以 CPU 功耗在現有目錄裡不會改變需求瓦數；這裡直接檢查估算用的是實際功耗。"""
    import inspect
    from policies import psu_policy
    from data.cpu_power import get_actual_power
    catalog, _ = _setup()
    hot = _first(catalog, "CPU", lambda c: "14700F" in c.name)
    assert get_actual_power(hot.name, fallback_tdp=65) == 219
    assert "get_actual_power(cpu.name" in inspect.getsource(psu_policy._estimate_watt)


def test_gpu_floor_dominates_current_catalog():
    """記錄現況：目錄裡每一種 CPU × 顯卡，需求瓦數都等於顯卡等級的最低瓦數"""
    catalog, _ = _setup()
    floors = {450, 550, 600, 750, 850}
    for cpu in catalog.get("CPU"):
        for gpu in catalog.get("GPU"):
            assert round(required_watt(cpu, gpu)) in floors, (cpu.name, gpu.name, required_watt(cpu, gpu))


# ── 升級建議 ───────────────────────────────────────────────

def test_upgrade_advisor_respects_adapter_setting():
    """原本沒用 16-pin 顯卡、電源也沒有原生接頭：不允許轉接線時，升級建議不能推薦 16-pin 顯卡"""
    catalog, scorer = _setup()
    advisor = UpgradeAdvisor(catalog, scorer, CompatibilityChecker())
    ga = _ga()
    random.seed(2)
    b = ga._random_build()
    b.parts["GPU"] = min((g for g in catalog.get("GPU") if not needs_16pin(g) and g.specs.get("benchmark")),
                         key=lambda g: g.price)
    b.parts["電源"] = max((p for p in catalog.get("電源") if not has_native_16pin(p)), key=_watt)
    ga._repair(b)
    for allow in (False, True):
        recs = advisor.recommend_upgrades(b, 60000, "遊戲", allow_16pin_adapter=allow)
        gpu_recs = [r for r in recs if r["category"] == "GPU"]
        if not allow:
            assert all(not needs_16pin(r["upgrade"]) for r in gpu_recs)


# ── 預算下限也要顧到電源 ─────────────────────────────────────

def test_min_budget_build_has_enough_psu():
    """預算下限那組配置的電源不能被 psu_policy 判成供電不足（原本挑了 350W，需求 450W），
    顯卡要 16-pin 時電源要有原生接頭"""
    from core.feasibility import min_budget
    from policies.psu_policy import psu_score
    catalog, _ = _setup()
    _, parts = min_budget(catalog)
    cpu, gpu, psu = parts["CPU"], parts["GPU"], parts["電源"]
    assert _watt(psu) >= required_watt(cpu, gpu) * 0.95
    assert psu_score(psu, cpu, gpu) > -0.5
    assert not needs_16pin(gpu) or has_native_16pin(psu)
