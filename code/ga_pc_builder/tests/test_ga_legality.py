#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GA 個體合法性測試（審查項目 A）。

驗證：
  1. 交叉不會產生「同時有風冷和水冷」的子代
  2. 修補後：散熱器剛好一個、必選零件齊全、CPU-主機板腳位一致、記憶體世代符合主機板、
     散熱器支援 CPU 腳位、機殼放得下主機板板型
  3. 隨機產生、交叉、突變一路下來的族群，全部都是合法個體
  4. 缺少必選零件時，fitness 走硬約束（< -20）
  5. repair=False 時完全不呼叫修補（保留舊行為）
  6. 使用者指定散熱類型時，修補會保留指定的那個
  7. 記憶體修補沿用 pick_ram 的挑法（相符世代、預算比例內）
  8. 固定 random.seed 時，不同的 Python 雜湊種子跑出相同結果（交叉不受 set 順序影響）

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_ga_legality.py -v
"""
import os
import random
import subprocess
import sys
from pathlib import Path

from config import DB_PATH
from data.catalog import Part, PartCatalog
from data.sentiment import SentimentScorer
from advisor.compatibility import CompatibilityChecker, form_factor_fits
from core.ga_engine import GARecommender, Build, REQUIRED_CATS, COOLING_CATS

_catalog = _scorer = None


def _setup():
    global _catalog, _scorer
    if _catalog is None:
        _catalog = PartCatalog(DB_PATH)
        _scorer = SentimentScorer()
    return _catalog, _scorer


def _ga(budget=50000, usage="遊戲", **kw):
    catalog, scorer = _setup()
    return GARecommender(catalog, scorer, CompatibilityChecker(), usage=usage,
                         budget=budget, pop_size=20, generations=1, **kw)


def _problems(build):
    """回傳這個菜單不合法的地方（空清單＝合法）"""
    out = []
    coolers = [c for c in COOLING_CATS if c in build.parts]
    if len(coolers) != 1:
        out.append(f"散熱器數量 {len(coolers)}")
    for c in REQUIRED_CATS:
        if c not in build.parts:
            out.append(f"缺 {c}")
    cpu, mb, ram = build.parts.get("CPU"), build.parts.get("主機板"), build.parts.get("記憶體")
    if cpu and mb and cpu.specs.get("socket") != mb.specs.get("socket"):
        out.append("CPU/主機板腳位不符")
    if mb and ram and str(ram.specs.get("type")) not in str(mb.specs.get("memory_type")):
        out.append("記憶體世代不符")
    for c in coolers:
        socks = build.parts[c].specs.get("supported_sockets") or []
        if cpu and socks and cpu.specs.get("socket") not in socks:
            out.append("散熱器不支援 CPU 腳位")
    if mb and build.parts.get("機殼") and not form_factor_fits(mb, build.parts["機殼"]):
        out.append("機殼放不下主機板")
    return out


def _water():
    """測試用的水冷（目錄裡的水冷已由 config.ENABLE_WATER_COOLING 關掉，
    但雙散熱器、交叉整組交換這些邏輯仍要測）"""
    return Part(category="水冷", name="測試水冷 360", price=3000,
                specs={"supported_sockets": ["AM4", "AM5", "LGA1700", "LGA1851"], "radiator_size": 360})


def _part(catalog, cat, **spec):
    for p in catalog.get(cat):
        if all(p.specs.get(k) == v for k, v in spec.items()):
            return p
    raise LookupError(f"{cat} 找不到 {spec}")


def test_crossover_never_gives_two_coolers():
    random.seed(0)
    ga = _ga(repair=False)  # 不靠修補，單看交叉本身
    catalog, _ = _setup()
    air = Build({"風冷": catalog.get("風冷")[0]})
    water = Build({"水冷": _water()})
    for _ in range(500):
        c1, c2 = ga._crossover(air, water)
        for c in (c1, c2):
            assert sum(k in c.parts for k in COOLING_CATS) == 1


def test_repair_fixes_every_illegal_case():
    random.seed(1)
    ga = _ga()
    catalog, _ = _setup()
    cpu = _part(catalog, "CPU", socket="AM5")
    bad_mb = _part(catalog, "主機板", socket="LGA1700", memory_type="DDR4")
    ddr4 = _part(catalog, "記憶體", type="DDR4")
    b = Build({
        "CPU": cpu, "主機板": bad_mb, "記憶體": ddr4,
        "風冷": catalog.get("風冷")[0], "水冷": _water(),
        # 故意缺 GPU / SSD / 電源 / 機殼
    })
    ga._repair(b)
    assert _problems(b) == [], _problems(b)
    assert b.parts["CPU"] is cpu  # 以 CPU 為準，CPU 本身不換


def test_population_stays_legal_through_generations():
    random.seed(2)
    ga = _ga(budget=40000)
    pop = ga._init_population()
    for b in pop:
        assert _problems(b) == [], _problems(b)
    for _ in range(5):
        fits = [ga.fitness(b) for b in pop]
        new = []
        while len(new) < len(pop):
            c1, c2 = ga._crossover(ga._tournament(pop, fits), ga._tournament(pop, fits))
            new += [ga._mutate(c1), ga._mutate(c2)]
        pop = new
        for b in pop:
            assert _problems(b) == [], _problems(b)


def test_missing_required_part_is_hard_constraint():
    ga = _ga()
    catalog, _ = _setup()
    b = Build({"CPU": catalog.get("CPU")[0], "風冷": catalog.get("風冷")[0]})
    assert ga.fitness(b) < -20


def test_repair_can_be_turned_off():
    """repair=False 時完全不呼叫修補，行為回到原本只靠相容性懲罰"""
    random.seed(3)
    ga = _ga(repair=False)

    def boom(_b):
        raise AssertionError("repair=False 卻呼叫了 _repair")
    ga._repair = boom
    b = ga._random_build()
    ga._mutate(b)


def test_repair_keeps_preferred_cooler():
    """同時有風冷和水冷時，修補只留一個：指定風冷就留風冷"""
    random.seed(4)
    catalog, _ = _setup()
    for prefer in ("風冷", "auto"):
        ga = _ga(cooling_prefer=prefer)
        for _ in range(20):
            b = Build({"風冷": catalog.get("風冷")[0], "水冷": _water()})
            ga._repair(b)
            kept = [c for c in COOLING_CATS if c in b.parts]
            assert len(kept) == 1
            if prefer == "風冷":
                assert kept == ["風冷"]


def test_ram_repair_uses_pick_ram_rules():
    from policies.ram_policy import ram_feasible
    random.seed(5)
    ga = _ga(budget=50000)
    catalog, _ = _setup()
    cpu = _part(catalog, "CPU", socket="AM5")
    mb = _part(catalog, "主機板", socket="AM5")
    ddr4 = _part(catalog, "記憶體", type="DDR4")
    for _ in range(30):
        b = Build({"CPU": cpu, "主機板": mb, "記憶體": ddr4})
        ga._repair(b)
        ram = b.parts["記憶體"]
        assert ram.specs["type"] in mb.specs["memory_type"]
        assert ram_feasible(ram, ga.budget)


_RUN_ONCE = """
import random, sys
sys.path.insert(0, {root!r})
from config import DB_PATH
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer
from advisor.compatibility import CompatibilityChecker
from core.ga_engine import GARecommender
random.seed(123)
ga = GARecommender(PartCatalog(DB_PATH), SentimentScorer(), CompatibilityChecker(),
                   usage="遊戲", budget=50000, pop_size=40, generations=8)
best = ga.run(verbose=False)[0]
print("RESULT", sorted(p.name for p in best.parts.values()))
"""


def test_same_seed_same_result_across_hash_seeds():
    root = str(Path(__file__).resolve().parent.parent)
    outs = []
    for h in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=h)
        r = subprocess.run([sys.executable, "-c", _RUN_ONCE.format(root=root)],
                           capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
        assert r.returncode == 0, r.stderr
        outs.append([line for line in r.stdout.splitlines() if line.startswith("RESULT")][-1])
    assert outs[0] == outs[1]
