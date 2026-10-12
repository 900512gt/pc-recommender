#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
前五名測試（審查 #5）。

原本直接從最後一代挑前五名；族群收斂後最後一代常只剩一兩種配置，
80,000 遊戲實測前五名是同一組配置重複 5 次。改成記錄整個演化過程的好配置（名人堂）再挑。

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_top5.py -v
"""
import random

from config import DB_PATH
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer
from advisor.compatibility import CompatibilityChecker
from core.ga_engine import GARecommender, HARD_FAIL

_catalog = _scorer = None


def _setup():
    global _catalog, _scorer
    if _catalog is None:
        _catalog = PartCatalog(DB_PATH)
        _scorer = SentimentScorer()
    return _catalog, _scorer


def _run(budget, usage="遊戲", seed=0, pop=80, gens=60):
    catalog, scorer = _setup()
    random.seed(seed)
    ga = GARecommender(catalog, scorer, CompatibilityChecker(), usage=usage, budget=budget,
                       pop_size=pop, generations=gens)
    return ga, ga.run(verbose=False)


def _sig(b):
    return tuple(sorted((c, p.name) for c, p in b.parts.items()))


def test_top5_are_distinct_and_sorted():
    for budget, usage in ((25000, "一般文書"), (50000, "遊戲"), (80000, "遊戲"), (150000, "工作")):
        ga, top = _run(budget, usage)
        assert len(top) == 5
        assert len({_sig(b) for b in top}) == 5, f"{budget} {usage} 前五名有重複的配置"
        fits = [ga.fitness(b) for b in top]
        assert fits == sorted(fits, reverse=True), "名次要照 fitness 排"


def test_top1_is_the_best_ever_seen():
    ga, top = _run(50000)
    best_in_hall = max(f for f, _ in ga._hall.values())
    assert ga.fitness(top[0]) == best_in_hall


def test_no_hard_constraint_failures_when_feasible_exists():
    """低預算時名人堂裡有很多超支的配置，不能為了湊不同顯卡把它們挑進來"""
    ga, top = _run(15000)
    assert all(ga.fitness(b) > HARD_FAIL for b in top)


def test_falls_back_when_nothing_is_feasible():
    """預算低於最便宜的組合時全部都超支，仍要回傳結果（API 拿不到結果會出錯）"""
    ga, top = _run(10000, "一般文書", gens=20)
    assert top
    assert len({_sig(b) for b in top}) == len(top)


def test_tiny_runs_do_not_crash():
    for pop, gens in ((1, 0), (2, 1), (7, 3)):
        ga, top = _run(50000, pop=pop, gens=gens)
        assert 1 <= len(top) <= 5
        assert len({_sig(b) for b in top}) == len(top)


def test_pick_top_before_run_returns_empty():
    catalog, scorer = _setup()
    ga = GARecommender(catalog, scorer, CompatibilityChecker(), budget=50000, pop_size=2, generations=0)
    assert ga._pick_top() == []
