#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
水冷停用（config.ENABLE_WATER_COOLING = False）的測試。

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_no_water.py -v
"""
import random

import pytest

from config import DB_PATH, ENABLE_WATER_COOLING, CAT_MAP
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer
from advisor.compatibility import CompatibilityChecker
from core.ga_engine import GARecommender

_catalog = _scorer = None


def _setup():
    global _catalog, _scorer
    if _catalog is None:
        _catalog = PartCatalog(DB_PATH)
        _scorer = SentimentScorer()
    return _catalog, _scorer


pytestmark = pytest.mark.skipif(ENABLE_WATER_COOLING, reason="水冷開啟時不適用")


def test_catalog_has_no_water_cooler():
    catalog, _ = _setup()
    assert "WATER_COOLER" not in CAT_MAP
    assert catalog.get("水冷") == []
    assert catalog.get("風冷"), "風冷不能也被關掉"


def test_ga_only_uses_air_cooler():
    """auto 在預算 > 40,000 時原本有 50% 機率先挑水冷，關掉後要一律風冷，而且不能沒有散熱器"""
    catalog, scorer = _setup()
    for budget in (20000, 50000, 150000):
        random.seed(budget)
        ga = GARecommender(catalog, scorer, CompatibilityChecker(), budget=budget,
                           pop_size=30, generations=5)
        assert ga.cooling_available == ["風冷"]
        for b in ga._init_population():
            assert "水冷" not in b.parts and "風冷" in b.parts
        for b in ga.run(verbose=False):
            assert "水冷" not in b.parts and "風冷" in b.parts
            assert ga.fitness(b) > -20          # 沒有因為缺散熱器走硬約束


def test_prefer_water_is_rejected():
    catalog, scorer = _setup()
    with pytest.raises(ValueError):
        GARecommender(catalog, scorer, CompatibilityChecker(), cooling_prefer="水冷")
    GARecommender(catalog, scorer, CompatibilityChecker(), cooling_prefer="風冷")
    GARecommender(catalog, scorer, CompatibilityChecker(), cooling_prefer="auto")
