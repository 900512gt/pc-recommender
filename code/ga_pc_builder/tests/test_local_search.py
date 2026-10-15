#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
區域搜尋與最後爬山的測試。

40,000 遊戲有一組 31,749 元的配置（R5 5600XT + RX7650GRE），任換一個或任換兩個零件
分數都不會更高，要 CPU、主機板、顯卡同時換對才會（遊戲用途 CPU／顯卡平衡規則是門檻式扣分）。
原本 12 個種子有 10 個停在這裡；加上「CPU 型號 × 顯卡型號」區域搜尋後全部跳出。

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_local_search.py -v
"""
import random

from config import DB_PATH
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer
from advisor.compatibility import CompatibilityChecker
from core.ga_engine import GARecommender, Build

_catalog = _scorer = None

# 40,000 遊戲卡住的那組配置
STUCK = {
    "CPU": "AMD R5 5600XT MPK (含風扇)【6核/12緒】3.7G(↑4.7G)65W /32M /7nm代理商三年保",
    "GPU": "藍寶石 黑鑽版 RX7650GRE GAMING OC 8GB(2810MHz/30cm/三風扇/三年保)*最後優惠",
    "記憶體": "UMAX 32GB DDR4-3200原生顆粒製(2048*8)超頻CL16",
    "主機板": "微星 B550M-A PRO(M-ATX/1D1H/LAN 1Gb/註四年)4+2相供電*尾盤",
    "SSD": "致態 ZhiTai TiPlus7100s 1TB/Gen4/PCIe 4.0/讀7400/寫6900/TLC【五年保】",
    "電源": "全漢 VITA GM 650W(MIT) 白色版 雙8/金牌/全模/ATX3.1/主日系/10年保",
    "機殼": "Montech X1 黑 顯卡長30.5/CPU高16.5/網孔面板/玻璃透側/ATX",
    "風冷": "利民 Peerless Assassin 120 MINI 白化版/6導管/雙塔單扇/高13.5cm【WXZ】",
}


def _ga(budget=40000, **kw):
    global _catalog, _scorer
    if _catalog is None:
        _catalog = PartCatalog(DB_PATH)
        _scorer = SentimentScorer()
    return GARecommender(_catalog, _scorer, CompatibilityChecker(), usage="遊戲", budget=budget,
                         pop_size=60, generations=30, **kw)


def _stuck(ga):
    b = Build()
    for cat, name in STUCK.items():
        b.parts[cat] = next(p for p in ga.catalog.get(cat) if p.name == name)
    return b


def _trapped(ga):
    """卡住的配置先做單一零件爬山，推到目前口碑資料下的局部最佳。
    口碑檔重建後小零件（主機板、散熱器）的最佳款會變，所以不直接用寫死的那組當局部最佳"""
    random.seed(0)
    return ga._polish(_stuck(ga))


def test_single_part_moves_stay_trapped():
    """前提：只換單一零件（以及顯卡＋電源成對）怎麼爬都離不開 5600XT + RX7650GRE"""
    ga = _ga()
    b = _trapped(ga)
    assert b.parts["CPU"].specs.get("ptt_model") == "AMD R5 5600XT"
    assert b.parts["GPU"].specs.get("ptt_model") == "RX7650GRE"


def test_local_search_escapes_stuck_build():
    ga = _ga()
    b = _trapped(ga)
    random.seed(0)
    better = ga._local_search(b)
    assert better is not None
    assert ga.fitness(better) > ga.fitness(b) + 0.03
    assert better.parts["GPU"].specs.get("ptt_model") != "RX7650GRE"


def test_model_reps_one_per_model_and_cheapest():
    ga = _ga()
    reps = ga._model_reps("GPU")
    models = [p.specs.get("ptt_model") for p in reps]
    assert len(models) == len(set(models))
    for p in reps:
        same = [q.price for q in ga.catalog.get("GPU") if q.specs.get("ptt_model") == p.specs.get("ptt_model")]
        assert p.price == min(same)


def test_polish_never_worse():
    ga = _ga()
    random.seed(1)
    for _ in range(3):
        b = ga._random_build()
        assert ga.fitness(ga._polish(b)) >= ga.fitness(b) - 1e-9


def test_can_disable():
    """local_search_every=0、polish=False 時不做區域搜尋與爬山（對照實驗用）"""
    ga = _ga(local_search_every=0, polish=False)
    called = []
    ga._local_search = lambda b: called.append("ls")
    ga._polish = lambda b: called.append("polish") or b
    random.seed(0)
    ga.run(verbose=False)
    assert called == []
