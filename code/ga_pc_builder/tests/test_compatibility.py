#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
升級建議的相容性把關測試。

重點是驗證「換進來的零件會拿去重算整份配置的相容性」這件事真的有生效：
Intel 腳位的配置不能被推薦 AMD 主機板，350W 電源的配置不能被推薦旗艦顯卡。

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_compatibility.py -v
"""

from config import DB_PATH, MATCHED_FILES
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer
from advisor.upgrade_advisor import UpgradeAdvisor
from core.ga_engine import Build
from advisor.compatibility import CompatibilityChecker

_catalog = _scorer = _advisor = None


def _setup():
    """三份資料載入都不便宜，整個測試檔共用一次。"""
    global _catalog, _scorer, _advisor
    if _advisor is None:
        _catalog = PartCatalog(DB_PATH)
        _scorer = SentimentScorer(MATCHED_FILES, DB_PATH)
        _advisor = UpgradeAdvisor(_catalog, _scorer, CompatibilityChecker())
    return _catalog, _advisor


def _cheapest(catalog, category):
    return min(catalog.get(category), key=lambda p: p.price)


def _build_with(catalog, **overrides):
    parts = {
        cat: _cheapest(catalog, cat)
        for cat in ("CPU", "GPU", "記憶體", "主機板", "SSD", "機殼", "電源", "風冷")
    }
    parts.update(overrides)
    return Build(parts=parts)


def test_intel_build_is_not_offered_amd_motherboard():
    catalog, advisor = _setup()
    # 資料庫的腳位值是 AM5 / AM4 / LGA1700 / LGA1851
    intel_cpus = [p for p in catalog.get("CPU") if str(p.specs.get("socket", "")).startswith("LGA")]
    assert intel_cpus, "資料庫裡找不到 Intel 腳位的 CPU"
    cpu = intel_cpus[0]
    socket = str(cpu.specs.get("socket", ""))

    mbs = [p for p in catalog.get("主機板") if str(p.specs.get("socket", "")) == socket]
    assert mbs, f"資料庫裡找不到 {socket} 腳位的主機板"

    build = _build_with(catalog, CPU=cpu, 主機板=min(mbs, key=lambda p: p.price))
    recs = advisor.recommend_upgrades(build, 90000 - build.total_price, "工作")

    for rec in recs:
        if rec["category"] == "主機板":
            assert str(rec["upgrade"].specs.get("socket", "")) == socket, (
                f"推薦了腳位不符的主機板：{rec['upgrade'].name}"
            )


def test_weak_psu_blocks_flagship_gpu():
    """350W 電源不該被推薦吃電怪獸。這條在修好『電源不足固定罰 0.5』之前會失敗
    ——當時升級前後的懲罰一樣重，看不出換高階顯卡讓供電更吃緊。"""
    catalog, advisor = _setup()
    weak_psu = min(catalog.get("電源"), key=lambda p: float(p.specs.get("wattage", 0) or 0))
    build = _build_with(catalog, 電源=weak_psu)

    recs = advisor.recommend_upgrades(build, 90000 - build.total_price, "遊戲")
    gpu_rec = next((r for r in recs if r["category"] == "GPU"), None)

    if gpu_rec is not None:
        psu_watt = float(weak_psu.specs.get("wattage", 0) or 0)
        tgp = float(gpu_rec["upgrade"].specs.get("tgp_watts", 0) or 0)
        assert tgp < psu_watt, (
            f"{psu_watt:.0f}W 電源被推薦了 TGP {tgp:.0f}W 的 {gpu_rec['upgrade'].name}"
        )


def test_recommendations_fit_remaining_budget():
    """整份清單是「可以一起買」的，所以累計金額不能超過剩餘預算。"""
    catalog, advisor = _setup()
    build = _build_with(catalog)
    remaining = 50000 - build.total_price

    recs = advisor.recommend_upgrades(build, remaining, "遊戲")
    assert sum(r["cost"] for r in recs) <= remaining


def test_higher_tier_spends_more_and_stays_within_budget():
    """加價級距的兩個基本性質：花得越多換得越好，且每一級都不超出該級可動用的錢。"""
    catalog, advisor = _setup()
    build = _build_with(catalog)
    budget = 60000
    remaining = budget - build.total_price

    tiers = advisor.recommend_tiers(build, budget, remaining, "遊戲")
    assert tiers, "這個配置應該要有升級空間"

    for tier in tiers:
        assert tier["spent"] <= tier["available"], (
            f"加價 {tier['extra_ratio']:.0%} 這級花了 {tier['spent']} 超過可動用的 {tier['available']}"
        )

    spends = [t["spent"] for t in tiers]
    assert spends == sorted(spends), f"加價越多花費應該越高，實際是 {spends}"


def test_no_invented_performance_numbers():
    """沒有 PassMark 跑分的類別不准回報效能提升百分比。"""
    catalog, advisor = _setup()
    build = _build_with(catalog)
    recs = advisor.recommend_upgrades(build, 90000 - build.total_price, "遊戲")

    for rec in recs:
        has_benchmark = rec["upgrade"].specs.get("benchmark")
        if not has_benchmark:
            assert rec["benchmark_gain_pct"] is None, (
                f"{rec['category']} 沒有跑分資料卻回報了 {rec['benchmark_gain_pct']}%"
            )
