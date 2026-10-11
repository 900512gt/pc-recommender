#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主機板板型 ↔ 機殼相容性測試（審查 #2）。

舊寫法把機殼的 supported_mb（list）轉成字串再做「子字串包含」：
  - "Mini-ITX" in "['ITX']" 永遠不成立 → Mini-ITX 主機板裝哪個機殼都被判不相容
  - "ATX" in "['ITX', 'M-ATX']" 成立（M-ATX 裡有 ATX）→ ATX 主機板配 M-ATX 機殼被判相容

新規則：主機板尺寸 ≤ 機殼最大能裝的尺寸（ITX < M-ATX < ATX < E-ATX）。
機殼最大尺寸取「分類 series」和「supported_mb 最大值」裡比較小的那個，
因為資料庫有 ATX 機殼的 supported_mb 被寫成連 E-ATX 都支援。

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_form_factor.py -v
"""
import random

from config import DB_PATH
from data.catalog import Part, PartCatalog
from data.sentiment import SentimentScorer
from advisor.compatibility import (CompatibilityChecker, form_factor_fits, case_max_form_factor,
                                   normalize_form_factor, supported_form_factors, FORM_SIZE_ORDER)
from core.ga_engine import GARecommender, Build

_catalog = _scorer = None

SIZES = ["ITX", "M-ATX", "ATX", "E-ATX"]   # 由小到大


def _setup():
    global _catalog, _scorer
    if _catalog is None:
        _catalog = PartCatalog(DB_PATH)
        _scorer = SentimentScorer()
    return _catalog, _scorer


def _mb(form):
    return Part(category="主機板", name=f"測試主機板 {form}", price=1, specs={"form_factor": form})


def _case(supported, series=None):
    specs = {"supported_mb": supported}
    if series is not None:
        specs["series"] = series
    return Part(category="機殼", name=f"測試機殼 {series} {supported}", price=1, specs=specs)


def _form_issue(mb, case):
    _, issues = CompatibilityChecker().check(Build({"主機板": mb, "機殼": case}))
    return any("機殼不支援" in i for i in issues)


def test_aliases_normalize_to_same_name():
    assert normalize_form_factor("Mini-ITX") == "ITX"
    assert normalize_form_factor(" mini itx ") == "ITX"
    assert normalize_form_factor("Micro-ATX") == "M-ATX"
    assert normalize_form_factor("mATX") == "M-ATX"
    assert normalize_form_factor("EATX") == "E-ATX"
    assert normalize_form_factor("ATX") == "ATX"
    assert FORM_SIZE_ORDER == SIZES


def test_mini_itx_fits_itx_case():
    """舊寫法的錯誤一：Mini-ITX 永遠被判不相容"""
    assert form_factor_fits(_mb("Mini-ITX"), _case(["ITX"]))
    assert not _form_issue(_mb("Mini-ITX"), _case(["ITX"]))


def test_atx_does_not_fit_matx_case():
    """舊寫法的錯誤二：M-ATX 裡有 ATX 三個字，ATX 主機板被判放得進 M-ATX 機殼"""
    assert not form_factor_fits(_mb("ATX"), _case(["ITX", "M-ATX"]))
    assert _form_issue(_mb("ATX"), _case(["ITX", "M-ATX"]))


def test_every_size_combination():
    for i, form in enumerate(SIZES):
        for j in range(len(SIZES)):
            supported = SIZES[: j + 1]          # 機殼支援到第 j 級
            expected = i <= j
            mb_form = "Mini-ITX" if form == "ITX" else form
            assert form_factor_fits(_mb(mb_form), _case(supported)) == expected, (mb_form, supported)
            assert _form_issue(_mb(mb_form), _case(supported)) == (not expected), (mb_form, supported)


def test_supported_mb_in_other_formats():
    assert supported_form_factors(_case('["ITX", "M-ATX"]')) == {"ITX", "M-ATX"}
    assert supported_form_factors(_case("ITX, M-ATX")) == {"ITX", "M-ATX"}
    assert supported_form_factors(_case("ITX/M-ATX/ATX")) == {"ITX", "M-ATX", "ATX"}
    assert supported_form_factors(_case("ATX")) == {"ATX"}
    assert not form_factor_fits(_mb("ATX"), _case('["ITX", "M-ATX"]'))
    # 奇怪的型別不能讓程式當掉
    assert supported_form_factors(_case(None)) == set()
    assert supported_form_factors(_case(123)) == {"123"}
    assert form_factor_fits(_mb("ATX"), _case(None))


def test_missing_data_is_not_penalized():
    assert form_factor_fits(_mb(""), _case(["ITX"]))
    assert form_factor_fits(_mb("ATX"), _case([]))
    assert form_factor_fits(_mb("ATX"), Part(category="機殼", name="無資料", price=1, specs={}))


def test_smaller_board_fits_bigger_case():
    """ATX 主機板放 E-ATX 機殼可以，反過來不行；機殼清單漏列小尺寸也照尺寸大小判斷"""
    assert form_factor_fits(_mb("ATX"), _case(SIZES, series="E-ATX"))
    assert not form_factor_fits(_mb("E-ATX"), _case(["ITX", "M-ATX", "ATX"], series="ATX"))
    assert form_factor_fits(_mb("Mini-ITX"), _case(["M-ATX"]))     # 清單只寫 M-ATX，ITX 一樣放得下


def test_atx_case_wrongly_listing_eatx_is_capped_by_series():
    """資料庫的錯誤：ATX 機殼的 supported_mb 寫了四種。以分類為上限，E-ATX 主機板放不進去"""
    case = _case(SIZES, series="ATX")
    assert case_max_form_factor(case) == "ATX"
    assert not form_factor_fits(_mb("E-ATX"), case)
    assert _form_issue(_mb("E-ATX"), case)
    assert form_factor_fits(_mb("ATX"), case)
    # 分類不是尺寸（例如「靜音」）時只看 supported_mb
    assert case_max_form_factor(_case(SIZES, series="靜音")) == "E-ATX"
    # 分類比清單大時也取小的
    assert case_max_form_factor(_case(["ITX", "M-ATX"], series="ATX")) == "M-ATX"


def test_unknown_board_size_uses_case_list():
    assert form_factor_fits(_mb("SSI-EEB"), _case(["SSI-EEB", "E-ATX"]))
    assert not form_factor_fits(_mb("SSI-EEB"), _case(SIZES))
    assert form_factor_fits(_mb("SSI-EEB"), _case([]))


def test_real_catalog_matches_rule():
    """真實資料的每一種 主機板 × 機殼 組合，都要符合「主機板尺寸 ≤ 機殼上限」"""
    catalog, _ = _setup()
    rank = {f: i for i, f in enumerate(SIZES)}
    seen_forms = set()
    eatx_into_atx_case = 0
    for mb in catalog.get("主機板"):
        form = normalize_form_factor(mb.specs.get("form_factor"))
        seen_forms.add(form)
        for case in catalog.get("機殼"):
            listed_max = max((normalize_form_factor(x) for x in case.specs.get("supported_mb", [])), key=rank.get)
            series = normalize_form_factor(case.specs.get("series"))
            cap = min(listed_max, series, key=rank.get) if series in rank else listed_max
            assert form_factor_fits(mb, case) == (rank[form] <= rank[cap]), (mb.name, case.name)
            if form == "E-ATX" and series == "ATX":
                assert not form_factor_fits(mb, case)
                eatx_into_atx_case += 1
    assert seen_forms <= set(SIZES), f"資料庫出現認不得的板型：{seen_forms - set(SIZES)}"
    assert eatx_into_atx_case > 0   # 資料裡確實有這種組合，而且都被擋下


def test_repair_swaps_case_to_one_that_fits():
    catalog, scorer = _setup()
    random.seed(0)
    ga = GARecommender(catalog, scorer, CompatibilityChecker(), budget=50000, pop_size=10, generations=1)
    atx = [m for m in catalog.get("主機板") if m.specs.get("form_factor") == "ATX"]
    small_cases = [c for c in catalog.get("機殼") if not form_factor_fits(atx[0], c)]
    assert small_cases, "資料庫裡找不到放不下 ATX 的機殼"
    for _ in range(50):
        b = ga._random_build()
        cpu_sock = b.parts["CPU"].specs["socket"]
        mb = next((m for m in atx if m.specs["socket"] == cpu_sock), None)
        if mb is None:
            continue
        b.parts["主機板"] = mb
        b.parts["機殼"] = random.choice(small_cases)
        ga._repair(b)
        assert b.parts["主機板"] is mb                 # 主機板跟著 CPU，不換
        assert form_factor_fits(mb, b.parts["機殼"])
        _, issues = ga.checker.check(b)
        assert not any("機殼不支援" in i for i in issues)


def _ga(budget=50000, **kw):
    catalog, scorer = _setup()
    return GARecommender(catalog, scorer, CompatibilityChecker(), budget=budget,
                         pop_size=10, generations=1, **kw)


def test_case_repair_prefers_case_that_fits_radiator():
    """換機殼時，有水冷就優先挑水冷排放得下的機殼（水冷目前關掉，用測試用的 360 水冷驗證邏輯）"""
    catalog, _ = _setup()
    random.seed(1)
    ga = _ga()
    water = Part(category="水冷", name="測試水冷 360", price=3000,
                 specs={"supported_sockets": ["AM4", "AM5", "LGA1700", "LGA1851"], "radiator_size": 360})
    itx_only = [c for c in catalog.get("機殼") if supported_form_factors(c) == {"ITX"}]
    assert itx_only
    for _ in range(50):
        b = ga._random_build()
        if b.parts["主機板"].specs.get("form_factor") == "Mini-ITX":
            continue
        b.parts.pop("風冷", None)
        b.parts["水冷"] = water
        b.parts["機殼"] = random.choice(itx_only)
        ga._repair(b)
        case = b.parts["機殼"]
        assert form_factor_fits(b.parts["主機板"], case)
        max_rad = float(case.specs.get("max_water_mm") or 0)   # 0 = 沒資料，跟相容性檢查一樣視為放得下
        assert max_rad == 0 or max_rad >= float(b.parts["水冷"].specs["radiator_size"])


def test_mb_repair_prefers_board_that_fits_current_case():
    """CPU 換腳位要換主機板時，優先挑放得進目前機殼的板子，機殼就不用跟著換"""
    catalog, _ = _setup()
    random.seed(2)
    ga = _ga()
    cpu = next(p for p in catalog.get("CPU") if p.specs.get("socket") == "AM5")
    wrong_mb = next(p for p in catalog.get("主機板") if p.specs.get("socket") == "LGA1700"
                    and p.specs.get("memory_type") == "DDR5")
    ddr5 = next(p for p in catalog.get("記憶體") if p.specs.get("type") == "DDR5")
    matx_case = next(c for c in catalog.get("機殼") if supported_form_factors(c) == {"ITX", "M-ATX"})
    for _ in range(30):
        b = ga._random_build()
        b.parts.update({"CPU": cpu, "主機板": wrong_mb, "記憶體": ddr5, "機殼": matx_case})
        ga._repair(b)
        assert b.parts["主機板"].specs["socket"] == "AM5"
        assert b.parts["機殼"] is matx_case
        assert form_factor_fits(b.parts["主機板"], matx_case)
