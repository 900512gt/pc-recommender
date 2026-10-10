#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
面向口碑分數的測試。

用一份手寫的小口碑檔，驗證三件事：
  類別內正規化（該面向最好的型號 1、最差的 0，上下界只看有資料的型號）
  使用者指定面向時，指定的面向佔一半、其餘面向的平均佔一半
  沒有評論的零件拿類別平均，資料不夠的面向不開放指定

執行（在 code/ga_pc_builder/ 底下）：
  python -m pytest tests/test_sentiment.py -v
"""

import json

import pytest

from data.sentiment import SentimentScorer


def _aspect(pos, neg, score):
    return {"正面": pos, "負面": neg, "raw": None, "score": score}


@pytest.fixture
def scorer(tmp_path):
    data = {
        "_prior": {"GPU": {"score": 0.5, "aspects": {"效能": 0.7, "穩定": 0.3, "噪音": 0.5}}},
        "GPU|甲": {"score": 0, "review_count": 100, "aspects": {
            "效能": _aspect(80, 20, 0.8), "穩定": _aspect(10, 40, 0.2), "噪音": _aspect(1, 0, 0.9)}},
        "GPU|乙": {"score": 0, "review_count": 100, "aspects": {
            "效能": _aspect(60, 40, 0.6), "穩定": _aspect(20, 30, 0.4), "噪音": _aspect(0, 0, 0.5)}},
        # 效能只有 3 則：不能拿來定上下界，也不用它自己的分數，改用類別平均的位置
        "GPU|丙": {"score": 0, "review_count": 5, "aspects": {
            "效能": _aspect(3, 0, 0.95), "穩定": _aspect(15, 35, 0.3), "噪音": _aspect(0, 0, 0.5)}},
    }
    path = tmp_path / "part_aspect_sentiment.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return SentimentScorer(sentiment_path=path)


def test_normalizes_within_category(scorer):
    assert scorer.get_aspects("GPU", "甲")["效能"] == pytest.approx(1.0)
    assert scorer.get_aspects("GPU", "乙")["效能"] == pytest.approx(0.0)
    assert scorer.get_aspects("GPU", "甲")["穩定"] == pytest.approx(0.0)
    assert scorer.get_aspects("GPU", "丙")["穩定"] == pytest.approx(0.5)


def test_aspect_with_few_mentions_uses_category_average(scorer):
    # 丙的效能 0.95 只有 3 則：上界仍是甲的 0.8，丙拿類別平均 0.7 的位置 (0.7-0.6)/0.2
    assert scorer.get_aspects("GPU", "甲")["效能"] == pytest.approx(1.0)
    assert scorer.get_aspects("GPU", "丙")["效能"] == pytest.approx(0.5)


def test_aspect_nobody_has_data_for_is_neutral(scorer):
    for model in ("甲", "乙", "丙"):
        assert scorer.get_aspects("GPU", model)["噪音"] == pytest.approx(0.5)


def test_score_is_mean_of_normalized_aspects(scorer):
    assert scorer.get("GPU", "甲") == pytest.approx((1.0 + 0.0 + 0.5) / 3)


def test_selected_aspect_takes_half_and_the_rest_share_the_other_half(scorer):
    # 甲：效能 1.0、穩定 0.0、噪音 0.5
    assert scorer.get("GPU", "甲", aspects=["穩定"]) == pytest.approx(0.5 * 0.0 + 0.5 * (1.0 + 0.5) / 2)
    # 乙：效能 0.0、穩定 1.0、噪音 0.5
    assert scorer.get("GPU", "乙", aspects=["穩定"]) == pytest.approx(0.5 * 1.0 + 0.5 * (0.0 + 0.5) / 2)
    assert scorer.get("GPU", "甲", aspects=["效能", "穩定"]) == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)


def test_selecting_an_aspect_moves_the_score_toward_it(scorer):
    # 甲的穩定最差、乙最好：指定穩定後甲的口碑下降、乙上升
    assert scorer.get("GPU", "甲", aspects=["穩定"]) < scorer.get("GPU", "甲")
    assert scorer.get("GPU", "乙", aspects=["穩定"]) > scorer.get("GPU", "乙")


def test_selecting_every_aspect_equals_no_selection(scorer):
    assert scorer.get("GPU", "甲", aspects=["效能", "穩定", "噪音"]) == pytest.approx(scorer.get("GPU", "甲"))


def test_unknown_aspect_falls_back_to_overall_score(scorer):
    assert scorer.get("GPU", "甲", aspects=["不存在"]) == scorer.get("GPU", "甲")


def test_part_without_reviews_gets_category_average(scorer):
    # 類別平均也照同一組上下界正規化：效能 (0.7-0.6)/0.2、穩定 (0.3-0.2)/0.2、噪音 0.5
    assert scorer.get("GPU", "沒人討論的卡") == pytest.approx(0.5)
    assert scorer.get("GPU", "沒人討論的卡", aspects=["效能"]) == pytest.approx(0.5)
    assert scorer.get("不存在的類別", "x") == 0.5


def test_only_aspects_with_enough_data_are_selectable(scorer):
    assert scorer.selectable["GPU"] == ["效能", "穩定"]


def test_evidence_ranks_only_aspects_with_enough_mentions(scorer):
    rows = {r["aspect"]: r for r in scorer.aspect_evidence("GPU", "甲")["aspects"]}
    assert (rows["效能"]["positive"], rows["效能"]["negative"]) == (80, 20)
    assert (rows["效能"]["rank"], rows["效能"]["ranked_total"]) == (1, 2)
    assert rows["穩定"]["rank"] == 3            # 0.2 是三個型號裡最低的
    assert rows["效能"]["category_avg"] == pytest.approx(0.7)
    # 噪音只有 1 則：照實給則數，但不排名次
    assert rows["噪音"]["rank"] is None

    # 丙的效能分數 0.95 是全場最高，但只有 3 則，不能排到第一名
    assert {r["aspect"]: r["rank"] for r in scorer.aspect_evidence("GPU", "丙")["aspects"]}["效能"] is None


def test_evidence_is_none_without_reviews(scorer):
    assert scorer.aspect_evidence("GPU", "沒評論的型號") is None


# ── 指定面向在 GA fitness 裡獨立成一項 ──

class _EmptyCatalog:
    def get(self, category):
        return []


def _ga(scorer, prefs, **kwargs):
    from core.ga_engine import GARecommender
    return GARecommender(catalog=_EmptyCatalog(), scorer=scorer, checker=None,
                         usage="遊戲", aspect_prefs=prefs, **kwargs)


def _gpu_build(model):
    from core.ga_engine import Build
    from data.catalog import Part
    return Build(parts={"GPU": Part("GPU", model, 10000)})


def test_pref_term_scores_only_the_picked_aspect(scorer):
    ga = _ga(scorer, {"GPU": ["穩定"]})
    assert ga._pref_score(_gpu_build("乙")) == pytest.approx(1.0)   # 穩定最好
    assert ga._pref_score(_gpu_build("甲")) == pytest.approx(0.0)   # 穩定最差
    # 指定的面向已經獨立成項，一般口碑不再重複加重它
    assert ga._sentiment_score(_gpu_build("乙")) == pytest.approx(scorer.get("GPU", "乙"))


def test_pref_term_share_zero_keeps_old_blend(scorer):
    ga = _ga(scorer, {"GPU": ["穩定"]}, pref_term_share=0)
    assert ga._pref_score(_gpu_build("乙")) == 0.0
    assert ga._sentiment_score(_gpu_build("乙")) == pytest.approx(
        scorer.get("GPU", "乙", aspects=["穩定"]))


def test_pref_term_is_off_when_nothing_usable_is_picked(scorer):
    # 噪音的資料不夠、不開放指定：整個偏好被忽略，fitness 與沒指定時相同
    ga = _ga(scorer, {"GPU": ["噪音"]})
    assert ga.aspect_prefs == {}
    assert ga.pref_term_share == 0


def test_pref_share_one_looks_only_at_picked_aspects(scorer):
    ga = _ga(scorer, {"GPU": ["穩定"]}, pref_term_share=0, pref_share=1)
    # 有指定面向的類別，口碑就是指定面向的分數；其餘面向不看
    assert ga._sentiment_score(_gpu_build("乙")) == pytest.approx(1.0)
    assert ga._sentiment_score(_gpu_build("甲")) == pytest.approx(0.0)


def test_common_scale_keeps_narrow_aspects_close(tmp_path):
    # 效能全距 0.4、穩定全距只有 0.1：共用一把尺時，穩定最好與最差只差 0.25，不會被拉成 0 與 1
    data = {
        "_prior": {"GPU": {"score": 0.5, "aspects": {"效能": 0.7, "穩定": 0.25}}},
        "GPU|甲": {"score": 0, "review_count": 100, "aspects": {
            "效能": _aspect(80, 20, 0.9), "穩定": _aspect(10, 40, 0.2)}},
        "GPU|乙": {"score": 0, "review_count": 100, "aspects": {
            "效能": _aspect(50, 50, 0.5), "穩定": _aspect(15, 35, 0.3)}},
    }
    path = tmp_path / "part_aspect_sentiment.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    scorer = SentimentScorer(sentiment_path=path, common_scale=True)

    assert scorer.get_aspects("GPU", "甲")["效能"] == pytest.approx(1.0)
    assert scorer.get_aspects("GPU", "乙")["效能"] == pytest.approx(0.0)
    assert scorer.get_aspects("GPU", "甲")["穩定"] == pytest.approx(0.375)
    assert scorer.get_aspects("GPU", "乙")["穩定"] == pytest.approx(0.625)
