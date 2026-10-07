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
