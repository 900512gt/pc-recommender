"""
情感分數計算器 v2 — 使用 BERT 面向級情感分析

與舊版差異：
  舊版：統計正/負/中立標籤 + GP/BP 加權 → 單一籠統分數
  新版：BERT 分析評論內容 → 五大面向分數（效能/溫控/噪音/保固/CP值）
        → 依使用情境加權 → 情感分數

資料來源：
  part_sentiment.json — 由 fine-tune 後的中研院 BERT 跑全部 50,593 則評論產生
  每個分數背後都有「幾則正面、幾則負面」的統計依據（counts 欄位）

相容性：
  保留 get(category, model) 介面，既有程式（api.py、ga_engine.py）無需修改。
  新增 get_dimensions() 可取得面向明細，供進階使用。
"""
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional

from config import CAT_MAP

DIMENSIONS = ["效能", "溫控", "噪音", "保固", "CP值"]

USAGE_DIM_WEIGHTS = {
    "遊戲":     {"效能": 0.35, "溫控": 0.20, "噪音": 0.15, "保固": 0.10, "CP值": 0.20},
    "工作":     {"效能": 0.30, "溫控": 0.25, "噪音": 0.10, "保固": 0.20, "CP值": 0.15},
    "一般文書": {"效能": 0.10, "溫控": 0.15, "噪音": 0.20, "保固": 0.25, "CP值": 0.30},
}

MIN_SAMPLES = 10


class SentimentScorer:
    """
    情感分數計算器（BERT 面向版）

    介面與舊版相容：
      scorer.get(category, model) -> float [0,1]

    新增介面：
      scorer.get_dimensions(category, model) -> dict 面向明細
      scorer.set_usage(usage) -> 切換使用情境（影響加權）
    """

    def __init__(self, jsonl_paths: list = None,
                 db_path: Optional[Path] = None,
                 sentiment_path: Optional[Path] = None,
                 usage: str = "遊戲"):
        self.usage = usage if usage in USAGE_DIM_WEIGHTS else "遊戲"

        self.dimensions: dict = {}
        self.sample_counts: dict = {}
        self.scores: dict = {}
        self.counts: dict = {}

        if sentiment_path is None:
            sentiment_path = Path(__file__).parent / "part_sentiment.json"

        self._load_bert_scores(sentiment_path)

        if not self.dimensions:
            print("[SentimentScorer] 無 BERT 面向分數，回退至舊版標籤統計")
            self._load_legacy(jsonl_paths, db_path)

    def _load_bert_scores(self, path):
        if not Path(path).exists():
            print(f"[SentimentScorer] 找不到 {path}")
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for key_str, item in data.items():
            if "|" not in key_str:
                continue
            category, model = key_str.split("|", 1)
            key = (category, model)

            dims = item.get("dimensions", {})
            counts = item.get("counts", {})

            filtered_dims = {}
            sample_n = {}
            for dim in DIMENSIONS:
                c = counts.get(dim, {})
                n = int(c.get("正", 0)) + int(c.get("负", 0))
                sample_n[dim] = n
                if n >= MIN_SAMPLES:
                    filtered_dims[dim] = float(dims.get(dim, 0.5))
                else:
                    filtered_dims[dim] = 0.5

            self.dimensions[key] = filtered_dims
            self.sample_counts[key] = sample_n
            self.counts[key] = {
                "review_count": item.get("review_count", 0),
                "samples": sample_n,
            }

        self._recompute_scores()
        print(f"[SentimentScorer] 載入 BERT 面向分數：{len(self.dimensions)} 個零件"
              f"（情境：{self.usage}）")

    def _recompute_scores(self):
        weights = USAGE_DIM_WEIGHTS[self.usage]
        self.scores = {}
        for key, dims in self.dimensions.items():
            total = sum(weights[d] * dims.get(d, 0.5) for d in DIMENSIONS)
            self.scores[key] = round(total, 4)

    def set_usage(self, usage: str):
        if usage in USAGE_DIM_WEIGHTS and usage != self.usage:
            self.usage = usage
            self._recompute_scores()

    def _load_legacy(self, paths, db_path):
        if not paths:
            return
        raw = defaultdict(lambda: {
            "positive": 0, "neutral": 0, "negative": 0,
            "gp_sum": 0, "bp_sum": 0, "total": 0
        })
        for path in paths:
            if not Path(path).exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (d.get("category", ""), d.get("model", ""))
                    label = d.get("label", "中立")
                    r = raw[key]
                    if label == "正評":
                        r["positive"] += 1
                    elif label == "負評":
                        r["negative"] += 1
                    else:
                        r["neutral"] += 1
                    r["gp_sum"] += int(d.get("GP", 0) or 0)
                    r["bp_sum"] += int(d.get("BP", 0) or 0)
                    r["total"] += 1

        for key, r in raw.items():
            total = max(r["total"], 1)
            score = (r["positive"] - r["negative"]
                     + (r["gp_sum"] - r["bp_sum"]) / 10.0) / total
            self.scores[key] = 1.0 / (1.0 + math.exp(-3 * score))
            self.counts[key] = r

        if db_path and Path(db_path).exists():
            with open(db_path, encoding="utf-8") as f:
                db = json.load(f)
            for db_cat, cat in CAT_MAP.items():
                for item in db.get(db_cat, []):
                    key = (cat, item.get("ptt_model", ""))
                    ptt = 1.0 - float(item.get("neg_rate", 0.5) or 0.5)
                    if key in self.scores:
                        self.scores[key] = (self.scores[key] + ptt) / 2
                    else:
                        self.scores[key] = ptt

    def get(self, category: str, model: str, default: float = 0.5) -> float:
        """取得零件的情感分數 [0,1]。介面與舊版完全相同。"""
        key = (category, model)
        if key in self.scores:
            return self.scores[key]
        for (cat, mod), score in self.scores.items():
            if cat == category and (
                model.lower() in mod.lower() or mod.lower() in model.lower()
            ):
                return score
        return default

    def get_dimensions(self, category: str, model: str) -> dict:
        key = (category, model)
        if key in self.dimensions:
            return dict(self.dimensions[key])
        for (cat, mod), dims in self.dimensions.items():
            if cat == category and (
                model.lower() in mod.lower() or mod.lower() in model.lower()
            ):
                return dict(dims)
        return {d: 0.5 for d in DIMENSIONS}

    def get_with_custom_weights(self, category: str, model: str,
                                 weights: dict) -> float:
        dims = self.get_dimensions(category, model)
        total_w = sum(weights.values()) or 1.0
        return round(
            sum(weights.get(d, 0) * dims.get(d, 0.5) for d in DIMENSIONS) / total_w,
            4
        )

    def get_sample_info(self, category: str, model: str) -> dict:
        key = (category, model)
        return self.sample_counts.get(key, {d: 0 for d in DIMENSIONS})
