"""
情感分數計算器：整合巴哈評論 + PTT (ga_database_v2.json)
- 巴哈：從 matched_part1~3.jsonl 統計正/負/中立，加上 GP/BP 加權
- PTT：從 ga_database_v2.json 的 neg_rate 直接換算
- 最終分數：兩個來源平均（若只有一個來源則直接使用）
"""
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional

from config import CAT_MAP


class SentimentScorer:
    def __init__(self, jsonl_paths: list[Path], db_path: Optional[Path] = None):
        self.scores: dict[tuple, float] = {}  # (category, model) → [0,1]
        self.counts: dict[tuple, dict] = {}   # 方便除錯
        self._load(jsonl_paths, db_path)

    def _load(self, paths, db_path):
        raw = defaultdict(lambda: {
            "positive": 0, "neutral": 0, "negative": 0,
            "gp_sum": 0, "bp_sum": 0, "total": 0
        })

        # ── 巴哈資料 ──
        for path in paths:
            if not Path(path).exists():
                print(f"[SentimentScorer] 找不到檔案：{path}")
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (d.get("category", ""), d.get("model", ""))
                    label = d.get("label", "中立")
                    gp = int(d.get("GP", 0) or 0)
                    bp = int(d.get("BP", 0) or 0)
                    r = raw[key]
                    if label == "正評":
                        r["positive"] += 1
                    elif label == "負評":
                        r["negative"] += 1
                    else:
                        r["neutral"] += 1
                    r["gp_sum"] += gp
                    r["bp_sum"] += bp
                    r["total"] += 1

        # ── 轉換巴哈成 [0,1] 情感分數 ──
        baha_scores = {}
        for key, r in raw.items():
            total = max(r["total"], 1)
            raw_score = (
                r["positive"] * 1.0
                + r["neutral"] * 0.0
                + r["negative"] * (-1.0)
                + (r["gp_sum"] - r["bp_sum"]) / 10.0
            ) / total
            baha_scores[key] = self._sigmoid(raw_score)
            self.counts[key] = r

        # ── PTT 資料（從 ga_database_v2.json 的 neg_rate）──
        ptt_scores = {}
        if db_path and Path(db_path).exists():
            with open(db_path, encoding="utf-8") as f:
                db = json.load(f)
            for db_cat, cat in CAT_MAP.items():
                for item in db.get(db_cat, []):
                    model = item.get("ptt_model", "")
                    neg_rate = float(item.get("neg_rate", 0.5) or 0.5)
                    ptt_scores[(cat, model)] = 1.0 - neg_rate
        else:
            print("[SentimentScorer] 找不到 ga_database_v2.json，只使用巴哈資料")

        # ── 合併兩個來源 ──
        all_keys = set(baha_scores.keys()) | set(ptt_scores.keys())
        for key in all_keys:
            b = baha_scores.get(key)
            p = ptt_scores.get(key)
            if b is not None and p is not None:
                self.scores[key] = (b + p) / 2
            elif b is not None:
                self.scores[key] = b
            else:
                self.scores[key] = p

        print(f"[SentimentScorer] 巴哈：{len(baha_scores)} 筆  PTT：{len(ptt_scores)} 筆  合併：{len(self.scores)} 筆")

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-3 * x))

    def get(self, category: str, model: str, default: float = 0.5) -> float:
        key = (category, model)
        if key in self.scores:
            return self.scores[key]
        # 模糊比對
        for (cat, mod), score in self.scores.items():
            if cat == category and (
                model.lower() in mod.lower() or mod.lower() in model.lower()
            ):
                return score
        return default
