"""
情感分數計算器 v3 — 使用面向級口碑分數

與 v2 的差異：
  v2：BERT 預測 → 合併成五大面向（效能/溫控/噪音/保固/CP值）→ 依使用情境加權
  v3：面向標註直接加總 → 各類別用自己的面向（CPU 6 個、風冷 4 個…）→ 各面向等權平均
      不再合併成五大面向，也不再依使用情境預設面向權重。

資料來源：
  part_aspect_sentiment.json — 由 bert_train/build_part_sentiment.py 產生。
  每個面向的分數已依評價則數往類別平均修正過（則數少的不會出現極端分數）。

類別內正規化：
  每個面向在同類別的型號之間做 min-max（該面向最好的型號 1、最差的 0），與 GA 對
  效能的處理方式相同。原始分數各面向的水準差很多（顯卡的穩定普遍 0.2 上下、溫度
  0.75 上下），不正規化的話平均起來每個型號都差不多，口碑分不出高下。
  型號分數是正規化後各面向的平均；沒有任何評論的型號拿該類別的平均。

使用者指定面向：
  get(..., aspects=[...]) 指定的面向佔一半，其餘面向的平均佔另一半（PREF_SHARE）。
  不是只看指定的面向：在意穩定的人不代表完全不管效能、VRAM。

相容性：
  get(category, model) 介面不變，既有程式（api.py、ga_engine.py）無需修改。
"""
import json
import math
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

from config import CAT_MAP

# 一個型號在某面向的正負評價達這個則數才算「有資料」：
# 正規化的上下界只看有資料的型號，可供使用者指定的面向也依此判斷。
MIN_MENTIONS = 10

# 使用者指定面向時，指定的面向佔該零件口碑的比重；其餘面向的平均佔剩下的。
# 不指定時每個面向各佔 1/N（顯卡 7 個面向就是 14%），指定後提高到一半。
PREF_SHARE = 0.5


class SentimentScorer:
    """
    情感分數計算器（面向口碑版）

      scorer.get(category, model) -> float [0,1]
      scorer.get(category, model, aspects=["穩定"]) -> 加重指定的面向
      scorer.get_aspects(category, model) -> dict 面向明細（正規化後）
      scorer.selectable[category] -> 資料夠、可以讓使用者指定的面向
    """

    def __init__(self, jsonl_paths: list = None,
                 db_path: Optional[Path] = None,
                 sentiment_path: Optional[Path] = None,
                 common_scale: bool = False):
        # False：每個面向各自 min-max（最好 1、最差 0）。
        # True ：同類別的面向共用一把尺（該類別全距最大的面向），各型號分數很接近的面向
        #        不會被硬拉開。顯卡的穩定全距只有 0.17，各自 min-max 會把 0.25 與 0.31
        #        拉成 0.67 與 1.0，GA 會為了這點差距放棄一個等級的效能。
        self.common_scale = common_scale
        self.aspects: dict = {}
        self.raw_aspects: dict = {}
        self.sample_counts: dict = {}
        self.scores: dict = {}
        self.counts: dict = {}
        self.category_default: dict = {}
        self.category_aspects: dict = {}
        self.selectable: dict = {}
        self.raw_counts: dict = {}
        self.category_raw: dict = {}
        self._ranking: dict = {}
        self._resolved: dict = {}

        if sentiment_path is None:
            sentiment_path = Path(__file__).parent / "part_aspect_sentiment.json"

        self._load_aspect_scores(sentiment_path, db_path)

        if not self.scores:
            print("[SentimentScorer] 無面向口碑分數，回退至舊版標籤統計")
            self._load_legacy(jsonl_paths, db_path)

    def _load_aspect_scores(self, path, db_path):
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

            sample_n = {a: int(v["正面"]) + int(v["負面"]) for a, v in item["aspects"].items()}
            self.raw_aspects[key] = {a: float(v["score"]) for a, v in item["aspects"].items()}
            self.sample_counts[key] = sample_n
            self.raw_counts[key] = {a: (int(v["正面"]), int(v["負面"]))
                                    for a, v in item["aspects"].items()}
            self.counts[key] = {
                "review_count": item.get("review_count", 0),
                "samples": sample_n,
            }

        bounds = self._aspect_bounds(self._db_models(db_path))

        # 各類別全距最大的面向，common_scale 時當作該類別所有面向共用的尺
        widest = defaultdict(float)
        for (category, _), (lo, hi) in bounds.items():
            widest[category] = max(widest[category], hi - lo)

        def normalize(category, raw):
            out = {}
            for aspect, value in raw.items():
                lo, hi = bounds.get((category, aspect), (None, None))
                # 有資料的型號不到兩個就沒有可比的對象，這個面向對所有型號都是 0.5（不影響排序）
                if lo is None:
                    out[aspect] = 0.5
                elif self.common_scale:
                    # 以該面向的中點為 0.5，差距用類別共用的尺量：全距窄的面向落在 0.5 附近
                    position = 0.5 + (value - (lo + hi) / 2) / widest[category]
                    out[aspect] = max(0.0, min(position, 1.0))
                else:
                    out[aspect] = max(0.0, min((value - lo) / (hi - lo), 1.0))
            return out

        for category, prior in data.get("_prior", {}).items():
            self.category_raw[category] = dict(prior["aspects"])
            self.category_aspects[category] = normalize(category, prior["aspects"])
            self.category_default[category] = (sum(self.category_aspects[category].values())
                                               / len(prior["aspects"]))

        for key, raw in self.raw_aspects.items():
            own = normalize(key[0], raw)
            average = self.category_aspects.get(key[0], {})
            # 評價不到 MIN_MENTIONS 則的面向用類別平均的位置，不用自己的分數：
            # 有些面向各型號的分數很接近（顯卡的穩定全距只有 0.17），兩三則評價造成的
            # 些微差距正規化後會被放大成全場最高或最低。
            self.aspects[key] = {
                a: own[a] if self.sample_counts[key][a] >= MIN_MENTIONS else average.get(a, 0.5)
                for a in raw
            }
            self.scores[key] = sum(self.aspects[key].values()) / len(raw)

        print(f"[SentimentScorer] 載入面向口碑分數：{len(self.scores)} 個零件")

    @staticmethod
    def _db_models(db_path) -> set:
        """GA 資料庫裡的 (類別, 型號)。口碑檔另外含資料庫沒在賣的型號（舊顯卡等），
        正規化只在 GA 實際會挑的型號之間比。讀不到資料庫就回傳空集合（用全部型號）。"""
        if not db_path or not Path(db_path).exists():
            return set()
        with open(db_path, encoding="utf-8") as f:
            db = json.load(f)
        return {(cat, item.get("ptt_model", ""))
                for db_cat, cat in CAT_MAP.items() for item in db.get(db_cat, [])}

    def _aspect_bounds(self, db_models: set) -> dict:
        """回傳 {(類別, 面向): (最低, 最高)}，順便定出各類別可供指定的面向。"""
        pool = [k for k in self.raw_aspects if not db_models or k in db_models]
        values = defaultdict(list)
        n_models = defaultdict(int)
        for key in pool:
            n_models[key[0]] += 1
            for aspect, value in self.raw_aspects[key].items():
                if self.sample_counts[key][aspect] >= MIN_MENTIONS:
                    values[(key[0], aspect)].append(value)

        # 排名用的名單與正規化上下界是同一批型號（GA 會挑、該面向評價夠的）
        self._ranking = dict(values)
        bounds = {}
        for (category, aspect), vals in values.items():
            if len(vals) >= 2 and max(vals) > min(vals):
                bounds[(category, aspect)] = (min(vals), max(vals))
                # 過半的型號有資料才開放指定，否則選了也分不出型號差異
                if len(vals) * 2 >= n_models[category]:
                    self.selectable.setdefault(category, []).append(aspect)
        return bounds

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

    @staticmethod
    def _best_fuzzy_key(category: str, model: str, table: dict) -> tuple | None:
        """在 table（key 為 (category, model) tuple）裡找子字串包含關係的候選，
        取型號字串最長的那個。子字串比對本身是雙向的（"RTX5070" 是
        "RTX5070Ti" 的子字串），如果直接取第一個命中的候選，字典走訪順序
        一旦把 "RTX5070" 排在 "RTX5070Ti" 前面，Ti 型號就會被短字串誤攔截、
        拿到錯的分數——取最長匹配才能確保「更精確的型號名稱」優先命中。"""
        candidates = [
            k for k in table
            if k[0] == category and (
                model.lower() in k[1].lower() or k[1].lower() in model.lower()
            )
        ]
        if not candidates:
            candidates = SentimentScorer._fallback_candidates(category, model, table)
        if not candidates:
            return None
        return max(candidates, key=lambda k: len(k[1]))

    @staticmethod
    def _fallback_candidates(category: str, model: str, table: dict) -> list:
        """原價屋商品名稱跟評論 key 的寫法常對不上，整串子字串比對會落空：
          品牌是中文或多了系列名：「美光 Micron Crucial T500」vs "Micron T500"
          記憶體用縮寫：「D4-3600」vs "DDR4-3600"
          詞序不同：「Toshiba 2TB【P300系列】」vs "Toshiba P300 2TB"
        依序退一步：(1) 評論 key 去掉品牌後整段比對 (2) 只比對 key 裡含數字的型號字。
        前後都不能接英數字，避免 "NH-D15" 命中 "NH-D15S"。"""
        name = " ".join(model.lower().split())
        name = re.sub(r"(?<![a-z])d([45])-", r"ddr\1-", name)

        def hit(word):
            return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", name)

        keys = [k for k in table if k[0] == category]
        found = [k for k in keys if hit(k[1].lower())]
        if found:
            return found
        found = [k for k in keys if len(k[1].split()) >= 2
                 and hit(" ".join(k[1].lower().split()[1:]))]
        if found:
            return found
        return [k for k in keys
                if any(len(w) >= 4 and re.search(r"\d", w) and hit(w)
                       for w in k[1].lower().split()[1:])]

    def _resolve(self, category: str, model: str) -> tuple | None:
        """商品名稱對到口碑檔的鍵。一次 GA 會查幾十萬次，模糊比對的結果要快取。"""
        key = (category, model)
        if key not in self._resolved:
            self._resolved[key] = (key if key in self.scores
                                   else self._best_fuzzy_key(category, model, self.scores))
        return self._resolved[key]

    def get(self, category: str, model: str, default: float = 0.5,
            aspects: list | None = None, pref_share: float = PREF_SHARE) -> float:
        """取得零件的情感分數 [0,1]。
        aspects 有給（使用者指定在意的面向）就加重這些面向：它們的平均佔 pref_share，
        其餘面向的平均佔剩下的；pref_share=1 表示只看指定的面向。
        沒有評論的零件拿該類別的平均，而不是固定的 0.5。"""
        key = self._resolve(category, model)
        table = self.aspects.get(key) if key else self.category_aspects.get(category)
        if aspects and table:
            picked = [table[a] for a in aspects if a in table]
            rest = [v for a, v in table.items() if a not in aspects]
            if picked and rest:
                return (pref_share * sum(picked) / len(picked)
                        + (1 - pref_share) * sum(rest) / len(rest))
            if picked:
                return sum(picked) / len(picked)
        if key:
            return self.scores[key]
        return self.category_default.get(category, default)

    def get_aspects(self, category: str, model: str) -> dict:
        key = self._resolve(category, model)
        return dict(self.aspects.get(key) or self.category_aspects.get(category, {}))

    def aspect_evidence(self, category: str, model: str) -> dict | None:
        """零件各面向口碑的原始依據，給前端顯示「這顆在這個面向到底好不好」用。
        對不到任何評論的零件回傳 None。

        score 是口碑檔裡的面向分數（已往類別平均修正），rank / ranked_total 是它在
        同類別、該面向評價達 MIN_MENTIONS 則的型號之間的名次。評價不到 MIN_MENTIONS
        則的面向 rank 是 None：GA 對這種面向用的是類別平均，硬排名次會誤導。
        """
        key = self._resolve(category, model)
        if not key or key not in self.raw_aspects:
            return None
        rows = []
        for aspect, score in self.raw_aspects[key].items():
            pos, neg = self.raw_counts[key][aspect]
            ranked = self._ranking.get((category, aspect), [])
            enough = pos + neg >= MIN_MENTIONS
            rows.append({
                "aspect": aspect,
                "positive": pos,
                "negative": neg,
                "score": round(score, 3),
                "category_avg": round(self.category_raw.get(category, {}).get(aspect, 0.5), 3),
                # 同分的型號名次相同（同晶片的 K / KF 版共用一組分數）
                "rank": 1 + sum(v > score for v in ranked) if enough and ranked else None,
                "ranked_total": len(ranked),
            })
        return {"model": key[1], "aspects": rows}

    def get_sample_info(self, category: str, model: str) -> dict:
        return dict(self.sample_counts.get((category, model), {}))
