"""
升級建議：剩餘預算還能把哪個零件換得更好。

設計原則是「只講算得出來的數字」。CPU 與 GPU 在 ga_database_v2.json 裡 100% 有
PassMark 跑分（34/34、238/238），所以效能提升用真實跑分差計算，跟 GA 引擎
_perf_score() 判斷效能的依據完全一致；記憶體、SSD、主機板、電源沒有跑分欄位，
就只陳述規格差異（16GB → 32GB），不換算成任何百分比。

口碑分數另外用 sentiment_delta 欄位回報，不混進效能——兩者是不同的東西，
混在一起講會變成「用網友情緒推估 FPS」。
"""
from dataclasses import dataclass, field

from core.ga_engine import Build
from data.catalog import Part, PartCatalog
from data.sentiment import SentimentScorer

# 各類別「換得更好」的判斷依據，依序比較。CPU/GPU 用跑分，其餘用規格。
# 沒列在這裡的類別（機殼、風冷、水冷）不提供升級建議：它們沒有可量化的優劣軸，
# 硬要推薦就只能回到「比較貴所以比較好」的假設。
UPGRADE_AXIS: dict[str, tuple[str, ...]] = {
    "CPU": ("benchmark",),
    "GPU": ("benchmark",),
    "記憶體": ("capacity_gb", "speed"),
    "SSD": ("capacity_gb", "read_mbs"),
    "HDD": ("capacity_gb",),
    "主機板": ("m2_slots",),
    "電源": ("wattage",),
}

# 要在建議裡陳述的規格差異：欄位 → (顯示名稱, 單位)
SPEC_LABELS: dict[str, tuple[str, str]] = {
    "benchmark": ("跑分", ""),
    "cores": ("核心", "核"),
    "threads": ("執行緒", "緒"),
    "boost_ghz": ("加速時脈", "GHz"),
    "vram_gb": ("顯示記憶體", "GB"),
    "capacity_gb": ("容量", "GB"),
    "speed": ("頻率", "MHz"),
    "read_mbs": ("循序讀取", "MB/s"),
    "m2_slots": ("M.2 插槽", "個"),
    "wattage": ("瓦數", "W"),
}

USAGE_PRIORITY: dict[str, tuple[str, ...]] = {
    "遊戲": ("GPU", "CPU", "記憶體", "SSD", "主機板", "電源"),
    "工作": ("CPU", "記憶體", "SSD", "GPU", "主機板", "電源"),
}
DEFAULT_PRIORITY = ("CPU", "記憶體", "GPU", "SSD", "主機板", "電源")


@dataclass
class UpgradeOption:
    category: str
    current_part: Part
    upgrade_part: Part
    price_increase: int
    # 只有 CPU/GPU 算得出來，其餘類別是 None——沒有跑分就不給百分比
    benchmark_gain_pct: float | None
    # 論壇口碑的變化量 [-1, 1]，跟效能完全分開回報
    sentiment_delta: float
    spec_changes: list[str] = field(default_factory=list)


def _num(part: Part, key: str) -> float | None:
    value = part.specs.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format(value: float, unit: str) -> str:
    text = f"{value:g}"
    return f"{text}{unit}" if unit else text


def _describe_changes(current: Part, upgrade: Part, keys: tuple[str, ...]) -> list[str]:
    """列出實際變好的規格。只描述資料庫裡真的有的欄位，沒有就不提。"""
    changes = []
    for key in keys:
        before, after = _num(current, key), _num(upgrade, key)
        if before is None or after is None or after <= before:
            continue
        label, unit = SPEC_LABELS.get(key, (key, ""))
        changes.append(f"{label} {_format(before, unit)} → {_format(after, unit)}")
    return changes


class UpgradeAdvisor:
    def __init__(self, catalog: PartCatalog, scorer: SentimentScorer, checker=None):
        self.catalog = catalog
        self.scorer = scorer
        self.checker = checker

    def get_smart_recommendations(
        self,
        build: Build,
        budget: int,
        remaining_budget: int,
        usage: str = "遊戲",
    ) -> list[dict]:
        """每個類別挑一個最值得的升級，依用途優先順序回傳。

        累計花費不超過剩餘預算，所以整份清單是可以一起買的，不是各自獨立的選項。
        """
        priority = USAGE_PRIORITY.get(usage, DEFAULT_PRIORITY)

        recommendations = []
        spent = 0
        for category in priority:
            current = build.parts.get(category)
            if not current:
                continue
            option = self._best_upgrade(category, current, build, remaining_budget - spent)
            if option is None:
                continue
            recommendations.append({
                "priority": len(recommendations) + 1,
                "category": category,
                "current": current,
                "upgrade": option.upgrade_part,
                "cost": option.price_increase,
                "benchmark_gain_pct": option.benchmark_gain_pct,
                "sentiment_delta": round(option.sentiment_delta, 3),
                "spec_changes": option.spec_changes,
            })
            spent += option.price_increase
        return recommendations

    def _best_upgrade(
        self,
        category: str,
        current: Part,
        build: Build,
        affordable: int,
    ) -> UpgradeOption | None:
        axis = UPGRADE_AXIS.get(category)
        if not axis or affordable <= 0:
            return None

        primary = axis[0]
        current_primary = _num(current, primary)
        if current_primary is None:
            return None

        best: UpgradeOption | None = None
        best_primary = current_primary

        for candidate in self.catalog.get(category):
            price_increase = candidate.price - current.price
            if price_increase <= 0 or price_increase > affordable:
                continue

            # 主要指標沒有變好就不算升級。原本的寫法是「比現在貴就當作比較好」，
            # 但貴不等於強——同價位帶裡效能倒退的型號很常見。
            candidate_primary = _num(candidate, primary)
            if candidate_primary is None or candidate_primary <= best_primary:
                continue

            sentiment_delta = (
                self.scorer.get(category, candidate.short_name)
                - self.scorer.get(category, current.short_name)
            )

            # 沒有跑分可以證明效能變好的類別（記憶體、SSD、主機板、電源），不接受
            # 口碑倒退的候選——這是個以論壇口碑為賣點的系統，為了多一個 M.2 插槽就
            # 換一張網友評價更差的板子說不過去。CPU/GPU 有實測跑分佐證，效能提升
            # 足以支撐口碑的小幅落差，所以不套用這條。
            if primary != "benchmark" and sentiment_delta < 0:
                continue

            if not self._compatible(category, candidate, build):
                continue

            gain = None
            if primary == "benchmark":
                gain = round((candidate_primary - current_primary) / current_primary * 100, 1)

            best_primary = candidate_primary
            best = UpgradeOption(
                category=category,
                current_part=current,
                upgrade_part=candidate,
                price_increase=price_increase,
                benchmark_gain_pct=gain,
                sentiment_delta=sentiment_delta,
                spec_changes=_describe_changes(current, candidate, axis),
            )
        return best

    def _compatible(self, category: str, candidate: Part, build: Build) -> bool:
        """把零件換進去重算整份配置的相容性，只要沒有比現在更糟就放行。

        用 CompatibilityChecker 而不是自己寫一套簡化版：它會連電源瓦數、機殼長度、
        散熱器高度與壓制力一起檢查，自己寫的版本只看得到腳位跟記憶體規格。
        比較「penalty 有沒有變高」而不是「有沒有 issue」，是因為原本的配置就可能
        帶著輕微問題（例如電源餘裕稍嫌不足），那不該讓所有升級選項都被擋掉。
        """
        if self.checker is None:
            return True
        candidate_build = build.copy()
        candidate_build.parts[category] = candidate
        before, _ = self.checker.check(build)
        after, _ = self.checker.check(candidate_build)
        return after <= before
