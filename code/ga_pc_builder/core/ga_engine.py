"""
GA 主體：Build（個體）+ GARecommender（演算法）
"""
import random
import numpy as np
from dataclasses import dataclass, field

from config import USAGE_WEIGHTS, CAT_MAP
from data.catalog import Part, PartCatalog
from data.sentiment import SentimentScorer
from policies.ram_policy import pick_ram, ram_score
from policies.psu_policy import psu_score
from policies.ssd_policy import ssd_score
from advisor.compatibility import CompatibilityChecker



REQUIRED_CATS = ["CPU", "GPU", "記憶體", "主機板", "SSD", "電源", "機殼"]
OPTIONAL_CATS = ["HDD"]
COOLING_CATS  = ["風冷", "水冷"]

PRICE_CEILING = {
    "CPU":  22900,
    "GPU":  59990,
    "記憶體": 27888,
    "主機板": 36990,
    "SSD":  54000,
    "HDD":   4890,
    "風冷":   4290,
    "水冷":  12490,
    "機殼":  12490,
    "電源":   9990,
}


@dataclass
class Build:
    parts: dict[str, Part] = field(default_factory=dict)

    @property
    def total_price(self) -> int:
        return sum(p.price for p in self.parts.values())

    def copy(self) -> "Build":
        return Build(parts=dict(self.parts))


class GARecommender:
    def __init__(
        self,
        catalog: PartCatalog,
        scorer: SentimentScorer,
        checker: CompatibilityChecker,
        usage: str = "遊戲",
        budget: int = 50000,
        pop_size: int = 150,
        generations: int = 300,
        elite_k: int = 2,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.25,
        tournament_k: int = 5,
        cooling_prefer: str = "auto",
        psu_tier: str = "standard",
    ):
        assert usage in USAGE_WEIGHTS, f"usage 必須是 {list(USAGE_WEIGHTS.keys())}"
        self.catalog     = catalog
        self.scorer      = scorer
        self.checker     = checker
        self.usage       = usage
        self.budget      = budget
        self.pop_size    = pop_size
        self.generations = generations
        self.elite_k     = elite_k
        self.cr          = crossover_rate
        self.mr          = mutation_rate
        self.tourn_k     = tournament_k
        self.cooling_prefer = cooling_prefer
        self.psu_tier = psu_tier  
        self.weights     = USAGE_WEIGHTS[usage]
        self.history_best: list[float] = []
        self.history_avg:  list[float] = []

    def _random_build(self) -> Build:
        b = Build()
        for cat in REQUIRED_CATS:
            if cat == "記憶體":
                ram = pick_ram(self.catalog, b, self.budget)
                if ram:
                    b.parts["記憶體"] = ram
            else:
                pool = self.catalog.get(cat)
                if pool:
                    b.parts[cat] = random.choice(pool)
        if random.random() < 0.5:
            pool = self.catalog.get("HDD")
            if pool:
                b.parts["HDD"] = random.choice(pool)
        cooling_cat = self._pick_cooling()
        pool = self.catalog.get(cooling_cat)
        if pool:
            b.parts[cooling_cat] = random.choice(pool)
        return b

    def _pick_cooling(self) -> str:
        if self.cooling_prefer in ("風冷", "水冷"):
            return self.cooling_prefer
        # auto 模式：依預算分級決定風冷/水冷機率
        # 低預算水冷會吃掉預算導致失衡，故大幅偏向風冷；
        # 高預算則維持公平競爭，讓 GA 自由選擇。
        if self.budget <= 25000:
            water_prob = 0.10      # 低預算：90% 風冷
        elif self.budget <= 40000:
            water_prob = 0.30      # 中預算：70% 風冷
        else:
            water_prob = 0.50      # 高預算：50/50 公平競爭
        return "水冷" if random.random() < water_prob else "風冷"

    def _init_population(self) -> list[Build]:
        return [self._random_build() for _ in range(self.pop_size)]

    def fitness(self, build: Build) -> float:
        w = self.weights

        # 硬約束：嚴重超支（>20%）直接判定為極差解，不再計算其他加分
        # 避免高效能加分蓋過預算懲罰，導致 11 萬配置出現在 2 萬預算
        if build.total_price > self.budget * 1.20:
            over_ratio = (build.total_price - self.budget) / self.budget
            return -10.0 - over_ratio   # 越超支排越後面

        # 硬約束：散熱壓不住 CPU（會降頻、噪音、壽命問題）
        # 用實際功耗對比推斷壓制力，壓不住直接淘汰
        cpu_part = build.parts.get("CPU")
        cooler_part = build.parts.get("風冷") or build.parts.get("水冷")
        if cpu_part and cooler_part:
            from data.cpu_power import get_actual_power
            from data.cooler_capacity import estimate_cooler_capacity
            nominal = float(cpu_part.specs.get("tdp", 0) or 0)
            cpu_pwr = get_actual_power(cpu_part.name, fallback_tdp=nominal or 65)
            cool_cap = estimate_cooler_capacity(cooler_part.name, cooler_part.specs)
            if cpu_pwr > cool_cap:
                # 壓不住：硬約束淘汰，差距越大排越後
                return -5.0 - (cpu_pwr - cool_cap) / 100

        perf     = self._perf_score(build)
        sent     = self._sentiment_score(build)
        cp       = self._cp_score(build)
        b_pen    = self._budget_penalty(build)
        c_pen, _ = self.checker.check(build,self.psu_tier)

        score = (
            w["w_perf"]   * perf
          + w["w_sent"]   * sent
          + w["w_cp"]     * cp
          - w["w_budget"] * b_pen
          - w["w_compat"] * c_pen
        )

        # 記憶體 
        score += ram_score(build.parts.get("記憶體"), self.budget)

        #  SSD
        ssd = build.parts.get("SSD")
        if ssd:
            score += ssd_score(ssd, self.usage)

        # 遊戲模式 CPU/GPU 平衡 
        if self.usage == "遊戲":
            cpu = build.parts.get("CPU")
            gpu = build.parts.get("GPU")
            if cpu and gpu:
                cpu_bench = float(cpu.specs.get("benchmark", 0) or 0)
                gpu_bench = float(gpu.specs.get("benchmark", 0) or 0)
                cpu_price = cpu.price
                gpu_price = gpu.price

                if gpu_bench < 18000:
                    score -= 0.10
                elif gpu_bench < 20000:
                    score -= 0.05

                if gpu_bench > 0 and cpu_bench < gpu_bench * 0.7:
                    score -= 0.15

                if gpu_price > 10000 and cpu_bench < 25000:
                    score -= 0.12

                # 若 CPU 明顯比 GPU 昂貴，才扣分；避免過度懲罰正常的 GPU 偏重組合。
                if cpu_price > gpu_price * 1.2:
                    score -= min((cpu_price - gpu_price * 1.2) / self.budget * 0.8, 0.20)

        # 高階 CPU 配入門主機板懲罰 
        cpu = build.parts.get("CPU")
        mb  = build.parts.get("主機板")
        if cpu and mb:
            if cpu.price > 8000 and mb.price < 4000:
                score -= 0.15
            elif cpu.price > 5000 and mb.price < 3500:
                score -= 0.10

        #  散熱器價格合理性 
        cooler = build.parts.get("風冷") or build.parts.get("水冷")
        cpu    = build.parts.get("CPU")
        if cooler and cpu:
            cpu_tdp      = float(cpu.specs.get("tdp", 65) or 65)
            cooler_price = cooler.price

            if cpu_tdp <= 65:
                if cooler_price > 1200:
                    score -= (cooler_price - 1200) / 60000 * 0.3
            elif cpu_tdp <= 125:
                if cooler_price > 2000:
                    score -= (cooler_price - 2000) / 60000 * 0.3
            else:
                if cooler_price > 3000:
                    score -= (cooler_price - 3000) / 60000 * 0.3

        #  預算偏緊時偏向風冷 
        water = build.parts.get("水冷")
        if water:
            # 低預算（≤25000）配水冷本身就不合理：水冷吃預算導致整機失衡
            if self.budget <= 25000:
                score -= 0.15
            # 水冷佔預算比例過高額外懲罰
            if water.price > self.budget * 0.12:
                score -= (water.price - self.budget * 0.12) / self.budget * 0.5
            # 接近預算上限時偏向風冷
            if build.total_price > self.budget * 0.85:
                score -= 0.05
        # 機殼優先選有風扇的
        case = build.parts.get("機殼")
        if case:
           if "無風扇" in case.name:
               score -= 0.08  #
        
        score += psu_score(
            build.parts.get("電源"),
            build.parts.get("CPU"),
            build.parts.get("GPU"),
            self.psu_tier
        )

        return score

    def _perf_score(self, build: Build) -> float:
        pw = self.weights["perf_weights"]
        score = total_w = 0.0
        for cat, part in build.parts.items():
            cat_key = "散熱" if cat in COOLING_CATS else cat
            w = pw.get(cat_key, pw.get(cat, 0.0))
            benchmark = part.specs.get("benchmark")
            if benchmark and float(benchmark) > 0:
                norm = min(float(benchmark) / 50000, 1.0)
            else:
                ceil = PRICE_CEILING.get(cat, 10000)
                norm = min(part.price / ceil, 1.0)
            score   += w * norm
            total_w += w
        return score / total_w if total_w > 0 else 0.0

    def _sentiment_score(self, build: Build) -> float:
        scores = [self.scorer.get(cat, part.short_name)
                  for cat, part in build.parts.items()]
        return float(np.mean(scores)) if scores else 0.5

    def _cp_score(self, build: Build) -> float:
        perf = self._perf_score(build)
        ratio = build.total_price / max(self.budget, 1)
        return min(perf / ratio, 1.0) if ratio > 0 else 0.0

    def _budget_penalty(self, build: Build) -> float:
        over = build.total_price - self.budget
        if over <= 0:
            return 0.0
        ratio = over / self.budget
        # 分段懲罰：小幅超支線性扣，大幅超支指數爆炸（移除上限）
        # 10% 超支扣 0.5；30% 超支扣 ~3；超支 1 倍以上扣到完全淘汰
        if ratio <= 0.30:
            return ratio * 5.0          # 0~1.5
        else:
            # 超過 30% 後改用指數，超支越多罰越狠，無上限
            return 1.5 + (ratio - 0.30) ** 1.5 * 20.0

    def _tournament(self, pop: list[Build], fits: list[float]) -> Build:
        idx = random.sample(range(len(pop)), min(self.tourn_k, len(pop)))
        return pop[max(idx, key=lambda i: fits[i])].copy()

    def _crossover(self, p1: Build, p2: Build):
        if random.random() > self.cr:
            return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        for cat in set(list(p1.parts) + list(p2.parts)):
            if random.random() < 0.5:
                v1 = p1.parts.get(cat)
                v2 = p2.parts.get(cat)
                if v2: c1.parts[cat] = v2
                elif cat in c1.parts: del c1.parts[cat]
                if v1: c2.parts[cat] = v1
                elif cat in c2.parts: del c2.parts[cat]
        return c1, c2

    def _mutate(self, build: Build) -> Build:
        b = build.copy()
        for cat in list(b.parts):
            if cat == "記憶體":
                if random.random() < self.mr:
                    ram = pick_ram(self.catalog, b, self.budget)
                    if ram:
                        b.parts["記憶體"] = ram
            elif random.random() < self.mr:
                pool = self.catalog.get(cat)
                if pool:
                    b.parts[cat] = random.choice(pool)
        if random.random() < self.mr:
            if "HDD" in b.parts:
                del b.parts["HDD"]
            else:
                pool = self.catalog.get("HDD")
                if pool:
                    b.parts["HDD"] = random.choice(pool)
        if random.random() < self.mr * 0.5:
            old = [c for c in COOLING_CATS if c in b.parts]
            if old:
                del b.parts[old[0]]
            new_cool = self._pick_cooling()
            pool = self.catalog.get(new_cool)
            if pool:
                b.parts[new_cool] = random.choice(pool)
        has_cooling = any(c in b.parts for c in COOLING_CATS)
        if not has_cooling:
            new_cool = self._pick_cooling()
            pool = self.catalog.get(new_cool)
            if pool:
                b.parts[new_cool] = random.choice(pool)
        return b

    def run(self, verbose: bool = True) -> list[Build]:
        pop = self._init_population()
        for gen in range(self.generations):
            fits = [self.fitness(b) for b in pop]
            idx  = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
            self.history_best.append(fits[idx[0]])
            self.history_avg.append(float(np.mean(fits)))
            if verbose and (gen % 20 == 0 or gen == self.generations - 1):
                print(f"  Gen {gen:4d} | best={fits[idx[0]]:.4f} "
                      f"avg={float(np.mean(fits)):.4f} | "
                      f"price={pop[idx[0]].total_price:,}")
            new_pop = [pop[i].copy() for i in idx[:self.elite_k]]
            while len(new_pop) < self.pop_size:
                c1, c2 = self._crossover(
                    self._tournament(pop, fits),
                    self._tournament(pop, fits)
                )
                new_pop += [self._mutate(c1), self._mutate(c2)]
            pop = new_pop[:self.pop_size]

        fits = [self.fitness(b) for b in pop]
        idx  = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        top5 = []
        seen_gpu = set()
        for i in idx:
            b = pop[i]
            gpu = b.parts.get("GPU")
            key = gpu.specs.get("ptt_model", gpu.name) if gpu else ""
            if key not in seen_gpu:
                seen_gpu.add(key)
                top5.append(b)
            elif len(top5) < 5:
                top5.append(b)
            if len(top5) >= 5:
                break
        return top5
