"""
GA 主體：Build（個體）+ GARecommender（演算法）
"""
import random
import numpy as np
from dataclasses import dataclass, field

from config import USAGE_WEIGHTS, CAT_MAP, PREF_TERM_SHARE
from data.catalog import Part, PartCatalog
from data.sentiment import SentimentScorer, PREF_SHARE
from policies.ram_policy import pick_ram, ram_score
from policies.psu_policy import psu_score
from policies.ssd_policy import ssd_score
from advisor.compatibility import CompatibilityChecker, form_factor_fits, needs_16pin, has_native_16pin



REQUIRED_CATS = ["CPU", "GPU", "記憶體", "主機板", "SSD", "電源", "機殼"]
OPTIONAL_CATS = ["HDD"]
COOLING_CATS  = ["風冷", "水冷"]
# fitness 的硬約束都回傳 -5 以下（散熱壓不住 -5、超支 20% 以上 -10、缺零件 -20），正常配置不會低於這個值
HARD_FAIL = -5.0

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


class _PoolView:
    """只提供 get(category) 的輕量目錄，讓 pick_ram 在限定的候選裡挑"""
    def __init__(self, pools: dict):
        self._pools = pools

    def get(self, category: str) -> list:
        return self._pools.get(category, [])


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
        mutation_rate: float | None = None,
        tournament_k: int = 5,
        cooling_prefer: str = "auto",
        psu_tier: str = "standard",
        custom_weights: dict | None = None,
        aspect_prefs: dict | None = None,
        pref_term_share: float = PREF_TERM_SHARE,
        pref_share: float = PREF_SHARE,
        repair: bool = True,
        allow_16pin_adapter: bool = False,
    ):
        assert usage in USAGE_WEIGHTS, f"usage 必須是 {list(USAGE_WEIGHTS.keys())}"
        self.catalog     = catalog
        self.scorer      = scorer
        self.checker     = checker
        self.usage       = usage
        # 使用者指定在意的面向，例如 {"GPU": ["穩定"]}：該類別零件的口碑加重這些面向。
        # 存在這裡而不是設到 scorer 上——scorer 是全服務共用的，不能帶單次請求的狀態。
        # 資料不夠的面向（scorer.selectable 沒列的）直接忽略，選了也分不出型號差異。
        selectable = getattr(scorer, "selectable", {})
        self.aspect_prefs = {
            cat: picked for cat, aspects in (aspect_prefs or {}).items()
            if (picked := [a for a in aspects if a in selectable.get(cat, [])])
        }
        # 整機口碑只看評論資料夠的類別（有可指定面向的類別：顯卡、CPU、主機板、記憶體）。
        # 機殼、電源、散熱、SSD 的評論太少，分數多半是類別平均加上幾則評論的雜訊，
        # 讓它們參與平均只會稀釋有資料的類別。空集合表示不限制（舊版口碑檔沒有這項資訊）。
        # 記憶體另外排除：它由 pick_ram 的規則直接決定，不經過 fitness 競爭，
        # 口碑分數改變不了選哪一款，算進平均只會佔掉其他類別的份量。
        self._sent_cats = set(selectable) - {"記憶體"}
        # 使用者指定的面向在 fitness 裡獨立成一項，從口碑的權重分這個比例出來（見 fitness()）。
        # 0 表示用舊做法：不獨立成項，只在該零件的口碑裡加重指定的面向（對照實驗用）。
        self.pref_term_share = pref_term_share if self.aspect_prefs else 0.0
        # 不獨立成項時（pref_term_share=0），指定的面向佔該零件口碑的比例；1 表示只看指定的面向
        self.pref_share = pref_share
        self.budget      = budget
        self.pop_size    = pop_size
        self.generations = generations
        self.elite_k     = elite_k
        self.cr          = crossover_rate
        self.mr          = mutation_rate if mutation_rate is not None else self._pick_mutation_rate()
        self.tourn_k     = tournament_k
        # 目錄裡實際有零件的散熱類別（水冷可以由 config.ENABLE_WATER_COOLING 關掉）
        self.cooling_available = [c for c in COOLING_CATS if catalog.get(c)]
        if cooling_prefer in COOLING_CATS and cooling_prefer not in self.cooling_available:
            raise ValueError(f"目前不提供{cooling_prefer}，散熱偏好請選 auto 或 "
                             + " / ".join(self.cooling_available))
        self.cooling_prefer = cooling_prefer
        self.psu_tier = psu_tier
        # 淺拷貝：USAGE_WEIGHTS[usage] 是模組層級共用的常數字典，直接引用的話
        # 下面對 self.weights 的滑桿覆寫會反過來污染全域預設值，影響到同一個
        # process 裡其他沒有帶自訂權重的請求（api.py 用 lifespan 全服務共用資源，
        # 同步 endpoint 可能被 FastAPI 丟到不同 thread 平行處理）。
        self.weights     = dict(USAGE_WEIGHTS[usage])
        if custom_weights:
            self._apply_custom_weights(custom_weights)
        # 合法性修補：交叉／突變／隨機產生之後，把個體修成「每個必選類別剛好一個、
        # 散熱器剛好一個、CPU-主機板-記憶體-散熱器腳位與世代相容」。
        # 相容性懲罰仍保留在 fitness 裡當後備（repair=False 時就回到原本只靠懲罰的行為）。
        self.repair = repair
        # 16-pin 顯卡能不能用轉接線接在沒有原生 16-pin 的電源上（使用者決定）。
        # 不允許時修補會把電源換成有原生接頭的；允許時只在相容性檢查裡提示。
        self.allow_16pin_adapter = allow_16pin_adapter
        self._index_compat_pools()
        self.history_best: list[float] = []
        # 名人堂：run() 時記錄整個演化過程的好配置，前五名從這裡挑（見 _pick_top）
        self._hall: dict[tuple, tuple[float, "Build"]] = {}
        self.history_avg:  list[float] = []
        # 類別內 min-max 正規化用的範圍快取，只在初始化時掃一次 catalog，
        # 避免每次 fitness 評估都重算（一次 run 約 90,000 次 fitness 呼叫）。
        self._compute_perf_ranges()

    def _apply_custom_weights(self, custom_weights: dict):
        """把使用者三個滑桿值（perf/sent/cp，各 0~100，相對重要程度，不是絕對佔比）
        依比例重新分配進該 usage 預設的「效能+口碑+CP值」總份額裡。
        w_budget、w_compat 完全不受影響，維持該 usage 的預設值——
        保底避免相容性/預算的軟性約束被使用者的滑桿稀釋掉。"""
        default = USAGE_WEIGHTS[self.usage]
        reserved = default["w_perf"] + default["w_sent"] + default["w_cp"]

        try:
            raw_perf = max(0.0, float(custom_weights.get("perf", 0) or 0))
            raw_sent = max(0.0, float(custom_weights.get("sent", 0) or 0))
            raw_cp   = max(0.0, float(custom_weights.get("cp", 0) or 0))
        except (ValueError, TypeError):
            return  # 傳入格式有問題，安全退回預設值，不讓整個 GA 掛掉

        total_raw = raw_perf + raw_sent + raw_cp
        if total_raw <= 0:
            # 三個滑桿都是 0（或沒給有效值）→ 視為「交給系統決定」，維持預設比例
            return

        self.weights["w_perf"] = raw_perf / total_raw * reserved
        self.weights["w_sent"] = raw_sent / total_raw * reserved
        self.weights["w_cp"]   = raw_cp   / total_raw * reserved

    def _compute_perf_ranges(self):
        """算出每個類別的 benchmark / price 範圍，供 _perf_score() 做類別內 min-max 正規化。
        有 benchmark 資料的類別（目前實務上是 CPU/GPU）用 benchmark 範圍；
        沒有的類別（RAM/主機板/SSD/HDD/散熱/機殼/電源）用 price 範圍。
        取代原本固定分母（benchmark/50000、price/PRICE_CEILING）壓扁同類別內差距的問題。
        """
        self._bench_range: dict[str, tuple[float, float]] = {}
        self._price_range: dict[str, tuple[float, float]] = {}
        for cat in PRICE_CEILING:
            parts = self.catalog.get(cat)
            if not parts:
                continue
            benchmarks = []
            for p in parts:
                try:
                    b = float(p.specs.get("benchmark", 0) or 0)
                except (ValueError, TypeError):
                    b = 0.0
                if b > 0:
                    benchmarks.append(b)
            if benchmarks:
                self._bench_range[cat] = (min(benchmarks), max(benchmarks))
            prices = [p.price for p in parts if p.price > 0]
            if prices:
                self._price_range[cat] = (min(prices), max(prices))

    # ── 合法性修補 ────────────────────────────────────────────────
    @staticmethod
    def _sockets(part) -> list:
        """散熱器支援的腳位清單（資料庫是 list，保險起見也接受 JSON 字串）"""
        s = part.specs.get("supported_sockets", []) if part else []
        if isinstance(s, str):
            import json
            try:
                s = json.loads(s)
            except Exception:
                s = [s]
        return list(s or [])

    def _index_compat_pools(self):
        """預先依腳位／記憶體世代分好候選清單，修補時不用每次掃整個目錄"""
        self._mb_by_socket: dict[str, list] = {}
        for mb in self.catalog.get("主機板") or []:
            sock = str(mb.specs.get("socket", "")).strip()
            self._mb_by_socket.setdefault(sock, []).append(mb)
        self._ram_by_type: dict[str, list] = {}
        for ram in self.catalog.get("記憶體") or []:
            t = str(ram.specs.get("type", "")).strip()
            self._ram_by_type.setdefault(t, []).append(ram)

    def _repair(self, b: Build) -> Build:
        """把個體修成合法的組合（就地修改並回傳）。

        1. 散熱器剛好一個：同時有風冷和水冷時隨機留一個；都沒有時依 _pick_cooling 補一個。
        2. 必選類別缺漏時補上（記憶體走 pick_ram，其他類別隨機挑）；目錄該類別是空的就留空，
           交給 fitness 的硬約束處理。
        3. 以 CPU 為準修相容性：主機板腳位不符 → 換成同腳位的主機板（優先挑支援目前記憶體世代的）；
           記憶體世代不符主機板 → 換成相符世代、且在預算比例內的記憶體；
           散熱器不支援 CPU 腳位 → 換成同類別（風冷/水冷）有支援的散熱器；
           機殼放不下主機板板型 → 換成放得下的機殼（優先挑顯卡、風冷高度、水冷排也放得下的）。
           換主機板時也優先挑放得進目前機殼的，減少連帶換機殼。
        找不到相容候選時不硬改，保留原狀，讓相容性懲罰去扣分。
        """
        # 1. 散熱器剛好一個
        coolers = [c for c in COOLING_CATS if c in b.parts]
        if len(coolers) > 1:
            # 使用者指定了風冷或水冷就留指定的那個，auto 才隨機
            keep = (self.cooling_prefer if self.cooling_prefer in coolers
                    else random.choice(coolers))
            for c in coolers:
                if c != keep:
                    del b.parts[c]
        elif not coolers:
            cat = self._pick_cooling()
            pool = self.catalog.get(cat)
            if pool:
                b.parts[cat] = random.choice(pool)

        # 2. 必選類別補齊
        for cat in REQUIRED_CATS:
            if cat in b.parts:
                continue
            if cat == "記憶體":
                ram = pick_ram(self.catalog, b, self.budget)
                if ram:
                    b.parts["記憶體"] = ram
            else:
                pool = self.catalog.get(cat)
                if pool:
                    b.parts[cat] = random.choice(pool)

        cpu = b.parts.get("CPU")
        if not cpu:
            return b
        cpu_sock = str(cpu.specs.get("socket", "")).strip()

        # 3a. CPU ↔ 主機板腳位
        mb = b.parts.get("主機板")
        if cpu_sock and mb and str(mb.specs.get("socket", "")).strip() != cpu_sock:
            cands = self._mb_by_socket.get(cpu_sock, [])
            ram = b.parts.get("記憶體")
            ram_type = str(ram.specs.get("type", "")).strip() if ram else ""
            same_mem = [m for m in cands if ram_type and ram_type in str(m.specs.get("memory_type", ""))]
            case = b.parts.get("機殼")
            # 優先順序：記憶體世代相符且放得進機殼 > 記憶體世代相符 > 同腳位
            fits_case = [m for m in same_mem if form_factor_fits(m, case)] if case else []
            pick_from = fits_case or same_mem or cands
            if pick_from:
                b.parts["主機板"] = random.choice(pick_from)

        # 3b. 記憶體 ↔ 主機板世代
        mb = b.parts.get("主機板")
        ram = b.parts.get("記憶體")
        if mb and ram:
            mb_mem = str(mb.specs.get("memory_type", "")).strip()
            ram_type = str(ram.specs.get("type", "")).strip()
            if mb_mem and ram_type and ram_type not in mb_mem:
                cands = [r for t, rs in self._ram_by_type.items() if t and t in mb_mem for r in rs]
                if cands:
                    # 沿用 pick_ram 的挑法（優先 32GB、預算比例、雙通道偏好），只是候選限定為相符世代
                    new_ram = pick_ram(_PoolView({"記憶體": cands}), b, self.budget)
                    if new_ram:
                        b.parts["記憶體"] = new_ram

        # 3c. 散熱器 ↔ CPU 腳位
        for cat in COOLING_CATS:
            cooler = b.parts.get(cat)
            if cooler is None:
                continue
            socks = self._sockets(cooler)
            if cpu_sock and socks and cpu_sock not in socks:
                cands = [c for c in (self.catalog.get(cat) or []) if cpu_sock in self._sockets(c)]
                if cands:
                    b.parts[cat] = random.choice(cands)

        # 3e. 顯卡 16-pin ↔ 電源原生接頭（使用者不接受轉接線時）：換成有原生接頭、
        #     瓦數不低於原本的電源裡最小的那一級，維持原本的供電餘裕、不多花錢
        psu = b.parts.get("電源")
        if (not self.allow_16pin_adapter and needs_16pin(b.parts.get("GPU"))
                and psu and not has_native_16pin(psu)):
            native = [p for p in (self.catalog.get("電源") or []) if has_native_16pin(p)]
            cur_w = self._watt(psu)
            enough = [p for p in native if self._watt(p) >= cur_w]
            if enough:
                lowest = min(self._watt(p) for p in enough)
                b.parts["電源"] = random.choice([p for p in enough if self._watt(p) == lowest])
            elif native:
                highest = max(self._watt(p) for p in native)
                b.parts["電源"] = random.choice([p for p in native if self._watt(p) == highest])

        # 3d. 機殼 ↔ 主機板板型（主機板已經跟著 CPU 定了，所以換機殼）
        mb, case = b.parts.get("主機板"), b.parts.get("機殼")
        if mb and case and not form_factor_fits(mb, case):
            cands = [c for c in (self.catalog.get("機殼") or []) if form_factor_fits(mb, c)]
            roomy = [c for c in cands if self._case_fits_others(c, b)]
            pick_from = roomy or cands
            if pick_from:
                b.parts["機殼"] = random.choice(pick_from)
        return b

    @staticmethod
    def _watt(psu) -> float:
        try:
            return float(psu.specs.get("wattage", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _case_fits_others(case, b: Build) -> bool:
        """機殼的顯卡長度、風冷高度、水冷排長度限制，是否放得下目前的零件（換機殼時的偏好，不是硬性條件）"""
        def num(part, key):
            try:
                return float(part.specs.get(key, 0) or 0) if part else 0.0
            except (ValueError, TypeError):
                return 0.0
        gpu_len, max_gpu = num(b.parts.get("GPU"), "length_mm"), num(case, "max_gpu_mm")
        if gpu_len and max_gpu and gpu_len > max_gpu:
            return False
        cool_h, max_cool = num(b.parts.get("風冷"), "height_mm"), num(case, "max_cooler_mm")
        if cool_h and max_cool and cool_h > max_cool:
            return False
        rad, max_rad = num(b.parts.get("水冷"), "radiator_size"), num(case, "max_water_mm")
        if rad and max_rad and rad > max_rad:
            return False
        return True

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
        return self._repair(b) if self.repair else b

    def _pick_mutation_rate(self) -> float:
        """依預算決定突變率（未手動指定 mutation_rate 時使用）。

        依據：多種子（6 seeds）穩健性測試，遊戲情境下對 40000/45000/50000/
        60000/80000/100000 六個預算點各跑 baseline(0.30) vs 降突變率(0.15)
        比較。結果：
          - 40000：0.15 輸多贏少（2/6），且波動大（std 0.02~0.03）
          - 45000/50000：0.15 勝率贏（5/6）但分數本身波動仍大，不夠穩定
          - 60000 以上：0.15 六種子全勝（6/6），標準差只有 0.0004~0.002
        預算越緊，可行解空間越窄、越需要較有破壞力的突變才跳得出局部最優；
        預算寬鬆時，低突變率能穩定微調、不打散已經找到的好解。
        45000~50000 這段還不夠穩定，所以分界點保守抓在 60000，不是實測到
        的最低翻轉點——原始數據在 docs/CONVERGENCE_ANALYSIS.md 跟對話紀錄裡。
        只驗證過 usage=遊戲，其他 usage 邏輯上應該同理但還沒實測。
        """
        return 0.15 if self.budget >= 60000 else 0.30

    def _pick_cooling(self) -> str:
        if self.cooling_prefer in ("風冷", "水冷"):
            return self.cooling_prefer
        if len(self.cooling_available) == 1:
            return self.cooling_available[0]     # 例如水冷關掉時一律風冷
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

        # 硬約束：缺少必選零件或散熱器（目錄某類別是空的、修補也補不回來時才會發生）
        missing = sum(1 for c in REQUIRED_CATS if c not in build.parts)
        if not any(c in build.parts for c in COOLING_CATS):
            missing += 1
        if missing:
            return -20.0 - missing

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
            from data.cooler_capacity import get_cooler_capacity
            nominal = float(cpu_part.specs.get("tdp", 0) or 0)
            cpu_pwr = get_actual_power(cpu_part.name, fallback_tdp=nominal or 65)
            cool_cap = get_cooler_capacity(cooler_part.name, cooler_part.specs)
            if cpu_pwr > cool_cap:
                # 壓不住：硬約束淘汰，差距越大排越後
                return -5.0 - (cpu_pwr - cool_cap) / 100

        perf     = self._perf_score(build)
        sent     = self._sentiment_score(build)
        cp       = self._cp_score(build)
        b_pen    = self._budget_penalty(build)
        c_pen, _ = self.checker.check(build, self.psu_tier, self.allow_16pin_adapter)

        # 使用者有指定面向時，口碑的權重分一部分給「指定面向」這一項。
        # 舊做法是把指定的面向混進該零件的口碑平均，再跟其他類別平均，最好與最差的
        # 型號在總分上只差 0.04，比任何一條規則扣分都小，實測有一半的情況選擇不變。
        # 獨立成項後同樣的差距是 w_sent × pref_term_share（預設約 0.12）。
        share = self.pref_term_share
        score = (
            w["w_perf"]   * perf
          + w["w_sent"]   * ((1 - share) * sent + share * self._pref_score(build))
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
            score -= self._cooler_price_penalty(cpu, cooler)

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

    @staticmethod
    def _cooler_price_penalty(cpu, cooler) -> float:
        """散熱器價格合理性：低功耗 CPU 配太貴的散熱器要扣分。

        用實際功耗分級（審查 #10）。原本用標稱 TDP：i7-14700F 標稱 65W、實際 219W，
        被當成低功耗 CPU，配好一點的散熱器反而扣分。
        分界 90W / 165W 對應原本的 65W / 125W 級距：AMD 每顆都維持原級距
        （65W 級實際 ≤ 88W、105~120W 級實際 ≤ 162W），只有標稱值偏低的 Intel 被改到正確級距。
        """
        from data.cpu_power import get_actual_power
        nominal   = float(cpu.specs.get("tdp", 65) or 65)
        cpu_power = get_actual_power(cpu.name, fallback_tdp=nominal)
        if cpu_power <= 90:
            limit = 1200
        elif cpu_power <= 165:
            limit = 2000
        else:
            limit = 3000
        return max(0.0, (cooler.price - limit) / 60000 * 0.3)

    def _perf_score(self, build: Build) -> float:
        pw = self.weights["perf_weights"]
        score = total_w = 0.0
        for cat, part in build.parts.items():
            cat_key = "散熱" if cat in COOLING_CATS else cat
            w = pw.get(cat_key, pw.get(cat, 0.0))
            try:
                benchmark = float(part.specs.get("benchmark", 0) or 0)
            except (ValueError, TypeError):
                benchmark = 0.0

            if benchmark > 0 and cat in self._bench_range:
                bmin, bmax = self._bench_range[cat]
                norm = 1.0 if bmax <= bmin else (benchmark - bmin) / (bmax - bmin)
            else:
                pmin, pmax = self._price_range.get(cat, (0.0, 0.0))
                if pmax <= pmin:
                    norm = 1.0
                else:
                    norm = (part.price - pmin) / (pmax - pmin)
            norm = max(0.0, min(norm, 1.0))
            score   += w * norm
            total_w += w
        return score / total_w if total_w > 0 else 0.0

    def _sentiment_score(self, build: Build) -> float:
        # 指定面向已經獨立成項時，這裡只算一般口碑（各面向平均），避免同一個偏好算兩次
        blend = self.pref_term_share == 0
        scores = [self.scorer.get(cat, part.short_name,
                                  aspects=self.aspect_prefs.get(cat) if blend else None,
                                  pref_share=self.pref_share)
                  for cat, part in build.parts.items()
                  if not self._sent_cats or cat in self._sent_cats]
        return float(np.mean(scores)) if scores else 0.5

    def _pref_score(self, build: Build) -> float:
        """使用者指定的面向上，這組配置的零件表現如何（0~1，各指定面向等權平均）。
        分數是類別內正規化後的面向分數；評價不足的型號拿類別平均的位置，不加分也不扣分。"""
        if not self.pref_term_share:
            return 0.0
        scores = []
        for cat, aspects in self.aspect_prefs.items():
            part = build.parts.get(cat)
            if part is None:
                continue
            table = self.scorer.get_aspects(cat, part.short_name)
            scores += [table[a] for a in aspects if a in table]
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
        # 「風冷」「水冷」視為同一個基因（散熱器），整組交換，
        # 避免一邊拿到風冷、另一邊拿到水冷而變成同時有兩個散熱器。
        # 排序：set 的走訪順序會隨 Python 每次啟動的雜湊種子改變，不排序的話
        # 就算固定 random.seed，同樣的輸入每次跑出來的結果也不一樣
        genes = [[c] for c in sorted(set(p1.parts) | set(p2.parts)) if c not in COOLING_CATS]
        genes.append(list(COOLING_CATS))
        for gene in genes:
            if random.random() < 0.5:
                for cat in gene:
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
        if self.repair:
            # 每個子代都會經過突變，所以在這裡修補就涵蓋交叉產生的不合法組合
            return self._repair(b)
        has_cooling = any(c in b.parts for c in COOLING_CATS)
        if not has_cooling:
            new_cool = self._pick_cooling()
            pool = self.catalog.get(new_cool)
            if pool:
                b.parts[new_cool] = random.choice(pool)
        return b

    @staticmethod
    def _signature(build: Build) -> tuple:
        """配置的識別碼：各類別的商品名稱。名稱相同就是使用者眼中的同一台"""
        return tuple(sorted((cat, part.name) for cat, part in build.parts.items()))

    def _remember(self, pop: list[Build], fits: list[float], idx: list[int], keep: int = 50):
        """把這一代前 keep 名記進名人堂（同一組配置只記一次，fitness 是確定值）"""
        for i in idx[:keep]:
            sig = self._signature(pop[i])
            if sig not in self._hall:
                self._hall[sig] = (fits[i], pop[i].copy())

    def _pick_top(self, n: int = 5) -> list[Build]:
        """從名人堂挑前 n 名，保證彼此不是同一組配置。
        只從沒被硬約束淘汰的配置裡挑（缺零件、超支 20% 以上、散熱壓不住的 fitness ≤ -5）；
        全部都被淘汰時（例如預算低於最便宜的組合）才退回全部。
        先挑顯卡不同的（維持原本「前五名盡量換不同顯卡」的用意），不夠再用顯卡相同但其他零件不同的補。"""
        ranked = sorted(self._hall.items(), key=lambda kv: (-kv[1][0], kv[0]))
        feasible = [kv for kv in ranked if kv[1][0] > HARD_FAIL]
        ranked = feasible or ranked
        picked, seen_gpu = [], set()
        for sig, (fit, b) in ranked:
            gpu = b.parts.get("GPU")
            key = gpu.specs.get("ptt_model", gpu.name) if gpu else ""
            if key not in seen_gpu and len(picked) < n:
                seen_gpu.add(key)
                picked.append((fit, sig, b))
        used = {sig for _, sig, _ in picked}
        for sig, (fit, b) in ranked:
            if len(picked) >= n:
                break
            if sig not in used:
                picked.append((fit, sig, b))
                used.add(sig)
        # 依 fitness 排名次：顯卡不同只影響「挑哪幾組」，不影響順序（第 1 名永遠是最高分）
        picked.sort(key=lambda x: (-x[0], x[1]))
        return [b for _, _, b in picked]

    def run(self, verbose: bool = True) -> list[Build]:
        pop = self._init_population()
        # 名人堂：整個演化過程出現過的好配置。族群收斂後最後一代常常只剩一兩種配置，
        # 原本直接從最後一代挑前五名，會挑到同一組配置重複 5 次（80,000 遊戲實測 5 個全一樣）
        self._hall: dict[tuple, tuple[float, Build]] = {}
        for gen in range(self.generations):
            fits = [self.fitness(b) for b in pop]
            idx  = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
            self._remember(pop, fits, idx)
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
        self._remember(pop, fits, idx)
        return self._pick_top(5)
