"""
相容性檢查：利用 ga_database_v2.json 的規格欄位做硬性約束
"""
import json


# 主機板板型的別名。資料庫裡主機板寫「Mini-ITX」，機殼的 supported_mb 寫「ITX」，
# 要先統一成同一種寫法才能比對。
_FORM_ALIASES = {
    "MINI-ITX": "ITX", "MINIITX": "ITX", "ITX": "ITX",
    "M-ATX": "M-ATX", "MATX": "M-ATX", "MICRO-ATX": "M-ATX", "MICROATX": "M-ATX",
    "ATX": "ATX",
    "E-ATX": "E-ATX", "EATX": "E-ATX",
}


def normalize_form_factor(value) -> str:
    """把板型字串統一成 ITX / M-ATX / ATX / E-ATX（認不得的就原樣轉大寫回傳）"""
    key = str(value or "").strip().upper().replace(" ", "").replace("_", "-")
    return _FORM_ALIASES.get(key, key)


def supported_form_factors(case) -> set:
    """機殼支援的主機板板型集合。supported_mb 可以是 list、JSON 字串或逗號分隔字串。"""
    raw = case.specs.get("supported_mb", []) if case else []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = raw.replace("/", ",").split(",")
    if raw is None:
        raw = []
    elif not isinstance(raw, (list, tuple, set)):
        raw = [raw]          # 單一字串、數字等奇怪型別：當成只有一項
    return {normalize_form_factor(x) for x in raw if str(x).strip()}


# 主機板尺寸由小到大。主機板只能裝進同尺寸或更大的機殼（ATX 板放 E-ATX 機殼可以，反過來不行）。
FORM_SIZE_ORDER = ["ITX", "M-ATX", "ATX", "E-ATX"]
_FORM_RANK = {f: i for i, f in enumerate(FORM_SIZE_ORDER)}


def case_max_form_factor(case) -> str | None:
    """機殼最大能裝的主機板尺寸。

    看兩個來源：機殼的分類（series，例如 ATX / E-ATX）和 supported_mb 裡最大的尺寸。
    兩個都有時取比較小的那個——資料庫有 ATX 機殼的 supported_mb 被寫成連 E-ATX 都支援，
    但分類是 ATX（商品名也寫「限ATX」），取小的才不會讓 E-ATX 主機板被判放得進去。
    分類不是尺寸（例如「靜音」）時只看 supported_mb；兩邊都沒有就回傳 None。
    """
    sizes = []
    series = normalize_form_factor(case.specs.get("series", "")) if case else ""
    if series in _FORM_RANK:
        sizes.append(series)
    listed = [f for f in supported_form_factors(case) if f in _FORM_RANK]
    if listed:
        sizes.append(max(listed, key=_FORM_RANK.get))
    return min(sizes, key=_FORM_RANK.get) if sizes else None


def form_factor_fits(mb, case) -> bool:
    """主機板放得進這個機殼嗎？規則：主機板尺寸 ≤ 機殼最大能裝的尺寸。
    任一邊沒有資料時視為放得進（不亂扣分）。

    不用字串包含：舊寫法 `"ATX" in "['ITX', 'M-ATX']"` 會因為「M-ATX」裡有「ATX」而誤判相容，
    `"Mini-ITX" in "['ITX']"` 則永遠不成立。
    """
    form = normalize_form_factor(mb.specs.get("form_factor", "")) if mb else ""
    if not form:
        return True
    if form not in _FORM_RANK:
        # 認不得的尺寸（例如 SSI-EEB）：沒辦法比大小，只看機殼有沒有明確列出
        supported = supported_form_factors(case)
        return not supported or form in supported
    max_form = case_max_form_factor(case)
    if max_form is None:
        return True
    return _FORM_RANK[form] <= _FORM_RANK[max_form]


def needs_16pin(gpu) -> bool:
    """顯卡用 16-pin（12VHPWR / 12V-2x6）供電。資料庫 power_conn 是 16-pin 或 1*16pin"""
    return bool(gpu) and "16" in str(gpu.specs.get("power_conn", ""))


def has_native_16pin(psu) -> bool:
    """電源有原生 16-pin 接頭（資料庫 pcie5 = "是"，即 ATX 3.x / PCIe 5.x 電源）"""
    return bool(psu) and str(psu.specs.get("pcie5", "")).strip() == "是"


class CompatibilityChecker:

    def check(self, build, psu_tier: str = "standard",
              allow_16pin_adapter: bool = False) -> tuple[float, list[str]]:
        """allow_16pin_adapter：使用者是否接受 16-pin 顯卡用轉接線接在沒有原生 16-pin 的電源上。
        接受時只提示、不扣分；不接受時當成不相容扣分。"""
        penalty = 0.0
        issues = []

        cpu   = build.parts.get("CPU")
        mb    = build.parts.get("主機板")
        ram   = build.parts.get("記憶體")
        gpu   = build.parts.get("GPU")
        case  = build.parts.get("機殼")
        psu   = build.parts.get("電源")
        air   = build.parts.get("風冷")
        water = build.parts.get("水冷")

        # 不能同時有風冷和水冷
        if air and water:
            penalty += 2.0
            issues.append("同時裝了風冷和水冷散熱器")

        # CPU ↔ 主機板 腳位
        if cpu and mb:
            cpu_socket = str(cpu.specs.get("socket", "")).strip()
            mb_socket  = str(mb.specs.get("socket", "")).strip()
            if cpu_socket and mb_socket and cpu_socket != mb_socket:
                penalty += 1.0
                issues.append(f"CPU 腳位 {cpu_socket} ≠ 主機板 {mb_socket}")

        # 記憶體 ↔ 主機板 DDR 規格
        if ram and mb:
            ram_type = str(ram.specs.get("type", "")).strip()
            mb_mem   = str(mb.specs.get("memory_type", "")).strip()
            if ram_type and mb_mem and ram_type not in mb_mem:
                penalty += 0.8
                issues.append(f"記憶體 {ram_type} 與主機板 {mb_mem} 不符")

        # 機殼 ↔ 主機板 板型
        if case and mb and not form_factor_fits(mb, case):
            penalty += 0.5
            issues.append(f"機殼不支援 {mb.specs.get('form_factor', '')} 主機板")

        # 機殼 ↔ GPU 長度
        if case and gpu:
            try:
                gpu_len  = float(gpu.specs.get("length_mm", 0) or 0)
                case_max = float(case.specs.get("max_gpu_mm", 999) or 999)
                if gpu_len > 0 and case_max > 0 and gpu_len > case_max:
                    penalty += 0.5
                    issues.append(f"GPU {gpu_len}mm > 機殼限制 {case_max}mm")
            except (ValueError, TypeError):
                pass

        # 散熱器相容性
        cooler = air or water
        if cpu and cooler:
            # 用「實際功耗」對比「推斷壓制力」，兩者都不靠不可靠的標稱值
            from data.cpu_power import get_actual_power
            from data.cooler_capacity import get_cooler_capacity
            try:
                nominal_tdp = float(cpu.specs.get("tdp", 0) or 0)
                cpu_power   = get_actual_power(cpu.name, fallback_tdp=nominal_tdp or 65)
                cooler_cap  = get_cooler_capacity(cooler.name, cooler.specs)

                if cpu_power > cooler_cap:
                    # 壓不住：差距越大罰越重
                    gap = cpu_power - cooler_cap
                    penalty += 1.5 + min(gap / 100, 1.5)
                    issues.append(
                        f"散熱不足：{cpu.name[:18]} 實際功耗約 {cpu_power:.0f}W，"
                        f"但散熱器只壓得住約 {cooler_cap:.0f}W"
                    )
                elif cpu_power > cooler_cap * 0.9:
                    penalty += 0.5
                    issues.append(
                        f"散熱餘裕不足：CPU 約 {cpu_power:.0f}W，散熱器約 {cooler_cap:.0f}W"
                    )
            except (ValueError, TypeError):
                pass

            # 腳位檢查
            cpu_socket = str(cpu.specs.get("socket", "")).strip()
            supported  = cooler.specs.get("supported_sockets", [])
            if isinstance(supported, str):
                try:
                    supported = json.loads(supported)
                except Exception:
                    supported = [supported]
            if cpu_socket and supported and cpu_socket not in supported:
                penalty += 1.5
                issues.append(f"散熱器不支援 CPU 腳位 {cpu_socket}，支援：{supported}")

            # 高度 ↔ 機殼
            if case:
                try:
                    cooler_h = float(cooler.specs.get("height_mm", 0) or 0)
                    case_max = float(case.specs.get("max_cooler_mm", 0) or 0)
                    if cooler_h > 0 and case_max > 0 and cooler_h > case_max:
                        penalty += 1.0
                        issues.append(f"散熱器高度 {cooler_h}mm > 機殼限制 {case_max}mm")
                except (ValueError, TypeError):
                    pass

        # 機殼 ↔ 水冷排尺寸
        if case and water:
            try:
                rad_size = float(water.specs.get("radiator_size", 0) or 0)
                case_max = float(case.specs.get("max_water_mm", 0) or 0)
                if rad_size > 0 and case_max > 0 and rad_size > case_max:
                    penalty += 0.3
                    issues.append(f"水冷排 {rad_size}mm > 機殼限制 {case_max}mm")
            except (ValueError, TypeError):
                pass

        # 顯卡 16-pin ↔ 電源原生接頭（審查 #19）
        # 顯卡通常附 8-pin 轉 16-pin 轉接線，技術上接得起來；但轉接線曾有燒熔案例，
        # 廠商建議用原生 16-pin 的 ATX 3.x 電源。由使用者決定接不接受轉接線。
        psu_part = build.parts.get("電源")
        if needs_16pin(gpu) and psu_part and not has_native_16pin(psu_part):
            if allow_16pin_adapter:
                issues.append("顯卡用 16-pin 供電，這顆電源沒有原生接頭，需使用顯卡附的轉接線（已允許）")
            else:
                penalty += 1.0
                issues.append("顯卡需要 16-pin（12V-2x6）供電，電源沒有原生接頭")

        # 電源評分（夠用/過度配置/用料）已統一由 policies/psu_policy.py 負責，
        # 此處不再重複檢查，避免同一問題被兩個模組各扣一次分。

        return penalty, issues
