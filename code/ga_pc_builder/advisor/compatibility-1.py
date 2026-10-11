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
    if isinstance(raw, str):
        raw = [raw]
    return {normalize_form_factor(x) for x in (raw or []) if str(x).strip()}


def form_factor_fits(mb, case) -> bool:
    """主機板放得進這個機殼嗎？任一邊沒有資料時視為放得進（不亂扣分）。

    逐項比對集合，不用字串包含：舊寫法 `"ATX" in "['ITX', 'M-ATX']"` 會因為
    「M-ATX」裡有「ATX」而誤判相容，`"Mini-ITX" in "['ITX']"` 則永遠不成立。
    """
    form = normalize_form_factor(mb.specs.get("form_factor", "")) if mb else ""
    supported = supported_form_factors(case)
    if not form or not supported:
        return True
    return form in supported


class CompatibilityChecker:

    def check(self, build,psu_tier:str = "standard") -> tuple[float, list[str]]:
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
            from data.cooler_capacity import estimate_cooler_capacity
            try:
                nominal_tdp = float(cpu.specs.get("tdp", 0) or 0)
                cpu_power   = get_actual_power(cpu.name, fallback_tdp=nominal_tdp or 65)
                cooler_cap  = estimate_cooler_capacity(cooler.name, cooler.specs)

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

        # 電源評分（夠用/過度配置/用料）已統一由 policies/psu_policy.py 負責，
        # 此處不再重複檢查，避免同一問題被兩個模組各扣一次分。

        return penalty, issues
