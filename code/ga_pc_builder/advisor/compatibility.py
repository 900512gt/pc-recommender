"""
相容性檢查：利用 ga_database_v2.json 的規格欄位做硬性約束
"""
import json


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
        if case and mb:
            mb_form   = str(mb.specs.get("form_factor", "")).strip()
            supported = str(case.specs.get("supported_mb", "")).strip()
            if mb_form and supported and mb_form not in supported:
                penalty += 0.5
                issues.append(f"機殼不支援 {mb_form} 主機板")

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
