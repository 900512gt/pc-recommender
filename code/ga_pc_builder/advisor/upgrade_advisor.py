"""
升級建議系統：分析當前配置並提供升級選項
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from data.catalog import Part, PartCatalog
from data.sentiment import SentimentScorer
from core.ga_engine import Build, PRICE_CEILING


@dataclass
class UpgradeOption:
    """升級選項"""
    category: str              # 零件類別
    current_part: Part         # 目前零件
    upgrade_part: Part         # 升級後零件
    price_increase: int        # 價格增加
    performance_boost: float   # 性能提升（基於情感評分）
    reason: str               # 升級理由


class UpgradeAdvisor:
    """升級建議引擎"""
    
    def __init__(self, 
                 catalog: PartCatalog,
                 scorer: SentimentScorer,
                 checker=None):
        self.catalog = catalog
        self.scorer = scorer
        self.checker = checker
    
    def get_upgrade_options(self,
                           build: Build,
                           budget: int,
                           remaining_budget: int,
                           usage: str = "遊戲",
                           max_options: int = 3) -> Dict[str, List[UpgradeOption]]:
        """
        分析配置並提供升級選項
        
        Args:
            build: 當前配置
            budget: 總預算
            remaining_budget: 剩餘預算
            usage: 使用目的（遊戲/工作等）
            max_options: 每個類別最多顯示幾個升級選項
        
        Returns:
            {類別: [升級選項列表]}
        """
        upgrades = {}
        
        # 優先順序：根據用途決定（遊戲優先升級GPU，工作優先升級CPU/RAM）
        if usage == "遊戲":
            priority = ["GPU", "CPU", "記憶體", "SSD", "主機板", "電源"]
        elif usage == "工作":
            priority = ["CPU", "記憶體", "SSD", "GPU", "主機板", "電源"]
        else:
            priority = ["CPU", "記憶體", "GPU", "SSD", "主機板", "電源"]
        
        for cat in priority:
            current = build.parts.get(cat)
            if not current:
                continue
            
            options = self._find_upgrades_for_category(
                cat, current, remaining_budget, build, max_options
            )
            if options:
                upgrades[cat] = options
        
        return upgrades
    
    def _find_upgrades_for_category(self,
                                    category: str,
                                    current: Part,
                                    remaining_budget: int,
                                    current_build: Build = None,
                                    max_options: int = 3) -> List[UpgradeOption]:
        """找出某個類別的升級選項"""
        options = []
        candidates = self.catalog.get(category)
        
        if not candidates:
            return options
        
        # 按價格排序
        candidates_sorted = sorted(candidates, key=lambda p: p.price)
        
        # 找出比目前更貴但在預算內的零件
        current_idx = -1
        try:
            current_idx = candidates_sorted.index(current)
        except ValueError:
            # 如果找不到完全相同的零件，找價格接近的
            for i, p in enumerate(candidates_sorted):
                if p.name == current.name:
                    current_idx = i
                    break
        
        if current_idx == -1:
            # 無法找到當前零件，從價格最接近開始
            for i, p in enumerate(candidates_sorted):
                if p.price >= current.price:
                    current_idx = i - 1
                    break
            if current_idx == -1:
                current_idx = len(candidates_sorted) - 1
        
        # 蒐集升級選項（比目前更高端且在預算內）
        for i in range(current_idx + 1, len(candidates_sorted)):
            candidate = candidates_sorted[i]
            price_increase = candidate.price - current.price
            
            # 檢查預算
            if price_increase <= 0 or price_increase > remaining_budget:
                continue
            
            # 檢查相容性（重要！）
            if current_build and not self._is_compatible_upgrade(
                category, current, candidate, current_build
            ):
                continue
            
            # 計算性能提升（使用情感評分作為代理）
            current_sentiment = self.scorer.get(category, current.short_name)
            candidate_sentiment = self.scorer.get(category, candidate.short_name)
            perf_boost = (candidate_sentiment - current_sentiment) * 100
            
            # 生成升級理由
            reason = self._generate_upgrade_reason(category, current, candidate, perf_boost)
            
            option = UpgradeOption(
                category=category,
                current_part=current,
                upgrade_part=candidate,
                price_increase=price_increase,
                performance_boost=perf_boost,
                reason=reason
            )
            options.append(option)
            
            if len(options) >= max_options:
                break
        
        return options
    
    def _is_compatible_upgrade(self, 
                               category: str,
                               current: Part,
                               upgrade: Part,
                               build: Build) -> bool:
        """檢查升級零件與當前配置是否相容"""
        
        # 主機板升級時：檢查新主機板是否與當前 CPU Socket 相同
        if category == "主機板":
            cpu = build.parts.get("CPU")
            if cpu:
                cpu_socket = str(cpu.specs.get("socket", "")).strip()
                mb_socket = str(upgrade.specs.get("socket", "")).strip()
                if cpu_socket and mb_socket and cpu_socket != mb_socket:
                    return False
        
        # CPU 升級時：檢查新 CPU 是否與當前主機板 Socket 相同
        elif category == "CPU":
            mb = build.parts.get("主機板")
            if mb:
                mb_socket = str(mb.specs.get("socket", "")).strip()
                cpu_socket = str(upgrade.specs.get("socket", "")).strip()
                if mb_socket and cpu_socket and mb_socket != cpu_socket:
                    return False
        
        # 記憶體升級時：檢查新記憶體是否與當前主機板相容
        elif category == "記憶體":
            mb = build.parts.get("主機板")
            if mb:
                mb_mem_type = str(mb.specs.get("memory_type", "")).strip()
                ram_type = str(upgrade.specs.get("type", "")).strip()
                if mb_mem_type and ram_type and ram_type not in mb_mem_type:
                    return False
        
        # 散熱器升級時：檢查 Socket 相容性
        elif category in ["風冷", "水冷"]:
            cpu = build.parts.get("CPU")
            if cpu:
                cpu_socket = str(cpu.specs.get("socket", "")).strip()
                supported = upgrade.specs.get("supported_sockets", [])
                if isinstance(supported, str):
                    try:
                        import json
                        supported = json.loads(supported)
                    except:
                        supported = [supported]
                if cpu_socket and supported and cpu_socket not in supported:
                    return False
        
        return True
    
    def _generate_upgrade_reason(self,
                                category: str,
                                current: Part,
                                upgrade: Part,
                                perf_boost: float) -> str:
        """生成升級理由"""
        if perf_boost > 10:
            quality_desc = "顯著提升"
        elif perf_boost > 5:
            quality_desc = "中等提升"
        else:
            quality_desc = "小幅提升"
        
        # 根據類別生成具體理由
        if category == "GPU":
            return f"遊戲性能{quality_desc}，支援更高特效設定"
        elif category == "CPU":
            return f"多核/多線程性能{quality_desc}，提升工作效率"
        elif category == "記憶體":
            return f"容量/頻率{quality_desc}，改善多工能力"
        elif category == "SSD":
            return f"讀寫速度{quality_desc}，加快系統響應"
        elif category == "主機板":
            return f"功能/供電{quality_desc}，提升穩定性與超頻潛力"
        elif category == "電源":
            return f"轉換效率{quality_desc}，降低功耗與發熱"
        else:
            return f"性能{quality_desc}"
    
    def get_smart_recommendations(self,
                                 build: Build,
                                 budget: int,
                                 remaining_budget: int,
                                 usage: str = "遊戲") -> List[Dict]:
        """
        智能升級推薦：根據剩餘預算推薦最佳升級組合
        
        Returns:
            [
                {
                    "priority": 1,
                    "category": "GPU",
                    "reason": "遊戲體驗提升最顯著",
                    "current": Part,
                    "upgrade": Part,
                    "cost": 2000,
                    "benefit": "顯著提升遊戲FPS"
                },
                ...
            ]
        """
        all_upgrades = self.get_upgrade_options(build, budget, remaining_budget, usage)
        
        recommendations = []
        priority = 1
        total_cost = 0
        
        for category, options in all_upgrades.items():
            if not options:
                continue
            
            # 選擇該類別中最值得升級的選項（性能提升/價格比最高）
            best_option = max(
                options,
                key=lambda x: (x.performance_boost / max(x.price_increase, 1))
                if x.price_increase > 0 else 0
            )
            
            # 檢查總預算約束
            if total_cost + best_option.price_increase > remaining_budget:
                continue
            
            # 生成推薦
            benefit = self._get_benefit_description(category, best_option.performance_boost)
            
            recommendations.append({
                "priority": priority,
                "category": category,
                "reason": best_option.reason,
                "current": best_option.current_part,
                "upgrade": best_option.upgrade_part,
                "cost": best_option.price_increase,
                "benefit": benefit,
                "sentiment_improvement": f"{best_option.performance_boost:+.2f}%"
            })
            
            priority += 1
            total_cost += best_option.price_increase
        
        return recommendations
    
    def _get_benefit_description(self, category: str, perf_boost: float) -> str:
        """根據類別和性能提升獲取效益描述"""
        if category == "GPU":
            if perf_boost > 15:
                return "遊戲FPS提升 20-40%"
            elif perf_boost > 10:
                return "遊戲FPS提升 10-20%"
            else:
                return "遊戲FPS提升 5-10%"
        elif category == "CPU":
            if perf_boost > 15:
                return "多工性能提升 25%+"
            elif perf_boost > 10:
                return "多工性能提升 15-25%"
            else:
                return "多工性能提升 5-15%"
        elif category == "記憶體":
            if perf_boost > 10:
                return "系統響應速度提升，支援更多同時應用"
            else:
                return "改善多工能力"
        elif category == "SSD":
            return "系統啟動與應用加載更快"
        elif category == "主機板":
            return "提升系統穩定性與超頻潛力"
        elif category == "電源":
            return "降低功耗，延長硬體壽命"
        else:
            return "性能提升"
