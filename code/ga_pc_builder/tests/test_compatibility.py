#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試升級建議系統的相容性檢查
"""

from config import DB_PATH, MATCHED_FILES
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer
from advisor.upgrade_advisor import UpgradeAdvisor
from core.ga_engine import Build
from advisor.compatibility import CompatibilityChecker


def test_compatibility_check():
    """測試相容性檢查"""
    
    # 初始化
    catalog = PartCatalog(DB_PATH)
    scorer = SentimentScorer(MATCHED_FILES, DB_PATH)
    checker = CompatibilityChecker()
    advisor = UpgradeAdvisor(catalog, scorer, checker)
    
    # 構建一個測試配置（Intel + B860M）
    test_build = Build()
    
    # 添加 Intel CPU
    cpus = [p for p in catalog.get("CPU") if "265K" in p.name and "Core Ultra" in p.name]
    if cpus:
        test_build.parts["CPU"] = cpus[0]
        print(f"✓ CPU: {test_build.parts['CPU'].name[:50]}")
        print(f"  Socket: {test_build.parts['CPU'].specs.get('socket', 'N/A')}\n")
    
    # 添加 Intel B860 主機板
    mbs = [p for p in catalog.get("主機板") if "B860" in p.name]
    if mbs:
        test_build.parts["主機板"] = mbs[0]
        print(f"✓ 主機板: {test_build.parts['主機板'].name[:50]}")
        print(f"  Socket: {test_build.parts['主機板'].specs.get('socket', 'N/A')}\n")
    
    # 添加其他必要零件
    for cat in ["GPU", "記憶體", "SSD", "電源", "機殼"]:
        parts = catalog.get(cat)
        if parts:
            test_build.parts[cat] = parts[0]
    
    # 添加散熱器
    coolers = catalog.get("風冷")
    if coolers:
        test_build.parts["風冷"] = coolers[0]
    
    # 測試升級建議
    remaining_budget = 10000
    
    print("="*70)
    print("【升級建議相容性檢查測試】")
    print("="*70)
    print(f"\n當前配置：")
    print(f"  CPU: {test_build.parts['CPU'].name[:45]}")
    print(f"  主機板: {test_build.parts['主機板'].name[:45]}")
    print(f"\n剩餘預算: NT${remaining_budget:,}")
    print(f"\n測試升級選項...\n")
    
    # 獲取升級選項
    upgrades = advisor.get_upgrade_options(
        test_build,
        budget=60000,
        remaining_budget=remaining_budget,
        usage="工作",
        max_options=5
    )
    
    # 顯示結果
    if not upgrades:
        print("❌ 未找到升級選項（可能是相容性過濾導致）")
        return
    
    for category, options in upgrades.items():
        print(f"\n【{category} 升級選項】")
        if not options:
            print(f"  ❌ 無相容選項（可能被相容性過濾排除）")
        else:
            for i, opt in enumerate(options[:3], 1):
                print(f"  {i}. {opt.upgrade_part.name[:45]}")
                print(f"     價格: NT${opt.upgrade_part.price:,}")
                print(f"     增加: +NT${opt.price_increase:,}")
                print(f"     理由: {opt.reason}")
                
                # 驗證相容性
                if category == "主機板":
                    cpu_socket = str(test_build.parts['CPU'].specs.get('socket', ''))
                    mb_socket = str(opt.upgrade_part.specs.get('socket', ''))
                    if cpu_socket == mb_socket:
                        print(f"     ✓ Socket 相容: {cpu_socket}")
                    else:
                        print(f"     ❌ Socket 不相容: CPU={cpu_socket}, MB={mb_socket}")
    
    # 檢查是否有 AMD 主機板被正確過濾
    all_mbs = catalog.get("主機板")
    amd_mbs = [m for m in all_mbs if "B650" in m.name or "X670" in m.name]
    if amd_mbs and upgrades.get("主機板"):
        amd_in_upgrades = any("B650" in opt.upgrade_part.name or "X670" in opt.upgrade_part.name 
                             for opt in upgrades["主機板"])
        if not amd_in_upgrades:
            print(f"\n✅ 相容性檢查成功：")
            print(f"   Intel CPU (B860) 配置未推薦 AMD 主機板 (B650/X670)")
        else:
            print(f"\n❌ 相容性檢查失敗：")
            print(f"   Intel CPU 配置被推薦了 AMD 主機板")


if __name__ == "__main__":
    test_compatibility_check()
