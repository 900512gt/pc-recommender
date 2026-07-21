


import pandas as pd
from config import DB_PATH, MATCHED_FILES
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer
from utils.plot_utils import plot_fitness, plot_budget_breakdown
from advisor.compatibility import CompatibilityChecker
from core.ga_engine import GARecommender, Build, COOLING_CATS
from advisor.upgrade_advisor import UpgradeAdvisor


def print_build(build: Build, scorer: SentimentScorer,
                checker: CompatibilityChecker, rank: int = 1):
    """顯示配置詳情"""
    penalty, issues = checker.check(build)
    print(f"\n{'='*60}")
    print(f"  第 {rank} 名推薦配置  |  總價：NT$ {build.total_price:,}")
    print(f"{'='*60}")
    for cat, part in sorted(build.parts.items()):
        sent = scorer.get(cat, part.short_name)
        print(f"  [{cat:5s}] {part.price:>6,}元  情感:{sent:.2f}  {part.name[:50]}")
    if issues:
        print(f"\n  ⚠️  相容性問題（懲罰 {penalty:.2f}）：")
        for i in issues:
            print(f"     • {i}")
    else:
        print(f"\n  ✅ 相容性：無問題")
 
 
def print_upgrade_recommendations(advisor: UpgradeAdvisor,
                                 build: Build,
                                 budget: int,
                                 remaining: int,
                                 usage: str = "遊戲"):
    """顯示智能升級建議"""
    if remaining <= 500:
        print(f"\n  預算已充分利用，升級空間有限")
        return
    
    print(f"\n  【根據您的預算和使用需求的升級建議】")
    print(f"  剩餘預算：NT${remaining:,} 可用於升級")
    print()
    
    recommendations = advisor.get_smart_recommendations(
        build, budget, remaining, usage
    )
    
    if not recommendations:
        print(f"  目前配置已是此預算最佳分配")
        return
    
    total_upgrade_cost = 0
    for rec in recommendations:
        total_upgrade_cost += rec["cost"]
        
        print(f"  【第 {rec['priority']} 優先】{rec['category']} 升級")
        print(f"    目前：{rec['current'].name[:45]}")
        print(f"           NT${rec['current'].price:,}")
        print(f"    升級：{rec['upgrade'].name[:45]}")
        print(f"           NT${rec['upgrade'].price:,} (增加 +NT${rec['cost']:,})")
        print(f"    效益：{rec['benefit']}")
        print(f"    理由：{rec['reason']}")
        print(f"    情感評分：{rec['sentiment_improvement']}")
        print()
    
    total_after = build.total_price + total_upgrade_cost
    print(f"  若全部升級：NT${build.total_price:,} → NT${total_after:,} "
          f"(增加 +NT${total_upgrade_cost:,})")
    print(f"  剩餘預算：NT${budget - total_after:,}")
    
    # 提供逐項選擇選項
    print(f"\n  您也可以選擇性地進行部分升級")
    for rec in recommendations:
        print(f"    • 升級 {rec['category']}：+NT${rec['cost']:,}")

 
 
def export_results(builds: list[Build], scorer: SentimentScorer,
                   checker: CompatibilityChecker,
                   output_path: str = "ga_results.xlsx"):
    rows = []
    for rank, build in enumerate(builds, 1):
        penalty, issues = checker.check(build)
        row = {
            "排名": rank,
            "總價(NT$)": build.total_price,
            "相容性懲罰": round(penalty, 3),
            "相容性問題": " | ".join(issues),
        }
        for cat, part in build.parts.items():
            sent = scorer.get(cat, part.short_name)
            row[f"{cat}_型號"]  = part.name[:60]
            row[f"{cat}_價格"]  = part.price
            row[f"{cat}_情感分"] = round(sent, 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)

    print(f"\n[Export] 結果已存至 {output_path}")

 
 
def get_user_input():
    # 預算
    while True:
        try:
            budget = int(input("\n請輸入預算（NT$）：").strip())
            if budget < 10000:
                print("預算至少需要 10,000 元")
                continue
            break
        except ValueError:
            print("請輸入數字")
 
    # 用途
    print("\n用途選擇：")
    print("1. 遊戲")
    print("2. 工作（開發中）")
    print("3. 一般文書（開發中）")
    while True:
        choice = input("請選擇（1/2/3）：").strip()
        if choice == "1":
            usage = "遊戲"
            break
        elif choice == "2":
            usage = "工作"
            break
        elif choice == "3":
            usage = "一般文書"
            break
        else:
            print("請輸入 1、2 或 3")
 
    # 散熱偏好
    print("\n散熱偏好：")
    print("1. 自動選擇")
    print("2. 風冷")
    print("3. 水冷")
    while True:
        choice = input("請選擇（1/2/3）：").strip()
        if choice == "1":
            cooling_prefer = "auto"
            break
        elif choice == "2":
            cooling_prefer = "風冷"
            break
        elif choice == "3":
            cooling_prefer = "水冷"
            break
        else:
            print("請輸入 1、2 或 3")

    # 預算策略
    print("\n預算策略：")
    print("1. 抓滿預算（在預算內配最強的配置）")
    print("2. 留有餘裕（先配主機，剩餘預算留給你自己決定升級）")
    while True:
        choice = input("請選擇（1/2）：").strip()
        if choice == "1":
            reserve = 0
            break
        elif choice == "2":
            # 讓使用者自己輸入要留多少
            while True:
                try:
                    reserve = int(input(
                        f"  你想保留多少升級空間？（NT$，預算 {budget:,}）："
                    ).strip())
                    if reserve < 0:
                        print("  請輸入正數")
                        continue
                    if reserve >= budget:
                        print(f"  保留金額不能超過預算 {budget:,}")
                        continue
                    if budget - reserve < 10000:
                        print(f"  保留 {reserve:,} 後主機預算只剩 {budget-reserve:,}，太少了")
                        continue
                    break
                except ValueError:
                    print("  請輸入數字")
            break
        else:
            print("請輸入 1 或 2")

    return budget, usage, cooling_prefer, reserve
 

 
 
def main():
    print("=" * 60)
    print("  GA 電腦組裝推薦系統")
    print("=" * 60)
 
    # 載入資料
    catalog = PartCatalog(DB_PATH)
    scorer  = SentimentScorer(MATCHED_FILES, DB_PATH)
    checker = CompatibilityChecker()
    advisor = UpgradeAdvisor(catalog, scorer, checker)
 
    # 使用者輸入
    budget, usage, cooling_prefer, reserve = get_user_input()
    psu_tier = "standard"

    # 留有餘裕：GA 用「預算 - 保留額」配主機，剩餘留給升級建議
    ga_budget = budget - reserve

    if reserve > 0:
        print(f"\n用途：{usage}  |  總預算：NT${budget:,}  |  散熱：{cooling_prefer}")
        print(f"主機預算：NT${ga_budget:,}（保留 NT${reserve:,} 作升級彈性）")
    else:
        print(f"\n用途：{usage}  |  預算：NT${budget:,}  |  散熱：{cooling_prefer}")

    # 執行 GA（用 ga_budget 配主機）
    ga = GARecommender(
        catalog=catalog,
        scorer=scorer,
        checker=checker,
        usage=usage,
        budget=ga_budget,
        pop_size=300,
        generations=300,
        elite_k=2,
        crossover_rate=0.8,
        mutation_rate=0.30,
        cooling_prefer=cooling_prefer,
        psu_tier=psu_tier,
    )
 
    print("\n[GA] 開始演化...")
    top5 = ga.run(verbose=True)
 
    print("\n\n【推薦結果 Top 5】")
    for i, build in enumerate(top5, 1):
        print_build(build, scorer, checker, rank=i)

    # 預算分析與升級建議
    best = top5[0]
    # 升級空間 = 原始總預算 - 實際花費（含被保留的餘裕）
    remaining = budget - best.total_price
    print(f"\n{'='*60}")
    print(f"【預算分析】")
    if reserve > 0:
        print(f"  主機配置：NT${best.total_price:,}（目標 NT${ga_budget:,}）")
        print(f"  總預算：NT${budget:,}  可升級空間：NT${remaining:,}")
        print(f"  （含你保留的 NT${reserve:,} + 主機未用完的部分）")
    else:
        print(f"  配置總價：NT${best.total_price:,}  剩餘預算：NT${remaining:,}")

    # 使用新的升級建議系統（用真正的剩餘空間）
    print_upgrade_recommendations(advisor, best, budget, remaining, usage)

    export_results(top5, scorer, checker, "ga_results.xlsx")
    plot_fitness(ga)
    plot_budget_breakdown(top5[0])
 
 
if __name__ == "__main__":
    main()
