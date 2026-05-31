"""
GA 演化過程視覺化
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False
 
 
def plot_fitness(ga, output_path: str = "fitness_history.png"):
    """畫出 GA 演化過程的 fitness 變化圖"""
    generations = list(range(len(ga.history_best)))
 
    plt.figure(figsize=(10, 5))
    plt.plot(generations, ga.history_best, label="最佳 Fitness", color="blue", linewidth=2)
    plt.plot(generations, ga.history_avg,  label="平均 Fitness", color="orange", alpha=0.7, linewidth=1.5)
 
    plt.xlabel("世代（Generation）")
    plt.ylabel("Fitness 分數")
    plt.title("GA 演化過程 - Fitness 收斂曲線")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] 演化圖已存至 {output_path}")
 
 
def plot_budget_breakdown(build, output_path: str = "budget_breakdown.png"):
    """畫出配置的預算分配圓餅圖"""
    labels = []
    prices = []
 
    for cat, part in sorted(build.parts.items()):
        labels.append(f"{cat}\n{part.price:,}")
        prices.append(part.price)
 
    plt.figure(figsize=(8, 8))
    plt.pie(prices, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title(f"預算分配（總價 NT${sum(prices):,}）")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] 預算分配圖已存至 {output_path}")