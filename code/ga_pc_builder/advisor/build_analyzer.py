"""
配置快速分析：針對用戶提供的配置進行分析和升級建議
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ComponentInfo:
    """零件信息"""
    category: str
    price: int
    sentiment: float
    name: str
    description: str


def analyze_user_config(components_data: Dict[str, ComponentInfo]) -> Dict:
    """
    分析用戶提供的配置
    
    Returns:
        {
            'total_price': int,
            'sentiment_analysis': {category: sentiment_score},
            'upgrade_priorities': [...]
        }
    """
    total = sum(c.price for c in components_data.values())
    
    # 情感評分分析
    sentiments = {cat: c.sentiment for cat, c in components_data.items()}
    avg_sentiment = sum(sentiments.values()) / len(sentiments) if sentiments else 0
    
    # 找出最弱環節（情感評分最低的部分）
    weaknesses = sorted(sentiments.items(), key=lambda x: x[1])
    
    analysis = {
        'total_price': total,
        'components': components_data,
        'sentiments': sentiments,
        'avg_sentiment': avg_sentiment,
        'weaknesses': weaknesses,  # 優先級最低 → 最高
    }
    
    return analysis


def generate_upgrade_suggestions(analysis: Dict) -> List[str]:
    """生成升級建議"""
    suggestions = []
    
    # 基於情感評分的升級優先級
    weaknesses = analysis['weaknesses']
    avg_sentiment = analysis['avg_sentiment']
    
    for category, sentiment in weaknesses[:3]:  # 最弱的3個
        if sentiment < avg_sentiment - 0.05:
            comp = analysis['components'].get(category)
            if comp:
                suggestions.append(
                    f"【優先升級】{category}（目前情感評分 {sentiment:.2f}）\n"
                    f"    目前選擇：{comp.name}\n"
                    f"    價格：NT${comp.price:,}\n"
                    f"    理由：此部分評分偏低，升級空間大"
                )
    
    return suggestions


def print_config_analysis(config_dict: Dict[str, Dict]):
    """
    打印配置分析結果
    
    config_dict 格式:
    {
        'CPU': {'price': 10200, 'sentiment': 0.70, 'name': '...', 'description': '...'},
        'GPU': {'price': 9490, 'sentiment': 0.60, 'name': '...', 'description': '...'},
        ...
    }
    """
    components = {
        cat: ComponentInfo(
            category=cat,
            price=data['price'],
            sentiment=data['sentiment'],
            name=data.get('name', ''),
            description=data.get('description', '')
        )
        for cat, data in config_dict.items()
    }
    
    analysis = analyze_user_config(components)
    
    print("\n" + "="*70)
    print(f"  配置分析報告")
    print("="*70)
    
    print(f"\n【基本信息】")
    print(f"  總價格：NT$ {analysis['total_price']:,}")
    print(f"  平均情感評分：{analysis['avg_sentiment']:.2f}")
    
    print(f"\n【零件情感評分】")
    for cat, sentiment in sorted(analysis['sentiments'].items(), key=lambda x: x[1], reverse=True):
        comp = components[cat]
        bar = "█" * int(sentiment * 20) + "░" * (20 - int(sentiment * 20))
        print(f"  {cat:6s} {sentiment:.2f} [{bar}]  {comp.name[:40]}")
    
    print(f"\n【升級優先級分析】")
    weakest = analysis['weaknesses']
    for i, (category, sentiment) in enumerate(weakest[:3], 1):
        comp = components[category]
        diff = analysis['avg_sentiment'] - sentiment
        print(f"  第{i}優先：{category}")
        print(f"    目前情感評分：{sentiment:.2f}（比平均低 {diff:.2f}）")
        print(f"    目前產品：{comp.name[:50]}")
        print(f"    升級建議：尋找評分 > {sentiment + 0.10:.2f} 的替代品")
        print(f"    預期效果：整體體驗提升約 {diff * 100:.0f}%\n")
    
    print(f"【性價比評估】")
    for cat, comp in components.items():
        efficiency = comp.sentiment / (comp.price / 1000) if comp.price > 0 else 0
        print(f"  {cat:6s} NT${comp.price:6,}/分 = {efficiency:.4f} 分/仟元")
    
    print("\n" + "="*70)


# 示例使用：提供的配置
example_config = {
    'CPU': {
        'price': 10200,
        'sentiment': 0.70,
        'name': 'Intel Core Ultra 7 265K【20核】3.9G(↑5.5G) /30M /內顯Xe',
        'description': '高性能主流處理器'
    },
    'GPU': {
        'price': 9490,
        'sentiment': 0.60,
        'name': '藍寶石 白金版 PULSE RX7650GRE GAMING 8GB(2695MHz/24cm/雙風',
        'description': '工作用途GPU（性能偏弱）'
    },
    'SSD': {
        'price': 4999,
        'sentiment': 0.73,
        'name': '威剛 ADATA LEGEND 900 1TB/Gen4/讀7000/寫4700(單面設計)贈散熱片',
        'description': '高速存儲'
    },
    '主機板': {
        'price': 4190,
        'sentiment': 0.54,
        'name': '華擎 B860M Pro-A WiFi(M-ATX/LAN2.5G+無線/註四年)',
        'description': '入門級高端主機板'
    },
    '機殼': {
        'price': 2690,
        'sentiment': 0.67,
        'name': '喬思伯 D300(TW) 黑 顯卡長43/U高18/SL-120風扇*4/全景環形玻璃',
        'description': '中塔機殼'
    },
    '記憶體': {
        'price': 8888,
        'sentiment': 0.44,
        'name': 'UMAX 單條32GB DDR5-4800/CL40',
        'description': '大容量單條記憶體'
    },
    '電源': {
        'price': 3490,
        'sentiment': 0.80,
        'name': '全漢 VITA GM 750W(MIT) 白色版 雙8/金牌/全模/ATX3.1(PCIe 5.1)',
        'description': '高效能電源'
    },
    '風冷': {
        'price': 1690,
        'sentiment': 0.70,
        'name': '利民 Peerless Assassin 120 Digital ARGB 黑/6導管/雙塔雙扇',
        'description': '高效散熱器'
    },
}


if __name__ == "__main__":
    # 分析示例配置
    print_config_analysis(example_config)
    
    # 額外的升級建議
    components = {
        cat: ComponentInfo(
            category=cat,
            price=data['price'],
            sentiment=data['sentiment'],
            name=data.get('name', ''),
            description=data.get('description', '')
        )
        for cat, data in example_config.items()
    }
    
    analysis = analyze_user_config(components)
    suggestions = generate_upgrade_suggestions(analysis)
    
    if suggestions:
        print("\n【詳細升級建議】\n")
        for suggestion in suggestions:
            print(suggestion)
