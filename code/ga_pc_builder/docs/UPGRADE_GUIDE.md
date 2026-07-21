# 升級建議系統 — 使用指南

## 功能概述

新增的升級建議系統包含三個主要模塊：

| 模塊 | 位置 | 說明 |
|------|------|------|
| 升級建議引擎 | `advisor/upgrade_advisor.py` | 根據配置、預算和使用目的提供智能升級建議，支持按優先級推薦，考慮預算約束和性價比 |
| 配置分析工具 | `advisor/build_analyzer.py` | 對用戶提供的配置進行完整分析，識別最弱環節並提供升級優先級與性價比分析 |
| 主程式整合 | `main.py` | 集成升級建議功能，`print_upgrade_recommendations()` 顯示智能升級建議 |

---

## 使用方式

### 方式 1：快速分析現有配置

```bash
python advisor/build_analyzer.py
```

輸出內容：
- 配置總價
- 平均情感評分
- 各零件的情感評分對比（視覺化柱狀圖）
- 升級優先級分析（按評分低→高排序）
- 性價比評估（每仟元的情感評分）
- 詳細升級建議

---

### 方式 2：在 GA 推薦系統中自動顯示升級建議

```bash
python main.py
```

工作流程：
1. 輸入預算、使用目的、散熱偏好
2. GA 演算法生成 Top 5 推薦配置
3. 顯示第1名推薦配置
4. 自動分析並顯示升級選項（根據剩餘預算）
   - 優先級最高的升級（性能提升/成本最優）
   - 具體升級細節（目前 → 升級、價格、效益）
   - 升級組合建議（全部升級或選擇性升級）

---

### 方式 3：在程式碼中直接整合

```python
from advisor.upgrade_advisor import UpgradeAdvisor
from advisor.build_analyzer import print_config_analysis
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer

# 初始化
catalog = PartCatalog(DB_PATH)
scorer  = SentimentScorer(MATCHED_FILES, DB_PATH)
advisor = UpgradeAdvisor(catalog, scorer)

# 獲取智能推薦
recommendations = advisor.get_smart_recommendations(
    build=best_config,
    budget=50000,
    remaining_budget=4500,
    usage="工作"
)

# 遍歷推薦
for rec in recommendations:
    print(f"{rec['priority']}. {rec['category']} 升級")
    print(f"   成本: +NT${rec['cost']:,}")
    print(f"   效益: {rec['benefit']}")
```

---

## 關鍵特性

1. **智能優先級分析**
   - 根據使用目的調整優先級（遊戲優先 GPU，工作優先 CPU/RAM）
   - 按性能提升/成本比排序建議

2. **預算感知**
   - 只推薦在剩餘預算內的升級選項
   - 計算升級後的總預算

3. **情感評分驅動**
   - 使用情感評分衡量零件品質
   - 推薦具有最大性價比的升級

4. **個性化建議**
   - 針對工作 / 遊戲 / 文書等不同使用場景
   - 提供具體的效益描述（FPS 提升、響應速度等）

5. **視覺化分析**
   - 柱狀圖顯示情感評分對比
   - 升級優先級清單
   - 性價比對比表

---

## 示例輸出

```
【根據您的預算和使用需求的升級建議】
剩餘預算：NT$4,500 可用於升級

【第 1 優先】記憶體 升級
  目前：UMAX 單條32GB DDR5-4800/CL40
         NT$8,888
  升級：G.Skill TRIDENT Z5 RGB 32GB DDR5-6000
         NT$11,990 (增加 +NT$3,102)
  效益：系統響應速度提升，支援更多同時應用
  理由：容量/頻率中等提升，改善多工能力
  情感評分：+0.15%

若全部升級：NT$45,637 → NT$48,739 (增加 +NT$3,102)
剩餘預算：NT$1,261
```

---

## 數據來源

| 資料 | 來源 |
|------|------|
| 零件價格 | `data/ga_database_v2.json` |
| 情感評分 | `matched_part*.jsonl` |
| 相容性規則 | `advisor/compatibility.py` |
| 性能權重 | `config.py` → `USAGE_WEIGHTS` |

---

## 未來擴展方向

- [ ] **預算優化**：自動組合升級以最大化預算利用率
- [ ] **互動式選擇**：交互式菜單選擇要升級的零件，動態重新計算剩餘預算
- [ ] **升級路徑規劃**：多階段升級計劃 + 成本收益分析
- [ ] **實時市場數據**：與線上商城 API 整合，自動更新零件價格與庫存
- [ ] **機器學習優化**：基於用戶選擇學習升級偏好，個性化推薦改進
