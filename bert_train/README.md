# 階段 B：fine-tune BERT 五大維度情感分類

> 這個專案分三個階段：**階段 A** 把評論資料標註好五大維度的情感標籤（產出 `train_dimensions.jsonl`）→ **階段 B**（就是這份文件在講的）用標註好的資料 fine-tune BERT，比較不同底層模型 → **階段 C** 選定模型後，套用到全部評論上，算出每個零件的面向分數，接進 GA 選配系統。這份 README 只涵蓋階段 B，階段 A 的標註程式不在這個資料夾裡。

## 環境需求（你已具備）
- PyTorch + CUDA（你的 4080 ✓）
- transformers, scikit-learn

如果缺 scikit-learn: `pip install scikit-learn`

## 訓練資料
`train_dimensions.jsonl`（16,392 則，五大維度標注）
放在跟 `train_bert.py` 同一資料夾。

## 使用順序

### 1. 先小資料試跑（確認沒錯，約 5-10 分鐘）
```
python train_bert.py --model ckip --sample 1000 --epochs 1
```
看到各維度的 Acc/F1 出現就代表腳本正常。

### 2. 正式訓練中研院 BERT（約 1-3 小時）— ✅ 已完成（`bert_ckip_dims.pt` 已在資料夾裡）
```
python train_bert.py --model ckip --epochs 3
```
產出：`bert_ckip_dims.pt`

### 3. 訓練 Google BERT（比較用）— ✅ 已完成（`bert_google_dims.pt` 已在資料夾裡）
```
python train_bert.py --model google --epochs 3
```
產出：`bert_google_dims.pt`

### 4. 比較兩個的 F1 分數，選表現好的接進 GA
把兩次訓練最後印出來的「平均 Macro-F1」記錄在這裡，方便之後回頭查、也方便寫報告：
- ckip：___
- google：___

## 模型選項
- `--model ckip`   : 中研院 ckiplab/bert-base-chinese
- `--model google` : Google bert-base-chinese
- `--model hfl`    : 哈工大 RoBERTa（可選，通常最強，尚未訓練）

## 參數
- `--epochs 3`   : 訓練輪數（3 夠用）
- `--batch 16`   : 批次大小（4080 可以 16，顯存不夠降到 8）
- `--lr 2e-5`    : 學習率（標準值，別亂改）
- `--sample N`   : 只用前 N 則（試跑用）

## 輸出解讀
```
效能: Acc 0.85  Macro-F1 0.78
溫控: Acc 0.88  Macro-F1 0.75
...
平均 Macro-F1: 0.76
```
Macro-F1 越高越好。中研院 vs Google 比這個數字選模型。

## 下一步（階段 C）
選好模型後，用它跑全部 50,593 則評論 → 每個零件的面向分數。

> 想看程式碼裡每一步在做什麼，可以看同資料夾裡的 `train_bert_說明文件.md`。
