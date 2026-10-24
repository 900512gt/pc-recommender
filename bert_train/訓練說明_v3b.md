# BERT 面向情感 v3b 訓練說明（2026-10-24）

## 為什麼要再訓練一次

v3 只訓練「至少一個面向有評價」的評論，把「相關但所有面向都未提及」的 4,674 則也拿掉了。

但實際使用時的流程是「相關性過濾 → BERT」：相關性過濾只分得出相關／不相關，分不出有沒有評價。所以真正送進 BERT 的評論裡，約 19%（4,674 / 24,973）是相關但沒有評價的，v3 卻沒在訓練時看過這種評論。

現有結果也看得到代價（不含總體）：

| 模型 | 訓練時看過「相關但無評價」嗎 | 這類評論的誤判率 |
|---|---|---|
| 舊模型（10/06） | 有 | 7.4% |
| v3_full | 沒有 | 9.1% |

v3b 的改法：**不相關的仍然拿掉**（上線時會先被過濾），**相關但無評價的保留**（上線時一定會遇到）。其他設定和 v3_full 完全相同（不減量、固定種子 42、4 輪）。

## 需要的程式版本

分支 `aspect-sentiment-rebuild`，pull 最新版。這次新增 `dump_confusion.py`（存混淆矩陣，給報告畫圖）。

## 步驟

以下指令都在 `bert_train/` 底下執行。

### 1. 建立 v3b 資料（約 10 秒）

和 v3_full 的指令只差在**不加 `--drop-no-opinion`**：

```
python build_aspect_dataset.py --drop-irrelevant --overall aspects_總體_p4.jsonl --neg-ratio 0 --output-dir aspect_dataset_v3b
```

對一下樣本數，一樣代表資料相同（完整輸出見 `aspect_dataset_v3b_建立紀錄.txt`）：

| | 樣本數 | 未提及 | 正面 | 負面 |
|---|---|---|---|---|
| train | 149,667 | 120,775 | 15,477 | 13,415 |
| val | 14,938 | 12,042 | 1,628 | 1,268 |
| test | 18,172 | 14,585 | 1,973 | 1,614 |
| test_irrelevant | 21,636 | 全部 | — | — |

正面、負面和 v3_full 完全相同，多出來的只有無評價評論帶來的未提及樣本（未提及比例 76.3% → 80.7%）。
這次沒有 `test_no_opinion.jsonl`：無評價的評論已經在 val、test 裡面了。

### 2. 試跑（幾分鐘）

```
python train_aspect.py --model ckip --sample 2000 --epochs 1 --data-dir aspect_dataset_v3b --save-dir runs/smoke --overwrite
```

確認沒有報錯、loss 有往下降、預測分布裡正面和負面不是 0。

### 3. 正式訓練

train 是 v3_full 的 1.23 倍，時間約多兩成。

```
$env:PYTHONIOENCODING="utf-8"
cmd /c "python -u train_aspect.py --model ckip --epochs 4 --data-dir aspect_dataset_v3b --save-dir runs/v3b > train_log_v3b.txt 2>&1"
```

### 4. v3_full 在 v3b 資料上重新評估（不訓練）

v3b 的 val、test 多了無評價評論，分數不能和 v3_full 原本的成績直接比。要讓 v3_full 也跑一次 v3b 的資料：

```
cmd /c "python -u train_aspect.py --eval-only --model-dir runs/v3_full --data-dir aspect_dataset_v3b > eval_v3full_on_v3b.txt 2>&1"
```

結果存在 `runs/v3_full/eval_aspect_dataset_v3b.json`。

### 5. 存混淆矩陣（兩個模型都要）

```
python dump_confusion.py --model-dir runs/v3b --data-dir aspect_dataset_v3b --out cm_v3b
python dump_confusion.py --model-dir runs/v3_full --data-dir aspect_dataset_v3b --out cm_v3full_on_v3b
```

每個約幾分鐘。會印出主要測試集、其中無評價評論、不相關評論三份混淆矩陣（不含總體）。

## 怎麼選模型

**看 val，不看 test**（和上次相同的規則）：

- 主要：兩個模型在 v3b val 上的 Macro-F1（不含總體）、正面／負面 precision 與 recall。
- 輔助：val 裡無評價評論的誤判率。

test 只用來報告最後選定那個模型的成績。

## 跑完要回傳的

- `runs/v3b/result.json`
- `runs/v3_full/eval_aspect_dataset_v3b.json`
- `cm_v3b/confusion.json`、`cm_v3full_on_v3b/confusion.json`（`.npy` 檔不用傳）
- `train_log_v3b.txt`、`eval_v3full_on_v3b.txt`
- 在 `BERT訓練成果_20261010/實驗紀錄.md` 加一行

模型檔（`model.safetensors`）先不用傳。
