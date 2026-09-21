# train_bert.py 程式說明文件

本文件說明 `train_bert.py` 這支程式（階段 B：fine-tune BERT 做五大維度情感分類）的運作原理，讓你不只會跑，還知道每一步在做什麼、為什麼要這樣設計。

## 1. 這支程式要解決什麼問題

輸入是一則使用者評論（例如某顆 CPU 的開箱心得），輸出是「這則評論在效能、溫控、噪音、保固、CP值這五個面向上，各自表達了什麼情緒」。

每個維度都各自獨立判斷成三種結果之一：

| 標籤 | 代碼 | 意思 |
|---|---|---|
| 未提及 | 0 | 這則評論完全沒談到這個面向 |
| 正面 | 1 | 對這個面向給出正面評價 |
| 負面 | 2 | 對這個面向給出負面評價 |

從 `train_dimensions.jsonl` 的實際資料可以看到範例：

```json
{
  "category": "CPU",
  "model": "AMD R7 7800X3D",
  "content": "拿到 9700X 做了一些小測試...",
  "dimensions": {
    "效能": "正面", "溫控": "未提及", "噪音": "未提及",
    "保固": "未提及", "CP值": "正面"
  }
}
```

也就是說，一則評論同時要輸出 5 個獨立的三分類結果，而不是單一個「這篇評論是正面還是負面」的整體判斷。這是「多標籤、多分類」（multi-task classification）問題，不是單純的情感分析。

## 2. 整體架構：一顆 BERT + 五個分類頭

```
輸入文字
   │
   ▼
BertTokenizerFast（斷詞、轉成 ID）
   │
   ▼
BertModel（共用的語言理解主幹）
   │
   ▼
取出 [CLS] 的 pooler_output（代表整句話的語意向量）
   │
   ▼
Dropout（0.3，防止過擬合）
   │
   ├──▶ 分類頭 1（效能）──▶ 3 個分數（未提及/正面/負面）
   ├──▶ 分類頭 2（溫控）──▶ 3 個分數
   ├──▶ 分類頭 3（噪音）──▶ 3 個分數
   ├──▶ 分類頭 4（保固）──▶ 3 個分數
   └──▶ 分類頭 5（CP值）──▶ 3 個分數
```

對應到程式碼裡的 `MultiDimBert` 類別（第 72–87 行）：

```python
class MultiDimBert(nn.Module):
    def __init__(self, model_name, n_dims=5, n_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleList([
            nn.Linear(hidden, n_classes) for _ in range(n_dims)
        ])

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        return [head(pooled) for head in self.heads]
```

關鍵設計：**BERT 主幹只有一份，但接了 5 個獨立的線性分類層（`nn.Linear`）**。這樣做的好處：

- **共用語意理解**：BERT 只需要學一次「怎麼讀懂中文評論」，5 個維度共用這個能力，比訓練 5 個獨立模型省資源、也讓少數維度（例如「保固」提到的次數較少）能借助其他維度的語意知識。
- **各自獨立判斷**：每個分類頭是獨立的 `nn.Linear(hidden, 3)`，維度之間互不干擾，效能判斷不會影響噪音判斷的輸出。
- `forward()` 回傳的是一個長度為 5 的 list，每個元素是 `(batch_size, 3)` 的張量，也就是這批資料在該維度上，屬於「未提及/正面/負面」的原始分數（logits，還沒經過 softmax）。

`pooler_output` 是 BERT 對 `[CLS]` 這個特殊 token 再經過一層 tanh 全連接層的輸出，Hugging Face 的慣例把它當成「整句話的摘要向量」，是做分類任務時最常用的取法。

## 3. 資料怎麼餵進模型：`ReviewDataset`

```python
class ReviewDataset(Dataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["content"],
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        labels = torch.tensor(
            [LABEL2ID[item["dimensions"][d]] for d in DIMENSIONS],
            dtype=torch.long,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
```

每次取一筆資料時：

1. 把 `content`（評論文字）丟給 tokenizer，切成最多 128 個 token（`max_len=128`），不夠長的補齊（padding），太長的截斷（truncation）。
2. `input_ids`：文字轉成的數字序列；`attention_mask`：標記哪些位置是真的文字、哪些是補齊用的 padding（讓模型忽略 padding 部分）。
3. `labels`：把 `dimensions` 這個 dict 依照 `DIMENSIONS = ["效能","溫控","噪音","保固","CP值"]` 的固定順序，轉成長度為 5 的整數向量，例如 `[1, 0, 0, 0, 1]` 對應「效能正面、CP值正面、其餘未提及」。**順序一定要固定**，這樣模型輸出的第 0 個分類頭永遠對應「效能」，訓練和推論才能對得起來。

`DataLoader` 再把多筆這樣的資料打包成一個 batch（預設 16 筆一批），並且訓練時 `shuffle=True` 讓每個 epoch 看到資料的順序都不同，避免模型記住資料順序。

## 4. 為什麼要處理類別不平衡（class weight）

大多數評論只會談到一兩個面向，例如一篇文章聊 CPU 效能，通常不會同時提到保固和噪音。這代表在「保固」這個維度上，「未提及」的樣本數量會遠遠多於「正面」和「負面」。如果直接訓練，模型會學到一個偷懶但準確率很高的策略：**無論輸入是什麼，永遠猜「未提及」**——因為多數樣本本來就是未提及，這樣準確率（Accuracy）看起來還不低，但完全沒有學到怎麼判斷情感。

程式碼用 class weight 來對抗這個問題（第 125–135 行）：

```python
weights = []
for di, dim in enumerate(DIMENSIONS):
    counts = [1, 1, 1]  # 平滑：避免除以 0
    for d in train_data:
        counts[LABEL2ID[d["dimensions"][dim]]] += 1
    total = sum(counts)
    w = torch.tensor([total / c for c in counts], dtype=torch.float).to(device)
    w = w / w.sum() * 3  # 正規化
    weights.append(w)

criterions = [nn.CrossEntropyLoss(weight=w) for w in weights]
```

運作邏輯：

1. 先統計訓練集裡，這個維度的 3 個類別（未提及/正面/負面）各出現幾次，`counts` 從 `[1,1,1]` 起算是做 Laplace smoothing，避免某類別剛好 0 筆導致除以零。
2. `total / c`：某類別出現越少，權重就越大。例如「保固-正面」只出現 50 次，「保固-未提及」出現 5000 次，那麼「保固-正面」的權重就會遠大於「保固-未提及」。
3. 除以總和再乘 3 只是把權重正規化到平均值附近，避免 loss 整體被放大太多倍而影響學習率的效果。
4. 每個維度各自算出一組獨立的權重，因為 5 個維度的不平衡程度不同（例如「效能」被提到的次數通常遠多於「保固」）。
5. 這組權重丟進 `nn.CrossEntropyLoss(weight=w)`：計算 loss 時，猜錯稀有類別（例如把真正的「負面」猜成別的）會被罰得比猜錯常見類別更重，逼模型認真學習少數類別，而不是躺平猜多數。

## 5. 訓練迴圈在做什麼

核心邏輯在 `train()` 函式的迴圈裡（第 145–169 行），拆解成幾個步驟：

```python
for epoch in range(args.epochs):
    for i, batch in enumerate(train_dl):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)          # (batch, 5)

        optimizer.zero_grad()
        outputs = model(input_ids, mask)               # 5 個 (batch, 3)

        loss = 0
        for di in range(len(DIMENSIONS)):
            loss += criterions[di](outputs[di], labels[:, di])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```

逐行意義：

- **`optimizer.zero_grad()`**：PyTorch 預設會累加梯度，所以每個 batch 開始前要先清空上一批留下的梯度。
- **`outputs = model(...)`**：一次前向傳播，同時拿到 5 個維度的預測分數。
- **合併 5 個維度的 loss**：`loss` 是 5 個 `CrossEntropyLoss` 的總和——每個維度各自算自己的分類誤差，再加總成一個總 loss。這代表反向傳播時，BERT 主幹會同時收到 5 個維度傳回來的梯度訊號，共同調整同一組參數；而每個分類頭只會被自己那個維度的誤差調整。
- **`loss.backward()`**：反向傳播，計算每個參數該怎麼調整才能讓 loss 變小。
- **`clip_grad_norm_(..., 1.0)`**：梯度裁剪。避免某個 batch 的梯度突然暴衝（gradient explosion），把梯度的總長度限制在 1.0 以內，讓訓練更穩定，尤其對 fine-tune 大型預訓練模型很常見。
- **`optimizer.step()`**：用 AdamW 優化器，依照算好的梯度真正更新模型參數。AdamW 是 BERT 類模型 fine-tune 的標準選擇，在 Adam 的基礎上修正了權重衰減（weight decay）的處理方式。
- **`scheduler.step()`**：更新學習率。用的是 `get_linear_schedule_with_warmup`——訓練一開始的 10%（`num_warmup_steps = total_steps * 0.1`）學習率會從 0 線性升到設定值（預設 `2e-5`），之後再線性下降到 0。Warmup 是為了避免訓練剛開始時，模型參數還很亂，太大的學習率會把預訓練好的 BERT 權重一下子破壞掉；後段線性下降則是讓模型在收斂階段做更細緻的微調。

每 50 個 step 印一次目前的 loss，方便觀察是否在下降；每個 epoch 結束後呼叫 `evaluate()` 在測試集上算一次分數，讓你即時看到訓練成效如何，而不必等全部訓練完才知道好壞。

## 6. 評估：`evaluate()` 在算什麼

```python
def evaluate(model, dl, device):
    model.eval()
    ...
    with torch.no_grad():
        for batch in dl:
            outputs = model(input_ids, mask)
            for di in range(len(DIMENSIONS)):
                preds = outputs[di].argmax(dim=1).cpu().numpy()
                ...
    for di, dim in enumerate(DIMENSIONS):
        acc = accuracy_score(...)
        f1 = f1_score(..., average="macro", zero_division=0)
```

- **`model.eval()`**：把模型切到評估模式，關閉 Dropout（訓練時才需要隨機丟掉神經元來防止過擬合，評估時要用完整的模型能力）。
- **`torch.no_grad()`**：評估不需要算梯度，關掉可以省記憶體、加快速度。
- **`argmax(dim=1)`**：每個維度的輸出是 3 個分數（對應未提及/正面/負面），取分數最高的那個當作模型的預測類別。
- **兩種指標**：
  - **Accuracy（準確率）**：預測對的比例。但前面提過，這個指標在類別不平衡時會誤導人（全猜「未提及」也能有高準確率）。
  - **Macro-F1**：分別算出「未提及」「正面」「負面」三個類別各自的 F1-score（精確率和召回率的調和平均），再取三者的平均，不看樣本數多寡加權。這代表就算「正面」樣本很少，只要模型在「正面」這個類別上表現差，Macro-F1 就會被拖累——**這才是這個任務真正該看的指標**，因為我們在意的是模型有沒有認真學會辨認正面/負面，而不是靠猜「未提及」混過去。
- 5 個維度算完後，再取這 5 個 Macro-F1 的平均值，作為整個模型的總評分，方便比較「中研院 BERT」和「Google BERT」誰表現比較好。

## 7. 模型選擇：`--model` 參數

```python
MODEL_MAP = {
    "ckip":   "ckiplab/bert-base-chinese",     # 中研院
    "google": "bert-base-chinese",              # Google 原版
    "hfl":    "hfl/chinese-roberta-wwm-ext",    # 哈工大(可選，通常最強)
}
```

三個都是 Hugging Face 上公開的預訓練中文 BERT，差別在訓練語料和訓練方式：

- **ckip（中研院）**：用繁體中文語料訓練，理論上對台灣的用語、斷詞習慣更合適。
- **google（Google 原版）**：`bert-base-chinese` 是 Google 官方釋出的中文 BERT，簡繁體混合語料訓練。
- **hfl（哈工大 RoBERTa）**：用 whole-word-masking 技術訓練的 RoBERTa，一般在中文 NLP benchmark 上分數最高，但模型比較大、訓練時間更長。

程式設計成只要換 `--model` 參數就能切換要 fine-tune 哪一個，其餘程式碼（資料處理、訓練迴圈、評估方式）完全共用，這樣才能公平比較三者在你的資料上表現優劣，最後選 Macro-F1 最高的那個接到後面的推論階段（階段 C，也就是 `ga_pc_builder` 使用的情感分數來源）。

## 8. 各項超參數的意義

| 參數 | 預設值 | 意義 |
|---|---|---|
| `--epochs` | 3 | 整份訓練資料要被模型看幾遍 |
| `--batch` | 16 | 一次丟幾筆資料進模型（RTX 4080 可以跑到 16，記憶體不夠就降到 8） |
| `--lr` | 2e-5 | 學習率，BERT fine-tune 的標準值（比從頭訓練模型的學習率小很多，因為只是要「微調」已經很懂語言的模型，不能調太大力道破壞原本學到的知識） |
| `--sample` | None | 只取資料集前 N 筆，用來快速試跑確認程式沒 bug，不用等正式訓練跑完才發現錯誤 |
| `--max_len`（寫死在 `ReviewDataset`） | 128 | 每則評論最多保留 128 個 token，超過會被截斷 |

`train_test_split(data, test_size=0.15, random_state=42)`：把 16,392 則資料切成 85% 訓練、15% 測試，`random_state=42` 固定亂數種子，確保每次執行切出來的訓練/測試集都一樣，方便重現結果、公平比較不同模型。

## 9. 訓練完成後產出什麼

```python
save_path = f"bert_{args.model}_dims.pt"
torch.save(model.state_dict(), save_path)

with open(f"bert_{args.model}_config.json", "w") as f:
    json.dump({"model_name": model_name, "dimensions": DIMENSIONS}, f)
```

- **`bert_{model}_dims.pt`**：模型權重檔（`state_dict`，只存參數數值，不含模型結構程式碼，推論時要重新用 `MultiDimBert` 把結構建出來，再載入這個檔案）。
- **`bert_{model}_config.json`**：記錄這個模型是用哪個底層 BERT（`model_name`）訓練的，以及 5 個維度的名稱順序（`dimensions`）。這個檔案很重要——推論時必須用同一個 `model_name` 去載入對應的 tokenizer 和 BERT 結構，順序也要跟訓練時一致，不然模型輸出的分類頭會跟維度名稱對不上。

從實際存在的 `bert_ckip_config.json` 和 `bert_google_config.json` 可以看到，這兩個模型都已經訓練過，維度順序都是 `["效能","溫控","噪音","保固","CP值"]`。

## 10. 這支程式如何接到後面的 GA 選配系統

依照 README.md 的規劃，這支程式是整個專案的「階段 B」：

1. **階段 A**（不在這支程式裡）：把評論資料標註好五大維度的情感標籤，產出 `train_dimensions.jsonl`。
2. **階段 B**（就是 `train_bert.py`）：用標註好的資料 fine-tune BERT，比較中研院/Google（/哈工大）三個底層模型，選出 Macro-F1 最高的一個。
3. **階段 C**（尚未看到程式碼）：把選定的模型套用到全部 50,593 則評論上（比訓練用的 16,392 則多很多，代表其餘的評論在階段 A 沒有標註），推論出每個零件在五大維度上的情感分數。
4. **接進 GA**：這些情感分數就是 `ga_pc_builder` 專案裡 `SentimentScorer`（`data/sentiment.py`）讀取的資料來源，`GARecommender`（`core/ga_engine.py`）拿它們當作遺傳演算法 fitness function 的一部分，用來評估某個零件組合的「口碑」好壞，進而推薦配置給使用者。

也就是說，這支程式訓練出來的模型品質，會直接影響 GA 選配系統推薦結果的準確度——如果情感分數判斷得不準，GA 演化出來的「高分配置」也會跟著失真。

## 11. 使用時的實務建議

- **一定要先用 `--sample 1000 --epochs 1` 小資料試跑**：這只是確認程式邏輯沒問題（tokenizer 正常、loss 會下降、評估能跑完），大約 5–10 分鐘，不要跳過這步直接跑正式訓練，否則要是有 bug，等於浪費 1–3 小時。
- **看 Macro-F1，不要只看 Accuracy**：前面第 6 節解釋過，Accuracy 在這個任務上容易被「未提及」這個多數類別撐高，容易誤判模型真的學會了。
- **三個模型都跑過再比較**：`ckip` vs `google`（vs 有空的話 `hfl`），用同一份資料、同樣的 epoch 數比較 Macro-F1，公平選出最適合這批中文 PC 評論的模型。
- **顯存不夠就降 `--batch`**：從 16 降到 8，訓練速度會變慢，但可以避免 CUDA out of memory 的錯誤。
