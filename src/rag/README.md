# RAG 檢索 backend 使用方式

目前有五套檢索實作可以切換測試，彼此完全獨立，不互相影響。前三套讀 v1 chunk
（`data/rag_chunks.jsonl`，一型號一個 chunk），後兩套讀 v2 chunk
（`data/rag_chunks_v2.jsonl`，一型號拆成 summary/aspect/pros_cons/comparison
多個語意 chunk，語意搜尋的最小單位是單一 chunk，理論上比 v1 更精準）：

| backend | 檔案 | 說明 |
|---|---|---|
| `tfidf`（預設） | `retriever.py` | v1，TF-IDF 語意搜尋 |
| `chroma` | `retriever_chroma.py` | v1，OpenAI embedding + Chroma 向量搜尋，選相似度最高的 top-k |
| `fulltext` | `retriever_fulltext.py` | v1，辨識類別後，把該類別「目前買得到」的全部型號丟給 LLM 自己判斷推薦 |
| `v2` | `retriever_v2.py` | v2，TF-IDF 語意搜尋，但最小單位是單一 chunk |
| `chroma_v2` | `retriever_chroma_v2.py` | v2，OpenAI embedding + Chroma 向量搜尋，最小單位是單一 chunk |

## 啟動方式（用 `scripts/start_all.sh`，會同時開 RAG 聊天 + GA 推薦服務）

```bash
# 預設 TF-IDF（v1），不用加任何環境變數
./scripts/start_all.sh

# OpenAI embedding + Chroma（v1）
RAG_RETRIEVER=chroma ./scripts/start_all.sh

# 類別全文丟給 LLM 判斷（v1）
RAG_RETRIEVER=fulltext ./scripts/start_all.sh

# TF-IDF，但用 v2 chunk（單一 chunk 為檢索單位）
RAG_RETRIEVER=v2 ./scripts/start_all.sh

# OpenAI embedding + Chroma，用 v2 chunk
RAG_RETRIEVER=chroma_v2 ./scripts/start_all.sh
```

啟動後開瀏覽器連 `http://localhost:8000`（GA 推薦頁面會連到 RAG 聊天）。`Ctrl+C` 會把兩個服務一起關掉。

## 前置作業

- `chroma` backend 第一次使用前，或 `data/rag_chunks.jsonl` 有更新時，需要重建向量索引：
  ```bash
  python src/rag/embed_chunks.py
  ```
- `chroma_v2` backend 第一次使用前，或 `data/rag_chunks_v2.jsonl` 有更新時，需要重建向量索引：
  ```bash
  python src/rag/embed_chunks_v2.py
  ```
- `tfidf`、`fulltext`、`v2` 不需要額外前置作業。

## 不開瀏覽器，直接離線比較結果

```bash
python src/rag/compare_retrievers.py                # v1 三套的離線比較，用內建的測試查詢
python src/rag/compare_retrievers.py "你的自訂查詢"
```

v2 兩套目前沒有對應的 compare 腳本，可以直接在 Python 互動模式呼叫
`RetrieverV2().retrieve(query)` / `RetrieverChromaV2().retrieve(query)` 比較。
