# RAG 檢索 backend 使用方式

目前有三套檢索實作可以切換測試，彼此完全獨立，不互相影響：

| backend | 檔案 | 說明 |
|---|---|---|
| `tfidf`（預設） | `retriever.py` | 原本的 TF-IDF 語意搜尋 |
| `chroma` | `retriever_chroma.py` | OpenAI embedding + Chroma 向量搜尋，選相似度最高的 top-k |
| `fulltext` | `retriever_fulltext.py` | 辨識類別後，把該類別「目前買得到」的全部型號丟給 LLM 自己判斷推薦 |

## 啟動方式（用 `scripts/start_all.sh`，會同時開 RAG 聊天 + GA 推薦服務）

```bash
# 預設 TF-IDF，不用加任何環境變數
./scripts/start_all.sh

# OpenAI embedding + Chroma
RAG_RETRIEVER=chroma ./scripts/start_all.sh

# 類別全文丟給 LLM 判斷
RAG_RETRIEVER=fulltext ./scripts/start_all.sh
```

啟動後開瀏覽器連 `http://localhost:8000`（GA 推薦頁面會連到 RAG 聊天）。`Ctrl+C` 會把兩個服務一起關掉。

## 前置作業

- `chroma` backend 第一次使用前，或 `data/rag_chunks.jsonl` 有更新時，需要重建向量索引：
  ```bash
  python src/rag/embed_chunks.py
  ```
- `tfidf`、`fulltext` 不需要額外前置作業。

## 不開瀏覽器，直接離線比較三套結果

```bash
python src/rag/compare_retrievers.py                # 用內建的測試查詢
python src/rag/compare_retrievers.py "你的自訂查詢"
```
