# RAG 檢索 backend 使用方式

目前有兩套檢索實作可以切換測試，都讀 v2 chunk（`data/rag_chunks_v2.jsonl`，
一型號拆成 summary/aspect/pros_cons/comparison 多個語意 chunk，語意搜尋的
最小單位是單一 chunk）：

| backend | 檔案 | 說明 |
|---|---|---|
| `chroma_v2`（預設，正式環境用） | `retriever_chroma_v2.py` | OpenAI embedding + Chroma 向量搜尋，最小單位是單一 chunk；型號比對跟類別分類都用 LLM 判斷 |
| `v2` | `retriever_v2.py` | TF-IDF 語意搜尋，一樣最小單位是單一 chunk，不需要額外索引，當 Chroma 向量服務出問題時的備援 |

v1（純字串/TF-IDF、不拆語意面向的舊 chunk 格式，`data/rag_chunks.jsonl`）
已經整套移除，不再是選項。

## 啟動方式（用 `scripts/start_all.sh`，會同時開 RAG 聊天 + GA 推薦服務）

```bash
# 預設 OpenAI embedding + Chroma（chroma_v2），不用加任何環境變數
./scripts/start_all.sh

# TF-IDF 備援版本
RAG_RETRIEVER=v2 ./scripts/start_all.sh
```

啟動後開瀏覽器連 `http://localhost:8000`（GA 推薦頁面會連到 RAG 聊天）。`Ctrl+C` 會把兩個服務一起關掉。

## 前置作業

- `chroma_v2` backend 第一次使用前，或 `data/rag_chunks_v2.jsonl` 有更新時，需要重建向量索引：
  ```bash
  python src/rag/embed_chunks_v2.py
  ```
- `v2` 不需要額外前置作業。

## 回歸測試

```bash
python -m pytest src/rag/test_retrieval_v2.py -v
```

需要 `.env` 設定 `OPENAI_API_KEY`，會真的呼叫 API（embedding + `gpt-4o-mini`
型號/類別抽取），費用很低但不是純離線測試。
