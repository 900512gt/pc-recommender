# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概覽

PC 零件推薦系統，整合 PTT / 巴哈姆特論壇評論，分析零件口碑並協助選購。主要模組：

- **爬蟲**（`src/crawler/`）：抓 PTT 硬體板評論
- **過濾**（`src/filter/`）：清洗評論、人工標注、取樣
- **資料庫建立**（`src/database/`）：整合 CoolPC 商品資料 + 評論分數 → `ga_database.json`
- **Benchmark 合併**（`src/benchmark_collection/`）：加入 PassMark 跑分 → `ga_database_v2.json`
- **RAG 聊天**（`src/rag/`）：用 GPT-4o 蒸餾評論摘要，FastAPI 服務（SSE 串流）供前端查詢零件口碑

## 環境設定

在專案根目錄建立 `.env`：
```
OPENAI_API_KEY=sk-你的金鑰
```

主要依賴：`openai`, `fastapi`, `uvicorn`, `sklearn`, `python-dotenv`, `openpyxl`, `requests`, `beautifulsoup4`

## 常用指令

```bash
# RAG 聊天服務（FastAPI + SSE，預設 port 8001）
python src/rag/server.py

# 同時啟動 RAG 服務（8001）與 GA 推薦服務（8000，位於另一個 repo ga_test）
./scripts/start_all.sh

# 蒸餾 RAG chunks（需 OPENAI_API_KEY，跑全部 ~161 個型號）
python src/rag/distill_chunks.py
python src/rag/distill_chunks.py --test             # 只測試 RTX4070
python src/rag/distill_chunks.py --test "7800X3D"  # 測試指定型號

# PTT 爬蟲
cd src/crawler
python main.py GPU CPU                              # 指定類別
python main.py --all                                # 所有類別

# 查看 / 管理資料庫
cd src/database
python ga_db_manager.py show
python ga_db_manager.py show GPU
python ga_db_manager.py show GPU RTX5080
```

## 資料架構

### 衍生資料庫（最終產物，給 GA / NSGA-II 使用）
- `data/ga_database.json` — v1，含商品規格 + score
- `data/ga_database_v2.json` — v2，含 v1 + benchmark

所有正式模組（NSGA-II、benchmark_collection 的 merge 流程）都讀這兩個檔，沒有其他副本。

### 中間查表（建立 v2 用的輔助資料）
- `data/raw/chip_benchmark.json` — 晶片 → PassMark 跑分查表
- `data/raw/scores.csv` — merge_and_score.py 的中間輸出，給 ga_db_manager.py score 指令讀取
- `data/rag_chunks.jsonl` — distill_chunks.py 產生的摘要，RAG 檢索用

### 原始資料（各模組的私有輸入，不走 data/）
- `src/database/input/` — coolpc 商品資料、PTT 評論、巴哈評論
- `src/filter/output/` — filter pipeline 的中間產物
- `data/coolpc_data.xlsx`、`data/ptt_comment.xlsx`、`data/ga_ptt_models.xlsx` — 較早期的原始檔，部分模組仍使用

### 重建資料庫的流程
1. （可選）更新原始資料：爬蟲、PTT 抓取、新晶片跑分等
2. 建立 v1：在 `src/database/` 執行 `python ga_db_builder.py`，然後 `python merge_and_score.py`，最後 `python ga_db_manager.py score data/raw/scores.csv`
3. 建立 v2：在專案根目錄執行 `python src/benchmark_collection/merge_benchmark.py`，再執行 `python src/benchmark_collection/merge_tgp.py` 補入 GPU 的 `tgp_watts` 欄位（此步驟容易被漏掉，因為它不在原本規劃的流程圖裡，但 v2 的 GPU 條目都依賴它）

### 已過期的副本（保留為時光膠囊，不要使用）
- `src/database/output/` — 過期的 v1 與 scores.csv 副本
- `src/nsga/input/` — 過期的 v2 副本，配 nsga/test.py 廢棄腳本

## RAG 系統架構

`distill_chunks.py` 讀取 `src/database/input/` 的 PTT + 巴哈評論，對每個零件型號呼叫 GPT-4o-mini 蒸餾成結構化摘要（pros / cons / aspects / comparisons / summary），寫入 `data/rag_chunks.jsonl`。

`retriever.py` 在 `rag_chunks.jsonl` 上做檢索：優先精確比對型號名稱（含縮寫別名），找不到才用 TF-IDF（字元 n-gram）做語意搜尋。

`chat.py` 是純邏輯模組（組 prompt、呼叫 GPT-4o 串流），不含任何介面程式碼，對話歷史最多保留 6 輪。實際對外服務的是 `server.py`：FastAPI + `POST /api/chat`，以 SSE（Server-Sent Events）將逐步累積的文字串流回前端；`GET /health` 供健康檢查。前端目前有兩份：`src/rag/static/widget.js`（可嵌入任意網頁的浮動小工具）與 `ga_test` repo 裡 Next.js 重寫的聊天元件，兩者呼叫同一支 API。

> 完整全端架構（含 GA 推薦引擎、API 層、Next.js 前端的細節與已知限制）見 `FULLSTACK_ARCHITECTURE.md`。
