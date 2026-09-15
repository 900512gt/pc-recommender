"""
server.py
FastAPI 後端：接收聊天請求，以 SSE 串流回傳 RAG 回答。

使用方式（從專案根目錄執行）：
  python src/rag/server.py
  # 或開發模式（程式碼異動自動重啟）：
  uvicorn src.rag.server:app --reload --port 8000
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import iterate_in_threadpool

from src.rag.chat import chat_stream, _has_substantive_data
from src.rag.evidence import get_store
from src.rag.timeline import get_store as timeline_store
from src.rag.aspects import get_store as aspect_store
from src.rag.models import get_store as model_store

load_dotenv(ROOT / ".env")

# 語意搜尋 backend 切換：
#   RAG_RETRIEVER=chroma_v2  （預設，正式環境用這個）OpenAI embedding + Chroma
#                            版本，讀 v2 chunk（data/rag_chunks_v2.jsonl），
#                            需先執行 python src/rag/embed_chunks_v2.py
#   RAG_RETRIEVER=v2         TF-IDF 版本，一樣讀 v2 chunk，不需要額外索引，
#                            當 Chroma 向量服務出問題時的備援
# v1（純字串/TF-IDF、不拆語意面向的舊 chunk 格式）已經整套移除，不再是選項。
RETRIEVER_BACKEND = os.environ.get("RAG_RETRIEVER", "chroma_v2")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="PC 零件口碑 API")

# 正式前端網域固定放行；沒設 CORS_ORIGIN_REGEX 時行為跟以前完全一樣，
# 所以正式環境不會因為這個機制多開任何來源。
# 本機開發要讓瀏覽器打得到這支 API，啟動前設：
#   export CORS_ORIGIN_REGEX='http://(localhost|127\.0\.0\.1):[0-9]+'
# 用 regex 而不是列舉 port，是因為 Next.js dev 遇到 3000 被佔用會自動換 3001、3002…
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pc-recommender.vercel.app"],
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX") or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 每則訊息都會呼叫 OpenAI 計費，限制每 IP 頻率避免被打爆額度
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Singletons（首次請求時延遲初始化）────────────────────────────────────────

_client: OpenAI | None = None
_retriever = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("請在 .env 設定 OPENAI_API_KEY")
        _client = OpenAI(api_key=api_key)
    return _client


def _get_retriever():
    global _retriever
    if _retriever is None:
        if RETRIEVER_BACKEND == "v2":
            from src.rag.retriever_v2 import RetrieverV2
            _retriever = RetrieverV2()
        else:
            from src.rag.retriever_chroma_v2 import RetrieverChromaV2
            _retriever = RetrieverChromaV2()
        print(f"[RAG] 載入 {len(_retriever.chunks)} 個零件評價 chunk（backend={RETRIEVER_BACKEND}）")
    return _retriever


# ── Schema ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
@limiter.limit("15/minute")
async def chat_endpoint(request: Request, req: ChatRequest):
    """
    SSE 串流端點。

    第一個 event 是本次回答依據的原始評論（依型號分組，可能是空陣列）：
      data: {"sources": [{"model": ..., "total": N, "comments": [...]}]}\n\n

    之後每個 event 是逐步累積的回答文字：
      data: {"text": "<累積回答>"}\n\n

    最終 sentinel：
      data: [DONE]\n\n

    錯誤時：
      data: {"error": "<訊息>"}\n\n
      data: [DONE]\n\n
    """
    def _sync_gen():
        try:
            chunks = _get_retriever().retrieve(req.query)

            # 先把佐證送出去，前端才能在等生成的同時就渲染來源清單。
            # 資料不足的型號視同沒有論壇依據（chat.py 也是這樣判斷），
            # 不能讓面板顯示一堆「評論數量過少」的空 chunk 佯裝成佐證。
            sources = get_store().collect(chunks) if _has_substantive_data(chunks) else []
            yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"

            for text in chat_stream(
                req.query,
                req.history,
                chunks,
                _get_client(),
            ):
                payload = json.dumps({"text": text}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception as exc:
            import traceback
            traceback.print_exc()
            err = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    async def _async_gen():
        async for chunk in iterate_in_threadpool(_sync_gen()):
            yield chunk

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/models")
@limiter.limit("60/minute")
def list_models(request: Request):
    """所有有口碑資料的型號，給 /parts 列表頁做搜尋與篩選。"""
    return {"models": model_store().index()}


@app.get("/api/model/{model}")
@limiter.limit("60/minute")
def model_detail(request: Request, model: str):
    """型號詳情：蒸餾出的摘要／面向敘述／優缺點，加上售價與跑分。

    逐月走勢與面向分數不含在這裡，詳情頁另外打 /timeline 與 /aspects
    ——那兩份資料量大，列表頁與只想看文字的情境都用不到。
    """
    data = model_store().detail(model)
    if data is None:
        raise HTTPException(status_code=404, detail="查無此型號")
    return data


@app.get("/api/model/{model}/comments")
@limiter.limit("60/minute")
def model_comments(request: Request, model: str, limit: int = 30):
    """某型號的原始論壇評論，正負評交錯取樣。"""
    return {"model": model, "comments": get_store().for_model(model, limit)}


@app.get("/api/model/{model}/timeline")
@limiter.limit("60/minute")
def model_timeline(request: Request, model: str):
    """某型號的逐月口碑走勢，給前端畫時間軸圖表。

    純記憶體查表，不呼叫 OpenAI。資料量不足門檻的型號一律回 404，前端就不畫圖
    ——寧可不顯示，也不要用三則評論畫出一條看起來很有結論的曲線。
    """
    data = timeline_store().get(model)
    if data is None:
        raise HTTPException(status_code=404, detail="這個型號沒有足夠的評論可以繪製走勢")
    return data


@app.get("/api/model/{model}/aspects")
@limiter.limit("60/minute")
def model_aspects(request: Request, model: str):
    """某型號的五大面向口碑分數，給前端畫雷達圖。

    純記憶體查表，不呼叫 OpenAI。資料充足的面向不到三個就回 404——兩個軸畫不出
    多邊形，硬畫只會讓人以為系統知道得比實際多。
    """
    data = aspect_store().get(model)
    if data is None:
        raise HTTPException(status_code=404, detail="這個型號沒有足夠的面向評價可以比較")
    return data


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
