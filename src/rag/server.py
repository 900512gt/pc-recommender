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
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import iterate_in_threadpool

from src.rag.chat import chat_stream
from src.rag.retriever import Retriever

load_dotenv(ROOT / ".env")

# 語意搜尋 backend 切換（實驗用）：
#   RAG_RETRIEVER=tfidf      （預設）TF-IDF 版本，讀 v1 chunk（data/rag_chunks.jsonl，一型號一 chunk）
#   RAG_RETRIEVER=chroma     OpenAI embedding + Chroma 版本（v1 chunk），需先執行 python src/rag/embed_chunks.py
#   RAG_RETRIEVER=fulltext   類別全文丟給 LLM 判斷的版本，不需要額外索引
#   RAG_RETRIEVER=v2         TF-IDF 版本，但讀 v2 chunk（data/rag_chunks_v2.jsonl，一型號拆成多個語意 chunk）
#   RAG_RETRIEVER=chroma_v2  OpenAI embedding + Chroma 版本（v2 chunk），需先執行 python src/rag/embed_chunks_v2.py
RETRIEVER_BACKEND = os.environ.get("RAG_RETRIEVER", "tfidf")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="PC 零件口碑 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pc-recommender.vercel.app"],
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
_retriever: Retriever | None = None


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
        if RETRIEVER_BACKEND == "chroma":
            from src.rag.retriever_chroma import RetrieverChroma
            _retriever = RetrieverChroma()
        elif RETRIEVER_BACKEND == "fulltext":
            from src.rag.retriever_fulltext import RetrieverFulltext
            _retriever = RetrieverFulltext()
        elif RETRIEVER_BACKEND == "v2":
            from src.rag.retriever_v2 import RetrieverV2
            _retriever = RetrieverV2()
        elif RETRIEVER_BACKEND == "chroma_v2":
            from src.rag.retriever_chroma_v2 import RetrieverChromaV2
            _retriever = RetrieverChromaV2()
        else:
            _retriever = Retriever()
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

    每個 event 格式：
      data: {"text": "<累積回答>"}\n\n

    最終 sentinel：
      data: [DONE]\n\n

    錯誤時：
      data: {"error": "<訊息>"}\n\n
      data: [DONE]\n\n
    """
    def _sync_gen():
        try:
            for text in chat_stream(
                req.query,
                req.history,
                _get_retriever(),
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


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
