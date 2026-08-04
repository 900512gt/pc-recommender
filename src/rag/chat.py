"""
chat.py
RAG 聊天核心：組 prompt、呼叫 LLM、管理對話歷史。
由 server.py 匯入 build_messages 和 chat_stream 使用。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from openai import OpenAI

from src.rag.retriever import Retriever, chunk_to_context

load_dotenv(ROOT / ".env")


def _chunk_to_context(chunk: dict) -> str:
    """v1/v2 chunk 格式不同，依 chunk 形狀分派：v2 chunk 有現成的 text 欄位，
    v1（含 retriever_chroma / retriever_fulltext，三者都讀同一份 v1 格式）沒有，走舊版邏輯。"""
    if "text" in chunk:
        from src.rag.retriever_v2 import chunk_to_context_v2
        return chunk_to_context_v2(chunk)
    return chunk_to_context(chunk)

SYSTEM_PROMPT = """你是一個專門協助台灣使用者選購電腦零件的聊天助理。
你的回答主要依據台灣論壇（PTT、巴哈姆特）的真實使用者評論，以 RAG 方式提供。

回答規則：
- 若有提供「社群評價參考資料」，請優先依據資料內容回答，並自然地融入回答中，不要逐條列出
- 若資料中有提到常被比較的型號，可主動提及做比較
- 若參考資料中多個型號評價相近、沒有明顯優劣之分，請如實告知使用者這幾款評價相近，不要為了給出單一答案而過度武斷
- 若沒有提供參考資料，可用你的背景知識回答，但需說明「以下為一般資訊，非來自論壇評價」
- 回答簡潔、口語化，適合聊天視窗的閱讀習慣
- 使用繁體中文台灣用語（效能、顯示卡、記憶體等）
- 不要在回答中重複或複製參考資料的格式，也不要自行加來源標注"""

MODEL       = "gpt-4o"
MAX_HISTORY = 6


def build_messages(user_query: str, history: list[dict], context_text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_text:
        messages.append({
            "role": "system",
            "content": f"以下是本次查詢的社群評價參考資料：\n\n{context_text}",
        })
    else:
        # 沒有檢索到相關 chunk 時，明確告知 LLM 這個事實並重申免責提醒，
        # 不要只靠它自己記得 SYSTEM_PROMPT 裡那條規則（實測發現常常會忘記加）。
        messages.append({
            "role": "system",
            "content": "本次查詢沒有找到相關的論壇評論資料。請依你的背景知識回答，"
                       "並在回答中明確告知使用者「以下為一般資訊，非來自論壇評價」。",
        })
    messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": user_query})
    return messages


def chat_stream(
    user_query: str,
    history: list[dict],
    retriever: Retriever,
    client: OpenAI,
    temperature: float = 0.7,
    max_tokens: int = 800,
):
    """
    串流版對話 generator：逐步 yield 累積文字。
    history 格式：[{"role": "user"|"assistant", "content": "..."}]
    """
    chunks = retriever.retrieve(user_query, top_k=2)
    context_text = "\n\n".join(_chunk_to_context(c) for c in chunks)
    messages = build_messages(user_query, history, context_text)

    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    accumulated = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        accumulated += delta
        yield accumulated
