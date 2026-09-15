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

load_dotenv(ROOT / ".env")


def _chunk_to_context(chunk: dict) -> str:
    """把 chunk 轉成給 LLM 讀的文字段落（v2 chunk 格式，有現成的 text 欄位）。"""
    from src.rag.retriever_v2 import chunk_to_context_v2
    return chunk_to_context_v2(chunk)


def _has_substantive_data(chunks: list[dict]) -> bool:
    """v2 chunk 在 confidence="insufficient" 時，除了 summary 是固定的
    「評論數量過少，無法產生可靠摘要」外，pros/cons/aspects 全是空的，等於沒有
    真正的論壇依據可用。實測過這種情況下 LLM 反而會更放心地自己編產品定位、
    推薦資料庫裡根本沒有的其他型號，卻不標注「非來自論壇評價」——因為它收到的
    訊號是「有 chunk」，不是「沒資料」。所以這裡要跟「完全沒檢索到 chunk」同樣
    處理，不能讓 LLM 誤以為手上有可用的論壇資料。
    （沒有 confidence 欄位的 chunk 一律回傳 True，行為不受影響，是防禦性寫法。）"""
    if not chunks:
        return False
    return not all(c.get("confidence") == "insufficient" for c in chunks)

SYSTEM_PROMPT = """你是一個提供台灣 PC 零件「社群口碑查詢」的聊天助理，資料來源是 PTT、
巴哈姆特上的真實使用者評論。

你的定位是回答「這個型號評價如何」「這幾款比較起來風評怎麼樣」這類口碑查詢，
不是配置推薦引擎——你看到的是各型號各自獨立的社群評價片段，沒有能力做嚴謹的
效能排序、性價比計算，或是考慮預算/相容性的整體配置最佳化，那是本站另一個
「配置建置」功能（GA 推薦引擎）在做的事。

回答規則：
- 若有提供「社群評價參考資料」，請優先依據資料內容回答，並自然地融入回答中，不要逐條列出
- 若資料中有提到常被比較的型號，可主動提及做比較
- 若參考資料中多個型號評價相近、沒有明顯優劣之分，請如實告知使用者這幾款評價相近，不要為了給出單一答案而過度武斷
- 若使用者問的是廣泛的「推薦」「選哪個好」，可以根據手上的口碑資料給出傾向，但要清楚說明這是「根據目前論壇討論整理的印象」，不是效能實測或性價比排序；如果使用者聽起來需要考慮預算、相容性的完整配置建議，提醒他們可以用本站的「配置建置」功能
- 若沒有提供參考資料，可用你的背景知識回答，但需說明「以下為一般資訊，非來自論壇評價」
- 回答簡潔、口語化，適合聊天視窗的閱讀習慣
- 使用繁體中文台灣用語（效能、顯示卡、記憶體等）
- 不要在回答中重複或複製參考資料的格式，也不要自行加來源標注"""

MODEL       = "gpt-5.5"
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
    chunks: list[dict],
    client: OpenAI,
    max_tokens: int = 2000,
):
    """
    串流版對話 generator：逐步 yield 累積文字。
    history 格式：[{"role": "user"|"assistant", "content": "..."}]

    chunks 由呼叫端先 retrieve 好再傳進來——呼叫端本來就需要那份 chunk（server.py
    要拿去查佐證評論、manual_qa_eval.py 要拿去列檢索結果），檢索又會呼叫 LLM 抽型號，
    留在這裡面會變成同一個問題檢索兩次。

    MODEL（gpt-5.5）是推理模型：不支援自訂 temperature（只吃預設值 1，
    傳其他值會直接 400），所以這裡不傳 temperature 給 API；另外它的
    max_completion_tokens 預算包含隱藏的推理 token，簡短問題也可能吃掉
    幾百個 token 才開始輸出可見文字，預設值比舊版 gpt-4o 的 800 高很多。
    """
    if _has_substantive_data(chunks):
        context_text = "\n\n".join(_chunk_to_context(c) for c in chunks)
    else:
        context_text = ""
    messages = build_messages(user_query, history, context_text)

    stream = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=max_tokens,
        messages=messages,
        stream=True,
    )

    accumulated = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        accumulated += delta
        yield accumulated
