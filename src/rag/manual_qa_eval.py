"""
manual_qa_eval.py
跑一組手動設計的測試問題（涵蓋「指名型號」「開放式推薦」「規格/價格查詢」
「兩型號比較」「離題」五種情境），把每題的檢索結果 + LLM 完整回答整理成一份
Markdown 報告，方便人工比對有沒有出現舊卡推薦、反問預算、價格幻覺、比較偏頗
等已知風險（細節見 docs/RAG_STAGE1_GUIDE.md）。

跟 test_retrieval_v2.py 的差別：那份是 pytest 回歸測試，斷言檢索結果的「型號」
對不對；這份是跑完整 chat_stream（含 gpt-5.5 生成），人工看「回答」寫得好不好，
不斷言正確性，因為這類問題本來就沒有唯一正確答案。

會呼叫 OpenAI API（gpt-5.5 生成 + chroma_v2 的 embedding 查詢），有費用、
每題數秒到數十秒，14 題全跑大約幾分鐘。

使用方式（從專案根目錄執行，需要 .env 設定 OPENAI_API_KEY）：
  python src/rag/manual_qa_eval.py                 # 用正式環境的 chroma_v2 backend
  python src/rag/manual_qa_eval.py --backend v2     # 換成 TF-IDF 版本比較（Chroma 備援）
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from openai import OpenAI

from src.rag.chat import chat_stream, _has_substantive_data

load_dotenv(ROOT / ".env")

CATEGORIES = [
    (
        "類別一：明確指名型號（對照組，預期表現應該不錯）",
        [
            "RTX5070 的散熱表現如何？",
            "A520M 主機板穩不穩定？",
            "4070 適合拿來打電動嗎",  # 故意用非正式簡稱，測別名/簡稱容錯度
        ],
    ),
    (
        "類別二：開放式推薦、沒指名型號（預期可能推到舊卡，或不反問就給篤定答案）",
        [
            "顯卡推薦",
            "兩萬元以內有推薦的顯卡嗎",
            "現在入手哪張顯卡比較划算",
            "我想組一台文書機，主機板該選哪款",
        ],
    ),
    (
        "類別三：純規格/價格查詢（預期出現價格幻覺——資料庫裡沒有即時報價）",
        [
            "RTX5070 現在多少錢",
            "A620 主機板現在還買得到嗎",
        ],
    ),
    (
        "類別四：同時指名兩型號要求比較（測試會不會偏向其中一個）",
        [
            "RTX4070 跟 RTX5070 該選哪個",
            "A520M 跟 A620 主機板差在哪",
        ],
    ),
    (
        "類別五：跟零件完全無關（測試會不會硬拗）",
        [
            "今天天氣如何",
            "幫我寫一首詩",
        ],
    ),
]


def _get_client() -> OpenAI:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("請在 .env 設定 OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def _get_retriever(backend: str):
    if backend == "v2":
        from src.rag.retriever_v2 import RetrieverV2
        return RetrieverV2()
    else:
        from src.rag.retriever_chroma_v2 import RetrieverChromaV2
        return RetrieverChromaV2()


def _describe_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "（無檢索結果）"
    parts = []
    for c in chunks:
        model = c.get("model", "?")
        chunk_type = c.get("chunk_type", "v1")
        confidence = c.get("confidence", "-")
        parts.append(f"{model}[{chunk_type}/{confidence}]")
    return ", ".join(parts)


def ask(query: str, retriever, client: OpenAI) -> dict:
    chunks = retriever.retrieve(query)
    context_used = _has_substantive_data(chunks)

    answer = ""
    for text in chat_stream(query, [], retriever, client):
        answer = text

    return {
        "query": query,
        "chunks_desc": _describe_chunks(chunks),
        "context_used": context_used,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="chroma_v2",
        choices=["v2", "chroma_v2"],
        help="要測試的 retriever backend（預設 chroma_v2，跟正式環境一致）",
    )
    parser.add_argument(
        "--out", default=None,
        help="報告輸出路徑（預設寫到 data/rag_eval_results/ 下，檔名含時間戳）",
    )
    args = parser.parse_args()

    print(f"載入 retriever（backend={args.backend}）...")
    retriever = _get_retriever(args.backend)
    client = _get_client()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else ROOT / "data" / "rag_eval_results" / f"qa_eval_{timestamp}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# RAG 手動測試報告",
        "",
        f"- 時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Retriever backend：`{args.backend}`",
        f"- 模型：`gpt-5.5`",
        "",
    ]

    total = sum(len(qs) for _, qs in CATEGORIES)
    done = 0

    for category_title, questions in CATEGORIES:
        print(f"\n=== {category_title} ===")
        lines.append(f"## {category_title}")
        lines.append("")

        for query in questions:
            done += 1
            print(f"[{done}/{total}] {query} ...", end=" ", flush=True)
            t0 = time.time()
            result = ask(query, retriever, client)
            elapsed = time.time() - t0
            print(f"完成（{elapsed:.1f}s）")

            lines.append(f"### 問：{result['query']}")
            lines.append("")
            lines.append(f"- 檢索到的 chunk：{result['chunks_desc']}")
            lines.append(f"- 是否有實質論壇資料可用：{'是' if result['context_used'] else '否（走一般知識 fallback）'}")
            lines.append("")
            lines.append("**答：**")
            lines.append("")
            lines.append(result["answer"] or "（空回覆）")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n報告已寫入：{out_path}")


if __name__ == "__main__":
    main()
