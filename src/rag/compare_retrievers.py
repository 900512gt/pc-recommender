"""
compare_retrievers.py
拿同一組測試查詢，並排比較三套檢索實作：
  - retriever.py          TF-IDF
  - retriever_chroma.py   OpenAI embedding + Chroma（相似度排序 top-k）
  - retriever_fulltext.py 類別全文丟給 LLM 判斷（不排序，回傳整個類別）

使用方式（從專案根目錄執行，需先跑過 python src/rag/embed_chunks.py）：
  python src/rag/compare_retrievers.py
  python src/rag/compare_retrievers.py "你的自訂查詢"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.retriever import Retriever
from src.rag.retriever_chroma import RetrieverChroma
from src.rag.retriever_fulltext import RetrieverFulltext

DEFAULT_QUERIES = [
    "推薦一張適合玩遊戲的顯卡",
    "哪張顯卡比較耐用不容易壞",
    "有沒有性價比高的主機板",
    "適合文書使用的處理器",
    "散熱效果好的水冷推薦",
    "RTX4070怎麼樣",
    "7800X3D 值得買嗎",
    "電源供應器怎麼選比較穩定",
    "我要打電動用的顯卡，預算普通",
]


def describe(chunk: dict) -> str:
    return f"{chunk['model']}（{chunk['category']}）"


def describe_fulltext(chunks: list[dict]) -> str:
    if not chunks:
        return "（無結果）"
    if len(chunks) > 6:
        return f"共 {len(chunks)} 筆（整個類別，例如 {', '.join(c['model'] for c in chunks[:3])} ...）"
    return str([describe(c) for c in chunks])


def main():
    queries = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUERIES

    print("載入 retriever...")
    tfidf    = Retriever()
    chroma   = RetrieverChroma()
    fulltext = RetrieverFulltext()
    print()

    for q in queries:
        tfidf_results    = tfidf.retrieve(q, top_k=2)
        chroma_results   = chroma.retrieve(q, top_k=2)
        fulltext_results = fulltext.retrieve(q, top_k=2)

        print(f"查詢：{q}")
        print(f"  TF-IDF    : {[describe(c) for c in tfidf_results] or '（無結果）'}")
        print(f"  Chroma    : {[describe(c) for c in chroma_results] or '（無結果）'}")
        print(f"  Fulltext  : {describe_fulltext(fulltext_results)}")
        print()


if __name__ == "__main__":
    main()
