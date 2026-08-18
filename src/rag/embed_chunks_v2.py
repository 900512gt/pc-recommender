"""
embed_chunks_v2.py
讀取 data/rag_chunks_v2.jsonl（一型號拆成多個語意 chunk），用 OpenAI embedding
產生向量，寫進本地 Chroma 向量資料庫（data/chroma_db/，collection 名稱
parts_v2），供 retriever_chroma_v2.py 查詢用。

最小單位是「一個 chunk 一筆向量」（不是一個型號一筆），id 用 chunk_id（因為
同一型號有多個 chunk，model 不是唯一鍵），embedding 的輸入直接用 chunk 現成
的 text 欄位（已經是組好的一段話），不需要再另外組合 pros/cons/summary。

使用方式（從專案根目錄執行）：
  python src/rag/embed_chunks_v2.py

依賴：
  pip install chromadb openai python-dotenv
  在專案根目錄建立 .env，寫入：OPENAI_API_KEY=sk-你的金鑰
"""

import json
import os
import sys
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

ROOT        = Path(__file__).parent.parent.parent
CHUNKS_FILE = ROOT / "data" / "rag_chunks_v2.jsonl"
CHROMA_DIR  = ROOT / "data" / "chroma_db"
COLLECTION  = "parts_v2"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 100
EMBED_RETRIES = 3

sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def _embed_batch_with_retry(client: OpenAI, batch: list[str]) -> list[list[float]]:
    """單一批次的 embedding 呼叫，rate limit / 暫時性錯誤時重試，避免整個 script 因為
    一次網路問題就前功盡棄（961 筆全部重跑一次很浪費）。"""
    for attempt in range(EMBED_RETRIES):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            return [item.embedding for item in resp.data]
        except Exception as e:
            is_last = attempt == EMBED_RETRIES - 1
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"    [Rate limit] 等待 {wait}s...")
                if not is_last:
                    time.sleep(wait)
            else:
                print(f"    [錯誤] {e}")
                if not is_last:
                    time.sleep(5)
            if is_last:
                raise
    raise RuntimeError("unreachable")  # for 迴圈一定會 return 或 raise，這行純粹讓型別檢查安心


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vectors.extend(_embed_batch_with_retry(client, batch))
    return vectors


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("請在 .env 設定 OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("載入 chunks...")
    chunks = load_chunks()
    print(f"  共 {len(chunks)} 筆")

    texts = [c["text"] for c in chunks]
    ids   = [c["chunk_id"] for c in chunks]
    # Chroma metadata 值不能是 None，aspect 是 None 時改存空字串
    metas = [
        {
            "category": c["category"],
            "model": c["model"],
            "chunk_type": c["chunk_type"],
            "aspect": c.get("aspect") or "",
        }
        for c in chunks
    ]

    print(f"呼叫 OpenAI embedding（{EMBED_MODEL}）...")
    vectors = embed_texts(client, texts)
    print(f"  完成，共 {len(vectors)} 個向量，維度 {len(vectors[0])}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 每次都重建整個 collection，確保跟 rag_chunks_v2.jsonl 完全同步（不留舊 chunk 的殘留向量）
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    print(f"完成！已寫入 {CHROMA_DIR}，collection={COLLECTION}，共 {collection.count()} 筆")


if __name__ == "__main__":
    main()
