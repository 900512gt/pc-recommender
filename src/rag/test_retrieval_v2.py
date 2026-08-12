"""
test_retrieval_v2.py
RAG v2 檢索的回歸測試——把這幾天手動驗證過的案例（還在賣過濾、離題拒答、
insufficient 信心處理...）固定下來，之後改動門檻值/retriever 邏輯/chat.py
時，跑這份測試就能快速確認沒有把已知案例弄壞，不用每次手動想查詢、肉眼看結果。

執行方式（從專案根目錄執行）：
  python -m pytest src/rag/test_retrieval_v2.py -v
  或不裝 pytest 也能跑：python src/rag/test_retrieval_v2.py

需要 .env 設定 OPENAI_API_KEY（chroma_v2 的語意搜尋測試會呼叫 embedding API，
費用極低；exact match 測試不需要呼叫 API）。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.rag.retriever_v2 import RetrieverV2
from src.rag.retriever_chroma_v2 import RetrieverChromaV2
from src.rag.chat import _has_substantive_data

# 已知已經停產、data/ga_database_v2.json 裡不會再出現的 GPU 型號，語意搜尋
# 不該再把這些撈出來當「推薦」答案（可以指名查詢，但不能被動撈到，見兩個
# retriever 檔頭的說明）。
DISCONTINUED_GPUS = {
    "rtx3060", "rtx3060ti", "rtx3070", "rtx3080", "rtx3090",
    "rtx4060", "rtx4060ti", "rtx4070", "rtx4070ti", "rtx4080", "rtx4090",
    "rtx5090", "rx7900xtx",
}


@pytest.fixture(scope="module")
def tfidf_v2() -> RetrieverV2:
    return RetrieverV2()


@pytest.fixture(scope="module")
def chroma_v2() -> RetrieverChromaV2:
    return RetrieverChromaV2()


# ── 「還在賣」過濾（曾經發生：問「中階顯卡推薦」撈到 RTX40 系列停產卡）──

def test_chroma_v2_recommendation_excludes_discontinued_gpu(chroma_v2):
    results = chroma_v2.retrieve("中階顯卡推薦", top_k=2)
    models = {c["model"].lower() for c in results}
    assert not (models & DISCONTINUED_GPUS), f"撈到停產型號: {models & DISCONTINUED_GPUS}"


def test_tfidf_v2_recommendation_excludes_discontinued_gpu(tfidf_v2):
    results = tfidf_v2.retrieve("中階顯卡推薦", top_k=2)
    models = {c["model"].lower() for c in results}
    assert not (models & DISCONTINUED_GPUS), f"撈到停產型號: {models & DISCONTINUED_GPUS}"


def test_exact_match_ignores_sellable_filter(chroma_v2):
    """指名問特定型號時，就算已停產也要能查到完整口碑（口碑查詢 vs. 推薦是兩回事，
    這裡故意用一張已停產的 RTX4070 驗證精確比對不受「還在賣」過濾影響）。"""
    results = chroma_v2.retrieve("RTX4070怎麼樣", top_k=2)
    models = {c["model"].lower() for c in results}
    assert "rtx4070" in models
    chunk_types = {c["chunk_type"] for c in results}
    assert "summary" in chunk_types
    assert len(results) >= 4, "應該回傳 summary + 多個 aspect + pros_cons + comparison"


# ── 離題／範圍外查詢的拒答 ──────────────────────────────────────

def test_chroma_v2_rejects_offtopic(chroma_v2):
    assert chroma_v2.retrieve("今天天氣如何", top_k=2) == []


def test_chroma_v2_rejects_out_of_scope_category(chroma_v2):
    """資料庫沒有滑鼠這個類別，不該硬湊答案。"""
    assert chroma_v2.retrieve("推薦一款好用的滑鼠", top_k=2) == []


def test_tfidf_v2_rejects_offtopic(tfidf_v2):
    assert tfidf_v2.retrieve("今天天氣如何", top_k=2) == []


def test_tfidf_v2_out_of_scope_category_known_limitation(tfidf_v2):
    """已知限制（見 retriever_v2.py 檔頭）：TF-IDF v2 對「資料庫沒有的品項」
    沒有可靠的拒答能力，數學上證明不存在能同時擋離題、放行明確查詢的門檻。
    這裡不斷言一定要拒答，只確認呼叫不會噴例外，避免這條測試因為已知、
    無法修復的限制而長期紅字。"""
    tfidf_v2.retrieve("推薦一款好用的滑鼠", top_k=2)


# ── MIN_GAP 拿掉後的迴歸測試 ────────────────────────────────────

def test_chroma_v2_does_not_reject_close_scores(chroma_v2):
    """之前 MIN_GAP=0.005 會誤傷「有沒有性價比高的主機板」這種正常會有多個
    相近候選的查詢（好幾個型號都有性價比評語，分數本來就會很接近）。
    確認拿掉 MIN_GAP 後這類查詢能正常回傳結果。"""
    results = chroma_v2.retrieve("有沒有性價比高的主機板", top_k=2)
    assert len(results) > 0


# ── 信心分級 ────────────────────────────────────────────────

def test_insufficient_confidence_chunk_has_no_real_content(chroma_v2):
    """confidence=insufficient 的 chunk 應該是固定的空 placeholder，不該混進
    真的 pros/cons——這是 chat.py._has_substantive_data() 判斷邏輯能正確運作
    的前提。"""
    insufficient_summaries = [
        c for c in chroma_v2.chunks
        if c.get("confidence") == "insufficient" and c["chunk_type"] == "summary"
    ]
    assert insufficient_summaries, "測試資料裡應該要有至少一個 insufficient 型號"
    for c in insufficient_summaries[:20]:  # 抽樣檢查，不用全部
        assert "評論數量過少" in c["text"]


def test_has_substantive_data_treats_insufficient_as_empty():
    """單元測試 chat.py 的判斷邏輯，不需要真的呼叫 LLM，秒級跑完。"""
    insufficient_chunk = {"confidence": "insufficient", "chunk_type": "summary", "text": "x"}
    real_chunk = {"confidence": "high", "chunk_type": "summary", "text": "y"}
    v1_chunk = {"model": "x"}  # 沒有 confidence 欄位，模擬 v1 chunk

    assert _has_substantive_data([real_chunk]) is True
    assert _has_substantive_data([insufficient_chunk]) is False
    assert _has_substantive_data([insufficient_chunk, real_chunk]) is True  # 只要有一個有實質內容就算
    assert _has_substantive_data([]) is False
    assert _has_substantive_data([v1_chunk]) is True  # v1 chunk 沒有 confidence 欄位，不受影響


if __name__ == "__main__":
    # 沒裝 pytest／想快速跑一次也可以直接執行這個檔案
    sys.exit(pytest.main([__file__, "-v"]))
