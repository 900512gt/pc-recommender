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
from unittest.mock import patch

from src.rag.retriever_v2 import RetrieverV2
from src.rag.retriever_chroma_v2 import RetrieverChromaV2, CATEGORY_VARIANTS
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


# ── LLM 型號抽取＋類別分類（取代字串比對／CATEGORY_KEYWORDS 關鍵字比對）──
# 這幾條是 2026-08-18 那次把 _match_models 字串比對、CATEGORY_KEYWORDS 關鍵字
# 比對都換成 LLM 判斷之後，手動驗證過的邊界案例。當時都是臨時在 shell 裡
# 跑腳本驗證，沒有留下永久回歸測試——如果之後又調整 prompt 或抽取邏輯，這些
# 案例會悄悄退步而不會有任何警告，所以補進來固定下來。

def test_chroma_v2_dedups_by_model_across_semantic_search(chroma_v2):
    """曾經發生：top_k 用「chunk 數量」截斷，同一型號的多則 chunk 佔滿候選，
    導致「顯卡推薦」這種開放式問題只回傳 1~2 個型號的零碎片段。確認依型號
    去重後，能湊到 top_k 個不同型號，且每個型號都是完整 chunk 組（不是只有
    一兩則）。"""
    results = chroma_v2.retrieve("顯卡推薦", top_k=3)
    models = {c["model"] for c in results}
    assert len(models) == 3, f"應該湊到 3 個不同型號，實際: {models}"
    for model in models:
        count = sum(1 for c in results if c["model"] == model)
        assert count >= 4, f"{model} 只有 {count} 則 chunk，應該是完整 chunk 組"


def test_chroma_v2_excludes_negated_model(chroma_v2):
    """曾經發生：「請推薦除了 RTX5070 以外的顯卡」因為句子裡包含「5070」
    這個字串，被舊版純字串比對誤判成「使用者在問 RTX5070」，直接回傳它的
    完整資料，跟使用者真實意圖（排除這張卡）相反。確認 LLM 抽取的極性
    判斷能正確處理「除了 X 以外」這種否定語意，RTX5070 不該出現在結果裡。"""
    results = chroma_v2.retrieve("請推薦除了 RTX5070 以外的顯卡", top_k=3)
    models = {c["model"].lower() for c in results}
    assert "rtx5070" not in models, f"RTX5070 不該出現，實際型號: {models}"
    assert len(results) > 0, "排除 5070 之後應該還有其他顯卡可以推薦"


def test_chroma_v2_comparison_includes_both_models(chroma_v2):
    """曾經發生：prompt 沒講清楚時，「A 跟 B 該選哪個」這種比較句型會被 LLM
    誤判成 B 是 exclude（把「二選一」誤解成「排除其中一個」）。確認比較句型
    兩個型號都判斷成 include，各自回傳完整資料，不會偏袒任何一邊。"""
    results = chroma_v2.retrieve("RTX4070 跟 RTX5070 該選哪個", top_k=3)
    models = {c["model"].lower() for c in results}
    assert "rtx4070" in models and "rtx5070" in models
    for model in ("rtx4070", "rtx5070"):
        count = sum(1 for c in results if c["model"].lower() == model)
        assert count >= 4, f"{model} 只有 {count} 則 chunk，應該是完整 chunk 組"


def test_chroma_v2_compound_include_and_exclude(chroma_v2):
    """複合案例：同一句話同時有比較（include）跟排除（exclude），確認兩種
    極性判斷不會互相干擾——4070/5070 該完整回傳，3050 不該出現。"""
    results = chroma_v2.retrieve(
        "RTX4070 跟 RTX5070 選一個，但不要跟我推薦 RTX3050", top_k=3
    )
    models = {c["model"].lower() for c in results}
    assert "rtx4070" in models and "rtx5070" in models
    assert "rtx3050" not in models, f"RTX3050 不該出現，實際型號: {models}"


def test_chroma_v2_normalizes_colloquial_model_alias(chroma_v2):
    """「4070」這種只講數字的口語簡稱，LLM 抽取時要能正規化回「RTX4070」
    才查得到資料（型號清單的 enum 只收正式名稱，簡稱查不到會直接漏掉）。"""
    results = chroma_v2.retrieve("4070 適合拿來打電動嗎", top_k=3)
    models = {c["model"].lower() for c in results}
    assert "rtx4070" in models


def test_chroma_v2_recognizes_colloquial_category_synonym(chroma_v2):
    """曾經發生兩層問題：① 舊版 CATEGORY_KEYWORDS 關鍵字表沒收錄「獨顯」，
    完全不篩類別；② 類別分類換成 LLM 判斷、正確辨識出 GPU 後，候選池變窄、
    distance 分布右移，MAX_DISTANCE 舊門檻反而會把這題誤判離題、回傳空
    結果。兩層都修過後，這題應該正常回傳 GPU 型號，不是空清單。"""
    results = chroma_v2.retrieve("獨顯選哪張比較好", top_k=3)
    assert len(results) > 0, "「獨顯」是口語同義詞，不該被當成離題問題拒答"
    categories = {c["category"] for c in results}
    assert categories == {"GPU"}, f"應該只有 GPU 類別，實際: {categories}"


def test_chroma_v2_category_filter_avoids_substring_false_positive(chroma_v2):
    """曾經發生：舊版關鍵字比對用裸字串比對，"mb" 會被藏在 "8MB" 這種數字
    後綴裡誤觸發，讓問顯卡的問題被誤篩進主機板類別。確認換成 LLM 判斷後，
    問「顯卡」不會被句子裡的 "8MB" 字樣污染成主機板類別。"""
    results = chroma_v2.retrieve("這張顯卡有8MB快取，正常嗎", top_k=3)
    assert len(results) > 0
    categories = {c["category"] for c in results}
    assert "主機板" not in categories and "MB" not in categories, (
        f"不該誤觸發主機板類別，實際類別: {categories}"
    )


def test_chroma_v2_hdd_category_removed():
    """這個系統不推薦傳統硬碟，HDD 不該出現在可分類的類別清單裡，LLM 不會
    知道有這個選項存在。純邏輯檢查，不呼叫 API。"""
    assert "HDD" not in CATEGORY_VARIANTS


def test_chroma_v2_falls_back_to_keyword_matching_when_llm_fails(chroma_v2):
    """LLM 型號＋類別抽取失敗時（例如網路問題），不該讓整個 retrieve() 掛掉，
    要 fallback 回舊版的純字串比對／CATEGORY_KEYWORDS 關鍵字比對安全網。
    用 mock 模擬抽取失敗（回傳 None），不需要真的斷網路。"""
    with patch.object(chroma_v2, "_extract_query_intents", return_value=None):
        # 指名型號的問題，安全網應該退回 _match_models 純字串比對
        exact = chroma_v2.retrieve("RTX5070 的散熱表現如何？", top_k=3)
        assert {c["model"].lower() for c in exact} == {"rtx5070"}

        # 開放式推薦問題，安全網應該正常跑語意搜尋（類別用關鍵字猜測）
        broad = chroma_v2.retrieve("顯卡推薦", top_k=3)
        assert len(broad) > 0


# ── is_pc_part_question：LLM 直接判斷離題，取代不可靠的距離門檻 ─────
# 原本完全離題靠 MAX_DISTANCE 事後判斷，但門檻本身承認不可靠（滑鼠/鍵盤
# 這類主題邊緣查詢的 distance 落在 0.60~0.61，跟「模糊但主題內」的合理
# 查詢 0.53~0.61 完全重疊）。2026-08-18 改成讓同一次型號/類別抽取的 LLM
# 呼叫直接判斷「這句話跟 PC 零件是否相關」，true 才會走語意搜尋。

def test_chroma_v2_rejects_non_pc_part_question_without_embedding_call(chroma_v2):
    """跟 PC 零件完全無關的問題，LLM 應該判斷 is_pc_part_question=False，
    retrieve() 直接短路回傳空清單，連 embedding API 都不該呼叫——用 mock
    _embed_query 斷言沒被呼叫，確認真的有短路，不是繞了一圈才回空清單。"""
    for query in ["滑鼠選哪個好", "今天天氣如何", "幫我寫一首詩", "台股大盤今天多少"]:
        with patch.object(
            chroma_v2, "_embed_query",
            side_effect=AssertionError(f"「{query}」不該呼叫 embedding API"),
        ):
            assert chroma_v2.retrieve(query, top_k=3) == []


def test_chroma_v2_broad_build_question_is_still_pc_part_related(chroma_v2):
    """曾經要特別區分：「五萬預算配一台電腦」這種廣泛型整機問題，
    categories 會是空陣列（沒有限定單一類別），但不能因此被誤判成離題——
    它仍然是 PC 零件相關問題，is_pc_part_question 要是 True，正常走語意
    搜尋、有結果，不能被新加的離題短路擋掉。"""
    intents = chroma_v2._extract_query_intents("五萬預算配一台電腦")
    assert intents["categories"] == []
    assert intents["is_pc_part_question"] is True

    results = chroma_v2.retrieve("五萬預算配一台電腦", top_k=3)
    assert len(results) > 0


# ── 邊界輸入（空字串／top_k 極端值）──────────────────────────────
# 跟上面那組「已知 bug 情境」不同，這組是純粹的程式邊界值測試（空輸入、
# 極端數值），不是在還原特定使用者問法。

def test_chroma_v2_empty_query_returns_empty_list(chroma_v2):
    """曾經發生：空字串會讓 _embed_query("") 對 OpenAI embeddings API 丟出
    400 BadRequestError（該 API 明確拒絕空字串輸入），例外沒被接住、直接
    往上拋，整個 retrieve() 會掛掉，不是優雅回傳空清單。"""
    assert chroma_v2.retrieve("", top_k=3) == []


def test_chroma_v2_whitespace_only_query_returns_empty_list(chroma_v2):
    """純空白不會讓 embeddings API 報錯（跟真正的空字串不同），但語意上
    一樣是「沒有內容可以搜尋」，行為應該跟空字串一致，不該真的去跑一次
    語意搜尋。"""
    assert chroma_v2.retrieve("   ", top_k=3) == []


def test_chroma_v2_top_k_zero_returns_empty_list(chroma_v2):
    """曾經發生：語意搜尋掃描迴圈是「先加進候選、再檢查有沒有湊滿
    top_k」，top_k=0 時這個檢查永遠在加了 1 個候選之後才觸發，會多回傳
    1 個型號，而不是語意上「要 0 個」該有的空清單。"""
    assert chroma_v2.retrieve("顯卡推薦", top_k=0) == []


def test_chroma_v2_negative_top_k_returns_empty_list(chroma_v2):
    """負數 top_k 沒有合理語意，應該當成「不要任何候選」處理，回傳空清單，
    而不是報錯或表現出未定義行為。"""
    assert chroma_v2.retrieve("顯卡推薦", top_k=-1) == []


def test_chroma_v2_top_k_larger_than_available_models_degrades_gracefully(chroma_v2):
    """top_k 超過該類別實際可用的型號數量時，應該盡量湊、如實回傳目前湊到
    的數量（可能小於 top_k），不能報錯、卡住，也不能為了湊滿數量放寬距離
    門檻硬湊進不相關的型號。"""
    results = chroma_v2.retrieve("電源供應器有推薦的型號嗎", top_k=100)
    models = {c["model"] for c in results}
    assert 0 < len(models) < 100, f"應該回傳合理數量（少於要求的 100），實際: {len(models)}"


def test_has_substantive_data_treats_insufficient_as_empty():
    """單元測試 chat.py 的判斷邏輯，不需要真的呼叫 LLM，秒級跑完。"""
    insufficient_chunk = {"confidence": "insufficient", "chunk_type": "summary", "text": "x"}
    real_chunk = {"confidence": "high", "chunk_type": "summary", "text": "y"}
    no_confidence_chunk = {"model": "x"}  # 沒有 confidence 欄位（防禦性案例）

    assert _has_substantive_data([real_chunk]) is True
    assert _has_substantive_data([insufficient_chunk]) is False
    assert _has_substantive_data([insufficient_chunk, real_chunk]) is True  # 只要有一個有實質內容就算
    assert _has_substantive_data([]) is False
    assert _has_substantive_data([no_confidence_chunk]) is True  # 沒有 confidence 欄位時不受影響


if __name__ == "__main__":
    # 沒裝 pytest／想快速跑一次也可以直接執行這個檔案
    sys.exit(pytest.main([__file__, "-v"]))
