"""
RAG chunk distillation: 將每個零件型號的 PTT + 巴哈評論，透過 LLM 蒸餾成結構化摘要，
再拆成多個語意 chunk（summary / aspect / pros_cons / comparison）。

輸出：
  data/distilled_models_v2.jsonl  每行一個型號的完整蒸餾結果＋統計，供除錯用
  data/rag_chunks_v2.jsonl        每行一個 chunk，供 embedding / 檢索用

使用方式（從專案根目錄執行）：
  python src/rag/distill_chunks.py --test                    # 測試單一型號 RTX4070
  python src/rag/distill_chunks.py --test "AMD R7 7800X3D"   # 測試指定型號
  python src/rag/distill_chunks.py --limit-models 2          # 只跑前 2 個待處理型號（小規模驗證用）
  python src/rag/distill_chunks.py                           # 跑全部型號

依賴：
  pip install openai python-dotenv
  在專案根目錄建立 .env，寫入：OPENAI_API_KEY=sk-你的金鑰
"""

import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv



load_dotenv()

# ── 路徑設定 ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
BAHA_FILES = [ROOT / f"src/database/input/baha_comment/matched_part{i}.jsonl" for i in [1, 2, 3]]
PTT_FILE   = ROOT / "src/database/input/ptt_comment/relevant.jsonl"
RAG_CHUNKS_V2        = ROOT / "data/rag_chunks_v2.jsonl"
DISTILLED_MODELS_V2  = ROOT / "data/distilled_models_v2.jsonl"

# ── 抽樣參數 ──────────────────────────────────────────────
MIN_COMMENT_LENGTH = 6    # 評論最短字數（原本 15 太嚴格，會誤刪「線圈音明顯」「溫度很高」這類短但有資訊的評論）
SAMPLES_PER_LABEL  = 50   # 每個 label × 每個來源最多取幾則
MIN_COMMENTS       = 5    # 有效評論數低於此數視為 insufficient，不呼叫 LLM

LABEL_VALUES = ("正評", "負評", "中立")

# 無資訊量的短句，即使超過 MIN_COMMENT_LENGTH 也要濾掉（去除頭尾標點後精確比對）
LOW_INFO_PHRASES = {
    "推", "+1", "推+1", "哈哈", "哈哈哈", "笑死", "路過", "樓下", "已買", "已入手", "簽到", "同意",
}
_TRAILING_PUNCT = "！!~～。，,.…"

# 原始資料的 category 欄位有時是英文代碼（GPU/MB/...），有時已經是中文（主機板/記憶體/...），
# 這裡統一映射成 (chunk_id 用的 slug, chunk text 用的中文顯示名稱)。
CATEGORY_INFO: dict[str, tuple[str, str]] = {
    "GPU": ("gpu", "顯示卡"), "顯示卡": ("gpu", "顯示卡"),
    "CPU": ("cpu", "處理器"), "處理器": ("cpu", "處理器"),
    "SSD": ("ssd", "固態硬碟"), "固態硬碟": ("ssd", "固態硬碟"),
    "MB": ("mb", "主機板"), "主機板": ("mb", "主機板"),
    "PSU": ("psu", "電源"), "電源": ("psu", "電源"),
    "RAM": ("ram", "記憶體"), "記憶體": ("ram", "記憶體"),
    "HDD": ("hdd", "硬碟"), "硬碟": ("hdd", "硬碟"),
    "AIR_COOLER": ("air_cooler", "風冷"), "風冷": ("air_cooler", "風冷"),
    "WATER_COOLER": ("water_cooler", "水冷"), "水冷": ("water_cooler", "水冷"),
    "CASE": ("case", "機殼"), "機殼": ("case", "機殼"),
}

# ── LLM 設定 ──────────────────────────────────────────────
MODEL      = "gpt-4o-mini"
MAX_TOKENS = 1200
DISTILLATION_VERSION = "v2"

SYSTEM_PROMPT = """你是一位電腦硬體評論分析師，專門整理台灣電腦組裝論壇（PTT、巴哈姆特）的使用者評論。
你的任務是將一組關於特定零件型號的評論，蒸餾成結構化的繁體中文摘要，供 RAG 系統回答使用者問題。

輸出必須是合法的 JSON，欄位如下：
{
  "pros": ["正面評價1", "正面評價2", ...],          // 3~6 點，保留具體感受
  "cons": ["負面評價1", "負面評價2", ...],          // 3~6 點，保留具體感受
  "aspects": {                                      // 根據零件類別選取相關面向，2~5 個
    "面向名稱": "社群的整體看法（一句話）",
    ...
  },
  "comparisons": ["常被拿來比較的型號1", ...],      // 0~4 個，沒有就空陣列
  "summary": "一段 50~120 字的整體評價摘要"         // 綜合正負評，呈現社群共識與主要爭議
}

規則：
- 只根據提供的評論內容作答，不要補充你自己的產品知識
- pros/cons 每點以「動詞或名詞開頭的短句」呈現，不加序號
- aspects 的 key 應反映該類別最重要的面向（GPU 可用：遊戲效能、價格感受、溫度、驅動穩定性；CPU 可用：效能表現、功耗發熱、超頻潛力、性價比；SSD 可用：讀寫速度、耐用性、發熱；依此類推）
- summary 要平衡呈現，避免過度正面或負面
- 使用台灣繁體中文慣用詞彙（例如：效能、記憶體、顯示卡、主機板、散熱器，避免使用性能、内存、显卡、主板、散热器等簡體或中國用語）
- 下面的評論是分層抽樣後的樣本，prompt 開頭會附上抽樣前的原始評論分布統計；判斷整體口碑比例時請以該統計為準，不要只用樣本則數的比例
- 不要把少數意見描述成普遍共識
- 若評論之間對同一面向意見分歧（例如有人說穩定、有人說常當機），summary 或對應的 aspects 要明確點出這種分歧，不要含糊帶過或只挑單一立場
- 不得推論評論沒有明確提到的原因（例如評論只說「很吵」，不要自行判斷是風扇、幫浦或其他零件造成）
- 輸出純 JSON，不加任何 markdown 或說明文字"""


def load_baha() -> list[dict]:
    records = []
    for path in BAHA_FILES:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["source"] = "bahamut"
                records.append(r)
    return records


def load_ptt() -> list[dict]:
    records = []
    with open(PTT_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["source"] = "ptt"
            records.append(r)
    return records


def normalize_label(label: str) -> str:
    mapping = {"负評": "負評", "null": "中立", None: "中立", "": "中立"}
    return mapping.get(label, label)


def group_by_model(records: list[dict]) -> dict[str, dict]:
    """回傳 {model: {category, comments: [...]}}"""
    groups: dict[str, dict] = {}
    for r in records:
        model = r["model"]
        if model not in groups:
            groups[model] = {"category": r["category"], "comments": []}
        groups[model]["comments"].append(r)
    return groups


def _is_low_info(content: str) -> bool:
    """去除頭尾空白/標點後，整句是否等於 LOW_INFO_PHRASES 裡的無資訊短句（如「推」「+1」）。"""
    stripped = content.strip().strip(_TRAILING_PUNCT).strip()
    return stripped in LOW_INFO_PHRASES


def _filter_valid_comments(comments: list[dict]) -> list[dict]:
    """套用長度門檻、無資訊短句過濾、合法標籤三個條件，回傳可用於抽樣的評論。"""
    valid = []
    for c in comments:
        content = c.get("content", "")
        if len(content) < MIN_COMMENT_LENGTH:
            continue
        if _is_low_info(content):
            continue
        if normalize_label(c.get("label")) not in LABEL_VALUES:
            continue
        valid.append(c)
    return valid


def stratified_sample(valid: list[dict]) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """
    依 source × label 分層抽樣，每格最多 SAMPLES_PER_LABEL 則。
    回傳 (抽樣結果, 每層的 population/sampled 筆數)，後者供 build_distribution_stats 使用，
    讓 LLM prompt 跟輸出 chunk 都能參考「抽樣前」的原始比例。
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for c in valid:
        key = (c["source"], normalize_label(c["label"]))
        buckets[key].append(c)

    sampled: list[dict] = []
    strata: dict[str, dict[str, int]] = {}
    for (source, label), bucket in buckets.items():
        random.shuffle(bucket)
        picked = bucket[:SAMPLES_PER_LABEL]
        sampled.extend(picked)
        strata[f"{source}_{label}"] = {
            "population_count": len(bucket),
            "sampled_count": len(picked),
        }

    return sampled, strata


def build_distribution_stats(
    comments: list[dict], valid: list[dict], sampled: list[dict],
    strata: dict[str, dict[str, int]],
) -> dict:
    """統計抽樣前後的分布，讓 LLM prompt 與輸出 chunk 都能參考原始比例，而不只看抽樣後的樣本則數。"""
    source_dist: dict[str, int] = defaultdict(int)
    label_dist: dict[str, int] = defaultdict(int)
    for c in comments:
        source_dist[c["source"]] += 1
        label_dist[normalize_label(c.get("label"))] += 1

    return {
        "total_raw_comments": len(comments),
        "total_valid_comments": len(valid),
        "sampled_comment_count": len(sampled),
        "source_distribution": dict(source_dist),
        "label_distribution": dict(label_dist),
        "strata_distribution": strata,
    }


def format_distribution_for_prompt(stats: dict) -> str:
    """把統計數字整理成一段文字，放進 LLM prompt 開頭。"""
    lines = [
        "原始評論分布（抽樣前，涵蓋全部相關評論，不只是下面列出的樣本）：",
        f"- 原始評論總數：{stats['total_raw_comments']} 則"
        f"（過濾過短/無資訊短句後有效：{stats['total_valid_comments']} 則，"
        f"分層抽樣後實際附在下方的樣本：{stats['sampled_comment_count']} 則）",
        "- 來源分布：" + "、".join(f"{k} {v} 則" for k, v in stats["source_distribution"].items()),
        "- 情緒分布：" + "、".join(f"{k} {v} 則" for k, v in stats["label_distribution"].items()),
    ]
    if stats["strata_distribution"]:
        lines.append("- 分層抽樣明細（母體 → 抽樣）：")
        for key, counts in stats["strata_distribution"].items():
            lines.append(f"  - {key}：{counts['population_count']} 則 → {counts['sampled_count']} 則")
    lines.append(
        "（以上是完整資料的分布，下面的評論只是分層抽樣後的樣本，用來呈現具體語句；"
        "判斷整體風向、比例時請以上面的原始分布為準，不要只看下面樣本的則數）"
    )
    return "\n".join(lines)


def classify_confidence(valid_count: int, source_counts: dict[str, int]) -> tuple[str, list[str]]:
    """依有效評論數分四級信心：insufficient(<5) / low(5~19) / medium(20~49) / high(50+)。
    若有效評論只來自單一來源，額外在 confidence_reasons 標記 only_one_source。"""
    if valid_count < MIN_COMMENTS:
        level = "insufficient"
    elif valid_count < 20:
        level = "low"
    elif valid_count < 50:
        level = "medium"
    else:
        level = "high"

    reasons: list[str] = []
    nonzero_sources = [src for src, cnt in source_counts.items() if cnt > 0]
    if len(nonzero_sources) == 1:
        reasons.append("only_one_source")

    return level, reasons


def _parse_date(date_str: str | None) -> str | None:
    """巴哈跟 PTT 的日期格式不同，統一解析成 YYYY-MM-DD；解析失敗回傳 None，不中斷整體流程。"""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _evidence_id(c: dict) -> str | None:
    """組出可重現的評論識別碼：原始資料沒有獨立的評論 ID，用 url + 樓層（巴哈）/ url + 原始索引
    （PTT）組合出來；缺 url 就放棄（不捏造）。"""
    url = c.get("url")
    if not url:
        return None
    anchor = c.get("floor") or c.get("_index")
    return f"{url}#{anchor}" if anchor is not None else url


def compute_evidence_ids(sampled: list[dict]) -> list[str]:
    return [i for i in (_evidence_id(c) for c in sampled) if i]


def compute_date_range(sampled: list[dict]) -> tuple[str | None, str | None]:
    dates = [d for d in (_parse_date(c.get("date")) for c in sampled) if d]
    if not dates:
        return None, None
    return min(dates), max(dates)


def _slugify(text: str) -> str:
    """把任意字串轉成穩定、可重現的 id 片段：小寫化，非英數/CJK 字元轉底線。"""
    text = text.strip().lower()
    text = re.sub(r"[^0-9a-z一-鿿]+", "_", text)
    return text.strip("_") or "unknown"


def _category_slug(category: str) -> str:
    return CATEGORY_INFO.get(category, (_slugify(category), category))[0]


def _category_zh(category: str) -> str:
    return CATEGORY_INFO.get(category, (_slugify(category), category))[1]


def make_chunk_id(category: str, model: str, chunk_type: str, aspect: str | None = None) -> str:
    """建立穩定、可重現的 chunk_id，如 gpu__rtx4070__summary、gpu__rtx4070__aspect__遊戲效能。"""
    cat_slug = _category_slug(category)
    model_slug = _slugify(model)
    if aspect:
        return f"{cat_slug}__{model_slug}__{chunk_type}__{_slugify(aspect)}"
    return f"{cat_slug}__{model_slug}__{chunk_type}"


def build_rag_chunks(model: str, category: str, distilled: dict, meta: dict) -> list[dict]:
    """
    把單一型號的一次 LLM 蒸餾結果，拆成多個語意獨立的 RAG chunk（summary / aspect / pros_cons / comparison）。
    每個 chunk 的 text 都帶入型號名稱與類別，避免拆分後失去上下文。
    """
    cat_zh = _category_zh(category)
    chunks: list[dict] = []

    def _base(chunk_type: str, aspect: str | None, text: str) -> dict:
        return {
            "chunk_id": make_chunk_id(category, model, chunk_type, aspect),
            "model": model,
            "category": category,
            "chunk_type": chunk_type,
            "aspect": aspect,
            "text": text,
            **meta,
        }

    summary = distilled.get("summary") or ""
    chunks.append(_base("summary", None, f"{model}（{cat_zh}）的論壇整體摘要：{summary}"))

    for aspect, comment in (distilled.get("aspects") or {}).items():
        chunks.append(_base(
            "aspect", aspect,
            f"{model}（{cat_zh}）在「{aspect}」方面，論壇評論：{comment}",
        ))

    pros = distilled.get("pros") or []
    cons = distilled.get("cons") or []
    if pros or cons:
        parts = []
        if pros:
            parts.append("正面評價包括：" + "、".join(pros))
        if cons:
            parts.append("負面評價包括：" + "、".join(cons))
        chunks.append(_base("pros_cons", None, f"{model}（{cat_zh}）的" + "，".join(parts) + "。"))

    comparisons = distilled.get("comparisons") or []
    if comparisons:
        chunks.append(_base(
            "comparison", None,
            f"{model}（{cat_zh}）在論壇中常被拿來與{'、'.join(comparisons)}比較。",
        ))

    return chunks


def build_prompt(model: str, category: str, sampled: list[dict], stats: dict) -> str:
    lines = [f"零件類別：{category}", f"型號：{model}", ""]
    lines.append(format_distribution_for_prompt(stats))
    lines.append("")
    lines.append(f"以下是分層抽樣後的 {len(sampled)} 則樣本評論（格式：[來源|標注] 評論內容）：")
    lines.append("")

    for c in sampled:
        src = "巴哈" if c["source"] == "bahamut" else "PTT"
        label = normalize_label(c.get("label"))
        content = c["content"].strip().replace("\n", " ")
        lines.append(f"[{src}|{label}] {content}")

    return "\n".join(lines)


def call_llm(client: OpenAI, prompt: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"    [警告] JSON 解析失敗（第 {attempt+1} 次）: {e}")
            if attempt < retries - 1:
                time.sleep(2)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"    [Rate limit] 等待 {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [錯誤] {e}")
                if attempt < retries - 1:
                    time.sleep(5)
    return None


def load_done_models_v2(path: Path) -> set[str]:
    """讀取 distilled_models_v2.jsonl 裡已完成的型號集合，斷點續跑用。
    一個型號的 distilled 紀錄跟它所有的 chunk 在同一輪迴圈內產生並立即 flush，
    所以只要 distilled 紀錄存在，就代表該型號的 chunk 也已完整寫入，不會有半個型號的殘留。"""
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["model"])
                except Exception:
                    pass
    return done


def process_model(
    client: OpenAI, model: str, category: str, comments: list[dict]
) -> tuple[dict, list[dict]] | None:
    """
    處理單一型號：過濾＋分層抽樣 → 依信心決定是否呼叫 LLM → 組成 distilled_record 與拆分後的 chunks。
    main() 與 test_single() 共用這個函式，避免兩邊邏輯各寫一份。
    回傳 None 代表 LLM 呼叫失敗，呼叫端應跳過這個型號、不寫入任何檔案。
    """
    valid           = _filter_valid_comments(comments)
    sampled, strata = stratified_sample(valid)
    stats           = build_distribution_stats(comments, valid, sampled, strata)

    valid_source_counts: dict[str, int] = defaultdict(int)
    for c in valid:
        valid_source_counts[c["source"]] += 1
    confidence, confidence_reasons = classify_confidence(len(valid), valid_source_counts)

    if confidence == "insufficient":
        # 評論太少，不呼叫 LLM，摘要留空並用固定文字說明
        distilled_fields = {
            "pros": [], "cons": [], "aspects": {}, "comparisons": [],
            "summary": "評論數量過少，無法產生可靠摘要。",
        }
    else:
        prompt = build_prompt(model, category, sampled, stats)
        distilled_fields = call_llm(client, prompt)
        if distilled_fields is None:
            return None

    distilled_record = {
        "model": model,
        "category": category,
        "pros": distilled_fields.get("pros", []),
        "cons": distilled_fields.get("cons", []),
        "aspects": distilled_fields.get("aspects", {}),
        "comparisons": distilled_fields.get("comparisons", []),
        "summary": distilled_fields.get("summary", ""),
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "low_confidence": confidence == "insufficient",  # 相容欄位，主邏輯請看 confidence
        "statistics": stats,
    }

    date_start, date_end = compute_date_range(sampled)
    meta = {
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "total_valid_comments": stats["total_valid_comments"],
        "sampled_comment_count": stats["sampled_comment_count"],
        "source_distribution": stats["source_distribution"],
        "label_distribution": stats["label_distribution"],
        "strata_distribution": stats["strata_distribution"],
        "evidence_ids": compute_evidence_ids(sampled),
        "data_start_date": date_start,
        "data_end_date": date_end,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "distillation_version": DISTILLATION_VERSION,
    }
    chunks = build_rag_chunks(model, category, distilled_record, meta)

    return distilled_record, chunks


def main(model_limit: int | None = None):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("請設定環境變數 OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    random.seed(42)

    print("載入資料...")
    baha = load_baha()
    ptt  = load_ptt()
    all_records = baha + ptt
    print(f"  巴哈: {len(baha)} 則 / PTT: {len(ptt)} 則 / 合計: {len(all_records)} 則")

    groups = group_by_model(all_records)
    all_models = sorted(groups.keys())
    print(f"  共 {len(all_models)} 個型號")

    done_models = load_done_models_v2(DISTILLED_MODELS_V2)
    if done_models:
        print(f"  已完成 {len(done_models)} 個（斷點續跑）")

    todo = [m for m in all_models if m not in done_models]
    if model_limit is not None:
        todo = todo[:model_limit]
    print(f"\n開始蒸餾（待處理 {len(todo)} 個型號）...\n")

    RAG_CHUNKS_V2.parent.mkdir(parents=True, exist_ok=True)

    with open(DISTILLED_MODELS_V2, "a", encoding="utf-8") as distilled_out, \
         open(RAG_CHUNKS_V2, "a", encoding="utf-8") as chunks_out:
        for idx, model in enumerate(todo, 1):
            info     = groups[model]
            category = info["category"]
            comments = info["comments"]

            print(f"[{idx}/{len(todo)}] {model} ({category}) | 原始 {len(comments)} 則...")

            result = process_model(client, model, category, comments)
            if result is None:
                print(f"    [跳過] LLM 呼叫失敗")
                continue
            distilled_record, chunks = result

            stats = distilled_record["statistics"]
            print(f"    有效 {stats['total_valid_comments']} 則 → 抽樣 {stats['sampled_comment_count']} 則 | "
                  f"信心：{distilled_record['confidence']} | 產生 {len(chunks)} 個 chunk")

            distilled_out.write(json.dumps(distilled_record, ensure_ascii=False) + "\n")
            distilled_out.flush()
            for chunk in chunks:
                chunks_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            chunks_out.flush()

            # 避免 rate limit：每處理 5 個型號稍微停頓
            if idx % 5 == 0:
                time.sleep(1)

    print(f"\n完成！輸出：{DISTILLED_MODELS_V2}、{RAG_CHUNKS_V2}")
    total = load_done_models_v2(DISTILLED_MODELS_V2)
    print(f"共 {len(total)} 個型號")


def test_single(model: str = "RTX4070"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("請在 .env 設定 OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    random.seed(42)

    baha = load_baha()
    ptt  = load_ptt()
    groups = group_by_model(baha + ptt)

    if model not in groups:
        print(f"找不到型號 {model}，可用型號：{sorted(groups.keys())[:10]} ...")
        sys.exit(1)

    info   = groups[model]
    result = process_model(client, model, info["category"], info["comments"])
    if result is None:
        print("LLM 呼叫失敗")
        return

    distilled_record, chunks = result
    print(json.dumps(distilled_record, ensure_ascii=False, indent=2))

    print(f"\n拆分後共 {len(chunks)} 個 chunk：")
    for c in chunks:
        preview = c["text"][:60] + ("…" if len(c["text"]) > 60 else "")
        print(f"  [{c['chunk_id']}] {preview}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        model = sys.argv[2] if len(sys.argv) > 2 else "RTX4070"
        test_single(model)
    elif len(sys.argv) > 1 and sys.argv[1] == "--limit-models":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        main(model_limit=limit)
    else:
        main()
