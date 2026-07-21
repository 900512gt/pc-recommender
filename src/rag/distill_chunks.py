"""
RAG chunk distillation: 將每個零件型號的 PTT + 巴哈評論，透過 LLM 蒸餾成結構化摘要。
輸出: data/rag_chunks.jsonl，每行一個型號的 chunk。

使用方式（從專案根目錄執行）：
  python src/rag/distill_chunks.py --test              # 測試單一型號 RTX4070
  python src/rag/distill_chunks.py --test "AMD R7 7800X3D"  # 測試指定型號
  python src/rag/distill_chunks.py                     # 跑全部 161 個型號

依賴：
  pip install openai python-dotenv
  在專案根目錄建立 .env，寫入：OPENAI_API_KEY=sk-你的金鑰
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv



load_dotenv()

# ── 路徑設定 ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
BAHA_FILES = [ROOT / f"src/database/input/baha_comment/matched_part{i}.jsonl" for i in [1, 2, 3]]
PTT_FILE   = ROOT / "src/database/input/ptt_comment/relevant.jsonl"
OUTPUT     = ROOT / "data/rag_chunks.jsonl"

# ── 抽樣參數 ──────────────────────────────────────────────
MIN_CONTENT_LEN   = 15   # 過濾過短的評論
SAMPLES_PER_LABEL = 50   # 每個 label × 每個來源最多取幾則
MIN_COMMENTS      = 5    # 低於此數的型號標記 low_confidence

# ── LLM 設定 ──────────────────────────────────────────────
MODEL      = "gpt-4o-mini"
MAX_TOKENS = 1200

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


def stratified_sample(comments: list[dict]) -> list[dict]:
    """
    依 source × label 分層抽樣，每格最多 SAMPLES_PER_LABEL 則。
    先過濾太短的評論。
    """
    valid = [c for c in comments if len(c.get("content", "")) >= MIN_CONTENT_LEN
             and normalize_label(c.get("label")) in ("正評", "負評", "中立")]

    # 依 source + label 分桶
    buckets: dict[tuple, list] = defaultdict(list)
    for c in valid:
        key = (c["source"], normalize_label(c["label"]))
        buckets[key].append(c)

    sampled = []
    for key, bucket in buckets.items():
        random.shuffle(bucket)
        sampled.extend(bucket[:SAMPLES_PER_LABEL])

    return sampled


def build_prompt(model: str, category: str, sampled: list[dict]) -> str:
    lines = [f"零件類別：{category}", f"型號：{model}", ""]
    lines.append(f"以下共 {len(sampled)} 則評論（格式：[來源|標注] 評論內容）：")
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


def load_done_models(output_path: Path) -> set[str]:
    """讀取已處理的型號，支援斷點續跑。"""
    done = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["model"])
                except Exception:
                    pass
    return done


def main():
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

    done_models = load_done_models(OUTPUT)
    if done_models:
        print(f"  已完成 {len(done_models)} 個（斷點續跑）")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    todo = [m for m in all_models if m not in done_models]
    print(f"\n開始蒸餾（待處理 {len(todo)} 個型號）...\n")

    with open(OUTPUT, "a", encoding="utf-8") as out:
        for idx, model in enumerate(todo, 1):
            info     = groups[model]
            category = info["category"]
            comments = info["comments"]
            sampled  = stratified_sample(comments)

            total_valid = len(sampled)
            low_conf    = total_valid < MIN_COMMENTS

            label_dist = defaultdict(int)
            src_dist   = defaultdict(int)
            for c in comments:
                label_dist[normalize_label(c.get("label"))] += 1
                src_dist[c["source"]] += 1

            print(f"[{idx}/{len(todo)}] {model} ({category}) | "
                  f"原始 {len(comments)} 則 → 抽樣 {total_valid} 則"
                  + (" [low_confidence]" if low_conf else ""))

            chunk = {
                "model":         model,
                "category":      category,
                "comment_count": len(comments),
                "sentiment":     dict(label_dist),
                "sources":       dict(src_dist),
                "low_confidence": low_conf,
            }

            if low_conf:
                # 評論太少，僅保留原始評論，不蒸餾
                chunk["raw_comments"] = [c["content"] for c in comments]
                chunk["pros"]        = []
                chunk["cons"]        = []
                chunk["aspects"]     = {}
                chunk["comparisons"] = []
                chunk["summary"]     = "評論數量過少，無法產生可靠摘要。"
            else:
                prompt = build_prompt(model, category, sampled)
                result = call_llm(client, prompt)
                if result is None:
                    print(f"    [跳過] LLM 呼叫失敗")
                    continue
                chunk.update(result)

            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            out.flush()

            # 避免 rate limit：每處理 5 個型號稍微停頓
            if idx % 5 == 0:
                time.sleep(1)

    print(f"\n完成！輸出：{OUTPUT}")
    total = load_done_models(OUTPUT)
    print(f"共 {len(total)} 個 chunk")


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

    info    = groups[model]
    sampled = stratified_sample(info["comments"])
    prompt  = build_prompt(model, info["category"], sampled)

    print(f"型號：{model}  抽樣：{len(sampled)} 則\n")
    result = call_llm(client, prompt)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("LLM 呼叫失敗")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        model = sys.argv[2] if len(sys.argv) > 2 else "RTX4070"
        test_single(model)
    else:
        main()
