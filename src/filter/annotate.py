"""
annotate.py
使用大模型篩選與硬體型號相關的評論

使用方式：
  python annotate.py --limit 100            # 跑前 100 則 + 含 reason（測試用）
  python annotate.py                        # 跑全部 + 含 reason
  python annotate.py --no-reason            # 跑全部，不含 reason（省 token）
  python annotate.py --start 1000           # 從第 1000 則開始（斷點續跑）
  python annotate.py --no-reason --start 0  # 正式跑全部的建議指令

輸出：
  annotated.jsonl   → 所有評論的判斷結果
  relevant.jsonl    → 只保留相關的評論（可直接拿去 fine-tune）

依賴：
  pip install openai python-dotenv
  在同資料夾建立 .env 檔案，寫入：OPENAI_API_KEY=sk-你的金鑰
"""

import json
import os
import sys
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# 讀取 .env 檔案裡的金鑰
load_dotenv()

# ── 設定 ──────────────────────────────────────────────

API_KEY    = os.environ.get("OPENAI_API_KEY", "")
MODEL      = "gpt-5.4-nano"  # 分類任務最便宜，$0.20/1M tokens
SLEEP      = 0.5             # 每次呼叫間隔（避免超過 rate limit）
MAX_RETRY  = 3               # 失敗重試次數

INPUT_FILE    = "output/filtered_comments.jsonl"
OUTPUT_FILE   = "output/annotated.jsonl"
RELEVANT_FILE = "output/relevant.jsonl"

# ── Prompt 設計 ────────────────────────────────────────

SYSTEM_PROMPT_BASE = """你是一個硬體評論分析專家，專門分析台灣 PTT 的電腦硬體討論。

【任務】判斷一則推文是否對指定硬體型號有明確評價，並判斷評價傾向。

【相關性判斷】
符合以下任一條件才算「相關」：
  1. 推文本身直接評價這個型號的效能、穩定性、溫度、噪音、CP值、使用體驗
  2. 推文明確說推薦或不推薦購買這個型號
  3. 推文描述這個型號的具體問題或優點（如縮缸、過熱、跑分高）

以下情況判定為「不相關」：
  - 內容是閒聊、討論其他主題、版規爭論
  - 只是在文章討論串回覆，但本身沒有對型號表達看法
  - 詢問購買管道、價格、哪裡買
  - 句子明顯截斷或不完整
  - 評論針對的是其他型號，不是指定型號
  - 評論批評的是「非本體的隨附配件」（如附贈散熱器、包裝內容物），而非硬體型號本身

【重要提醒】
文章標題有提到某型號，不代表底下每則推文都在評價這個型號。
每則推文必須「自身」對型號有明確看法，才算相關。

【台灣網路用語判斷原則】
評論可能使用隱晦或反諷語氣表達負評，
判斷時應理解語意而非逐字分析。
若評論的整體語意是「此型號在某方面明顯不如競品」
或「此型號不值得購買」，應判為負評，而非中立。

【好壞評判斷】（只在 relevant=true 時填寫）
- 正評：明確推薦、說效能好/穩定/值得買、使用滿意
- 負評：明確不推薦、說有問題/效能差/後悔/災情（如縮缸、過熱、掉速）
- 中立：有評價但正負不明確，或評論同時有優缺點

只需回傳 JSON 格式，不要有其他文字。

"""

SYSTEM_PROMPT_WITH_REASON = SYSTEM_PROMPT_BASE + """

回傳格式（含原因）：
{"relevant": true 或 false, "label": "正評/負評/中立/null", "reason": "一句話說明原因"}

範例：
{"relevant": true, "label": "負評", "reason": "明確表達縮缸嚴重不推薦"}
{"relevant": true, "label": "正評", "reason": "表達效能強、值得購買"}
{"relevant": true, "label": "中立", "reason": "討論使用情境但未明確表態好壞"}
{"relevant": false, "label": null, "reason": "在討論版規與閒聊，與型號評價無關"}"""

SYSTEM_PROMPT_NO_REASON = SYSTEM_PROMPT_BASE + """

回傳格式：
{"relevant": true 或 false, "label": "正評/負評/中立/null"}

範例：
{"relevant": true, "label": "負評"}
{"relevant": true, "label": "正評"}
{"relevant": true, "label": "中立"}
{"relevant": false, "label": null}"""


def make_prompt(row):
    """根據評論資料組出 user prompt"""
    return f"""硬體型號：{row.get('model', '未知')}
文章標題：{row.get('title', '未知')}
評論內容：{row.get('content', '')}

這則評論是否在評價「{row.get('model', '')}」這個硬體？"""


# ── API 呼叫 ───────────────────────────────────────────

def call_api(client, prompt, with_reason=True):
    """
    呼叫 OpenAI API，回傳解析後的 dict
    失敗時重試，超過次數回傳 None
    with_reason: True 時回傳含 reason 的結果（測試用），False 時省略（正式跑）
    """
    system = SYSTEM_PROMPT_WITH_REASON if with_reason else SYSTEM_PROMPT_NO_REASON
    for attempt in range(MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=100,
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return result

        except json.JSONDecodeError:
            print(f"    [警告] JSON 解析失敗，重試 {attempt+1}/{MAX_RETRY}")
            time.sleep(1)
        except Exception as e:
            print(f"    [錯誤] API 呼叫失敗：{e}，重試 {attempt+1}/{MAX_RETRY}")
            time.sleep(2)

    return None  # 重試全部失敗


# ── 斷點續跑：讀取已完成的進度 ────────────────────────

def load_done_indices(output_path):
    """讀取已標注的行數，用於斷點續跑"""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "_index" in row:
                        done.add(row["_index"])
                except:
                    pass
    return done


# ── 主程式 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",     type=int, default=None,  help="只處理前 N 則（測試用）")
    parser.add_argument("--start",     type=int, default=0,     help="從第 N 則開始")
    parser.add_argument("--no-reason", action="store_true",     help="不輸出 reason（省 token，正式跑用）")
    args = parser.parse_args()

    if not API_KEY:
        print("[錯誤] 請設定 OPENAI_API_KEY 環境變數")
        print("       export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if not os.path.exists(INPUT_FILE):
        print(f"[錯誤] 找不到 {INPUT_FILE}")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY)

    # 讀取所有評論
    with open(INPUT_FILE, encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f]

    # 套用 start / limit
    rows = all_rows[args.start:]
    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    mode = "含 reason（測試模式）" if not args.no_reason else "不含 reason（正式模式）"
    print(f"總計 {total} 則評論待處理（從第 {args.start} 則開始）")
    print(f"使用模型：{MODEL}  |  模式：{mode}")
    print()

    # 讀取已完成的進度（斷點續跑）
    done_indices = load_done_indices(OUTPUT_FILE)
    if done_indices:
        print(f"找到已完成的進度：{len(done_indices)} 則，跳過已完成的部分")

    # 開啟輸出檔案（append 模式，支援斷點續跑）
    out_f = open(OUTPUT_FILE,   "a", encoding="utf-8")
    rel_f = open(RELEVANT_FILE, "a", encoding="utf-8")

    relevant_count = 0
    error_count    = 0

    for i, row in enumerate(rows):
        global_idx = args.start + i

        # 跳過已完成的
        if global_idx in done_indices:
            continue

        prompt = make_prompt(row)

        # 呼叫 API
        result = call_api(client, prompt, with_reason=not args.no_reason)

        if result is None:
            error_count += 1
            result = {"relevant": False, "reason": "API 呼叫失敗"}

        # 合併原始資料和標注結果
        annotated = {
            **row,
            "_index":   global_idx,
            "relevant": result.get("relevant", False),
            "label":    result.get("label", None),
        }
        if not args.no_reason:
            annotated["reason"] = result.get("reason", "")

        # 寫入 annotated.jsonl（所有結果）
        out_f.write(json.dumps(annotated, ensure_ascii=False) + "\n")
        out_f.flush()

        # 相關的才寫入 relevant.jsonl
        if annotated["relevant"]:
            relevant_count += 1
            rel_f.write(json.dumps(annotated, ensure_ascii=False) + "\n")
            rel_f.flush()

        # 進度顯示
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"  [{i+1:6,}/{total:,}] 相關 {relevant_count} 則  錯誤 {error_count} 則")

        time.sleep(SLEEP)

    out_f.close()
    rel_f.close()

    print()
    print(f"✅ 完成")
    print(f"   annotated.jsonl：{total} 則（含所有判斷結果）")
    print(f"   relevant.jsonl：{relevant_count} 則（相關評論）")
    print(f"   錯誤/失敗：{error_count} 則")


if __name__ == "__main__":
    main()