"""
annotate_overall.py — 替相關評論補標「總體」面向（教授 2026-10-20 要求）

原本的面向標註（aspects_標註_p3.jsonl）只有各類別的具體面向，像「5070 讚」這種
只稱讚型號、沒講具體面向的評論，所有面向都是未提及（val/test 約 3 成相關評論如此）。
這支程式只對相關評論（orig_relevant=True）多問一個面向「總體」，結果另存一個檔，
不改動原本的 p3 標註；重建訓練資料時再依 _index 合併。

總體的定義（2026-10-21 定案採 B）：對指定型號的整體態度，包含具體面向的好壞
  正面：稱讚、滿意、推薦它、說它某方面好
  負面：批評、不滿、勸人別買、說它有問題或某方面差
  未提及：沒有對指定型號表達好壞（包括內容其實在講別的零件、別的型號、軟體或市場）
  過程：定義 A（只標「讚／垃圾」這種純整體評價）試標只有約一成有評價、且一半錯；
  相關性階段順便標的整則 label 盲測一致率約 58%；定義 B 用 nano 約 64%、用 gpt-5.4-mini 約 69%
  （140 則，與 Claude 的盲判比對，非人工）。選 nano 是為了和其他標註同一個模型、控制成本。

使用方式（在 bert_train/ 底下）：
  python annotate_overall.py --sample 100 --output aspects_總體_試標.jsonl   # 先隨機試標 100 則，人工看過再正式跑
  python annotate_overall.py                      # 全部相關評論（約 27,465 則），可中斷後重跑續標（預設定義 B、nano）
  python annotate_overall.py --workers 4          # 遇到 rate limit 時降低同時請求數

輸出：aspects_總體_p4.jsonl，每行 {"_index", "總體", "evidence"}；
_index 是 aspects_標註_p3.jsonl 的行號（從 0 起算）。
"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")

MODEL = "gpt-5.4-nano"      # 與 p3 面向標註、相關性判斷同一個模型
MAX_RETRY = 3
PROMPT_VERSIONS = {"A": "p4-overall-A-v3", "B": "p4-overall-B-v1"}
LABELS = {"正面", "負面", "未提及"}

PROMPT_A = """你是電腦硬體評論的標註員，評論來自台灣 PTT 與巴哈姆特。

【任務】判斷這則評論對「指定型號」有沒有明確的「總體」評價。

【總體的定義】只看對這個型號明確的整體評價（好不好、滿不滿意），不看具體面向，也不看購買建議：
- 正面：整體稱讚或滿意。例如「讚」「好用」「很滿意」「買了不後悔」「神卡」「用起來很舒服」
- 負面：整體批評或不滿。例如「雷」「垃圾」「爛」「後悔」「很失望」「地雷」
- 未提及：沒有整體評價

【判斷原則】
1. 購買建議不算總體評價，正反都一樣 → 未提及：
   - 「就拿 X」「推 X 這張」「建議選 X」「X 直上」「可以考慮 X」（配單、推薦清單）
   - 「別買 X」「不推 X」「X 就別了」「不如買 Y」
   只有同時說出評價時，才依評價判斷，例如「這張很雷別買」→ 負面、「我自己用很滿意推薦給你」→ 正面。
2. 只評論具體面向而沒有整體結論 → 未提及。具體面向另有標註，包括：
   - 價格、划算、太貴、CP 值高低 → 屬於「CP值」面向
   - 效能強弱、跑分、幀數、溫度、噪音、功耗、穩定、用料等 → 屬於各自的面向
   例如「換到 4080 也沒辦法 2K 穩 60」「功耗翻倍」都只是具體面向，不是總體。
3. 比較句不算總體：「A > B」「B 比 A 差」「整體還是比 B 強太多」→ 對 A、B 都是未提及。
4. 評價的對象必須是指定型號，逐字確認被誇或被罵的是哪個型號：
   - 「14750 很佛給推」→ 對 7800X3D 是未提及（被誇的是 14750）
   - 「P5 比較舊款，比 FOCUS GX 還差一點」→ 對 FOCUS GX 不是負面（被比下去的是 P5）
   - 「沒救，40 卡有 DLSS3，30 卡純靠硬實力」→「沒救」講的是 30 卡，不是 40 卡
5. PTT 推文的評價對象通常要從文章標題判斷，但以下不算對型號的評價 → 未提及：
   - 稱讚文章或作者：「好文」「感謝分享」「推開箱」「推詳細」
   - 回應其他推文、玩笑或幹話：「帶不動別用了賣我」
6. 詢問、規格資訊、新聞轉述 → 未提及。
7. 台灣網路用語常有反諷，依整體語意判斷；不確定時判未提及。

只回傳 JSON：{"總體": "正面/負面/未提及", "evidence": "原文中支持判斷的片段，未提及時為空字串"}"""


# 定義 B（2026-10-21 試）：整體態度，包含具體面向的好壞，只排除沒有在評價指定型號的內容。
# 定義 A 太嚴格，純粹的「讚／垃圾」在資料裡很少，試標只有約一成有評價且一半錯；
# 相關性階段的整則 label 雖是同一個概念，但盲測一致率只有約六成（對象搞錯、離題的也標了正負）。
# 範例全部是自編的，沒有用到盲測的 100 則，避免拿驗證題當範例而高估準確率。
PROMPT_B = """你是電腦硬體評論的標註員，評論來自台灣 PTT 與巴哈姆特。

【任務】判斷這則評論對「指定型號」的整體態度。

【標籤】
- 正面：整體而言對指定型號是好的評價。稱讚、滿意、推薦它、說它某方面好（效能強、很涼、很穩、很划算）
- 負面：整體而言對指定型號是不好的評價。批評、不滿、勸人別買、說它有問題或某方面差（跑不動、很吵、常當機、太貴不值得）
- 未提及：沒有對指定型號表達好壞

【判斷原則】
1. 先確認評論有沒有在講指定型號本身。只在講別的零件、別的型號、軟體或系統問題、主機板或筆電整機、通路或市場 → 未提及。
2. 一則評論提到多個型號時，只看指定型號被怎麼評價，逐字確認被誇或被罵的是誰：
   - 「我換了 4070S，3060 留給弟弟，現在 2K 很順」→ 對 4070S 是正面；對 3060 是未提及
   - 「13600K 很吵，換了 NH-D15 才安靜」→ 對 13600K 是未提及（吵的是原本的散熱器），對 NH-D15 是正面
   - 「6800 都要 1 萬 2 了？不如加一點買 7800XT」→ 對 6800 是負面；對 7800XT 是正面
   - 「被 A770 打爆，3050 的 8G 根本不夠」→ 對 A770 是正面；對 3050 是負面
3. 價格：說這個型號貴得不值、很划算 → 算評價；只抱怨通路、黃牛、漲價、匯率，沒有評價產品本身 → 未提及。
4. 好壞參半時依結論判斷；沒有結論 → 未提及。
5. 提問 → 未提及；但描述自己遇到的問題（「這張常常黑屏」「玩一下就當機」）→ 負面。
6. PTT 推文的對象通常要看文章標題；稱讚文章或作者（好文、感謝分享、推開箱）不算評價型號 → 未提及。
7. 台灣網路用語常有反諷，依整體語意判斷；不確定時判未提及。

只回傳 JSON：{"總體": "正面/負面/未提及", "evidence": "原文中支持判斷的片段，未提及時為空字串"}"""

PROMPTS = {"A": PROMPT_A, "B": PROMPT_B}
QUESTIONS = {"A": "這則評論對「{m}」有沒有明確的總體評價？", "B": "這則評論對「{m}」的整體態度是什麼？"}


def make_prompt(row: dict, aspects: list, definition: str = "A") -> str:
    return (f"指定型號：{row.get('model', '')}（類別：{row.get('category', '')}）\n"
            f"這個類別另外標註的具體面向：{'、'.join(aspects)}\n"
            f"文章標題：{row.get('title') or '（無）'}\n"
            f"評論內容：{row.get('content') or ''}\n\n"
            + QUESTIONS[definition].format(m=row.get("model", "")))


def call_api(client, prompt: str, system: str, model: str = MODEL):
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            if result.get("總體") in LABELS:
                return result
            print(f"    [警告] 標籤不在選項內：{result}，重試 {attempt + 1}/{MAX_RETRY}")
        except json.JSONDecodeError:
            print(f"    [警告] JSON 解析失敗，重試 {attempt + 1}/{MAX_RETRY}")
        except Exception as e:  # rate limit、連線錯誤等，等久一點再試
            print(f"    [錯誤] {e}，重試 {attempt + 1}/{MAX_RETRY}")
            time.sleep(5 * (attempt + 1))
    return None


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["_index"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main(args):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[錯誤] 找不到 OPENAI_API_KEY，請在專案根目錄的 .env 設定")
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    todo = [(i, r) for i, r in enumerate(rows) if r.get("orig_relevant")]
    if args.index_file:
        # 只標指定的 _index（例如盲測表抽到的那 100 則），用來和人工答案比對
        import csv
        wanted = {int(r["_index"]) for r in csv.DictReader(open(args.index_file, encoding="utf-8-sig"))}
        todo = [t for t in todo if t[0] in wanted]
    elif args.sample:
        # 試標用隨機抽樣，PTT、巴哈各半：檔案前段全是巴哈，而兩邊的評論型態差很多
        # （PTT 型號在標題、推文很短；巴哈型號在內文、常是長篇配單）
        import random
        rng = random.Random(args.seed)
        picked = []
        for src in ("ptt", "baha"):
            pool = [t for t in todo if t[1]["source"] == src]
            picked += rng.sample(pool, min(args.sample // 2, len(pool)))
        todo = sorted(picked, key=lambda t: t[0])
    elif args.limit:
        todo = todo[:args.limit]
    out = Path(args.output)
    done = load_done(out)
    todo = [(i, r) for i, r in todo if i not in done]
    print(f"相關評論待標 {len(todo)} 則（已完成 {len(done)} 則），模型 {args.model}，同時 {args.workers} 個請求")

    client = OpenAI()
    lock = threading.Lock()
    counts = {"正面": 0, "負面": 0, "未提及": 0, "失敗": 0}

    def work(item):
        i, r = item
        prompt = make_prompt(r, list(r.get("aspects", {})), args.definition)
        return i, call_api(client, prompt, PROMPTS[args.definition], args.model)

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, item) for item in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            i, result = fut.result()
            with lock:
                if result is None:
                    counts["失敗"] += 1   # 不寫入，下次重跑會再標
                else:
                    counts[result["總體"]] += 1
                    f.write(json.dumps({"_index": i, "總體": result["總體"],
                                        "evidence": result.get("evidence", ""),
                                        "prompt_version": PROMPT_VERSIONS[args.definition]}, ensure_ascii=False) + "\n")
                    f.flush()
                if n % 200 == 0 or n == len(todo):
                    print(f"  [{n:6,}/{len(todo):,}] {counts}", flush=True)
    print(f"完成：{counts}\n輸出：{out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "aspects_標註_p3.jsonl"))
    ap.add_argument("--output", default=str(HERE / "aspects_總體_p4.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="只標前 N 則相關評論")
    ap.add_argument("--sample", type=int, default=None, help="隨機抽 N 則相關評論試標（搭配 --output 另存）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--definition", choices=["A", "B"], default="B", help="A：只標明確整體評價；B：整體態度（含面向好壞）")
    ap.add_argument("--model", default=MODEL, help="預設與其他標註相同的 gpt-5.4-nano")
    ap.add_argument("--index-file", default=None, help="只標這個 CSV 裡 _index 欄列出的評論")
    main(ap.parse_args())
