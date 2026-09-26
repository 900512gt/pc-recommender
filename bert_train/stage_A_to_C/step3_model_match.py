"""
Step 3：型號比對（使用 GA PTT 型號表）
- 輸入：bahamut_labeled.jsonl
- 輸出：
  bahamut_matched.jsonl   → 比對到型號的
  bahamut_unmatched.jsonl → 未比對到的（之後送 GPT 補抽）

執行：python step3_model_match.py
"""

import json, re
import pandas as pd
from collections import defaultdict, Counter

INPUT_FILE     = "bahamut_labeled.jsonl"
MATCHED_FILE   = "bahamut_matched.jsonl"
UNMATCHED_FILE = "bahamut_unmatched.jsonl"
MODEL_FILE     = "ga_ptt_models.xlsx"

# 工作表對應類別
SHEET_TO_CAT = {
    'CPU': 'CPU', 'GPU': 'GPU', 'RAM': '記憶體', 'MB': '主機板',
    'SSD': 'SSD', 'HDD': 'HDD', 'AIR_COOLER': '風冷',
    'WATER_COOLER': '水冷', 'CASE': '機殼', 'PSU': '電源'
}

# 暱稱/縮寫對照表
ALIAS_MAP = {
    # CPU
    "7800X3D": ("CPU", "AMD R7 7800X3D"),
    "7900X3D": ("CPU", "AMD R9 7900X3D"),
    "9800X3D": ("CPU", "AMD R9 9800X3D"),
    "7500F":   ("CPU", "AMD R5 7500F"),
    "7600X":   ("CPU", "AMD R5 7600X"),
    "14400F":  ("CPU", "Intel i5-14400F"),
    "14400":   ("CPU", "Intel i5-14400"),
    "14700":   ("CPU", "Intel i7-14700"),
    "245K":    ("CPU", "Intel Core Ultra 5 245K"),
    "265K":    ("CPU", "Intel Core Ultra 7 265K"),
    "285K":    ("CPU", "Intel Core Ultra 9 285K"),
    # GPU
    "4090":    ("GPU", "RTX4090"),
    "4080":    ("GPU", "RTX4080"),
    "4070Ti":  ("GPU", "RTX4070Ti"),
    "4070S":   ("GPU", "RTX4070"),
    "4070":    ("GPU", "RTX4070"),
    "4060Ti":  ("GPU", "RTX4060Ti"),
    "4060":    ("GPU", "RTX4060"),
    "3090":    ("GPU", "RTX3090"),
    "3080":    ("GPU", "RTX3080"),
    "3070":    ("GPU", "RTX3070"),
    "3060Ti":  ("GPU", "RTX3060Ti"),
    "3060":    ("GPU", "RTX3060"),
    "5090":    ("GPU", "RTX5090"),
    "5080":    ("GPU", "RTX5080"),
    "5070Ti":  ("GPU", "RTX5070Ti"),
    "5070":    ("GPU", "RTX5070"),
    "5060Ti":  ("GPU", "RTX5060Ti"),
    "5060":    ("GPU", "RTX5060"),
    "7900XTX": ("GPU", "RX7900XTX"),
    "7900GRE": ("GPU", "RX9070GRE"),
    "9070XT":  ("GPU", "RX9070XT"),
    "9070":    ("GPU", "RX9070"),
    "大鵰":    ("GPU", "RX9070GRE"),
    # 記憶體
    "DDR4":    ("記憶體", "DDR4-3200"),
    "DDR5":    ("記憶體", "DDR5-6000"),
    "6000C30": ("記憶體", "DDR5-6000"),
    # 主機板
    "B650":    ("主機板", "B650M"),
    "B760":    ("主機板", "B760M"),
    "Z790":    ("主機板", "Z790"),
    "Z890":    ("主機板", "Z890"),
    "X870":    ("主機板", "X870"),
    "B850":    ("主機板", "B850M"),
    "B860":    ("主機板", "B860M"),
    "H610":    ("主機板", "H610M"),
    # SSD
    "T500":    ("SSD", "Micron T500"),
    "T700":    ("SSD", "Micron T700"),
    "T705":    ("SSD", "Micron T705"),
    "990 PRO": ("SSD", "Samsung 990 PRO"),
    "870 EVO": ("SSD", "Samsung 870 EVO"),
    "SN850":   ("SSD", "WD SN850X"),
    "KC3000":  ("SSD", "Kingston KC3000"),
    "GM7000":  ("SSD", "Acer GM7000"),
    # 風冷
    "NH-D15":  ("風冷", "Noctua NH-D15S"),
    "PA120":   ("風冷", "利民 Peerless Assassin 120"),
    "AXP90":   ("風冷", "利民 AXP90"),
    "AXP120":  ("風冷", "利民 AXP120"),
    "Hyper 212":("風冷", "酷碼 Hyper 212"),
    "虎徹":    ("風冷", "Scythe 虎徹3"),
    "無限6":   ("風冷", "Scythe 無限6"),
    # 機殼
    "NR200":   ("機殼", "酷碼 NR200"),
    "O11":     ("機殼", "聯力 O11 Dynamic EVO"),
    "H5":      ("機殼", "NZXT H5 Flow"),
    "H6":      ("機殼", "NZXT H6 Flow"),
    "H7":      ("機殼", "NZXT H7 Flow"),
    # 電源
    "LEADEX VII": ("電源", "振華 LEADEX VII"),
    "LEADEX III": ("電源", "振華 LEADEX III"),
    "FOCUS GX":   ("電源", "海韻 FOCUS GX"),
    "金鋼彈":     ("電源", "全漢 金鋼彈"),
    "聖武士":     ("電源", "全漢 聖武士"),
}


def load_models(model_file):
    """載入型號表，回傳 [(model_str, category)] 依長度降序"""
    xl = pd.ExcelFile(model_file)
    models = []
    for sheet in xl.sheet_names:
        cat = SHEET_TO_CAT.get(sheet, sheet)
        df  = pd.read_excel(xl, sheet_name=sheet, header=None)
        for _, row in df.iterrows():
            val = str(row.iloc[1]).strip() if len(row) > 1 else ""
            if val and val not in ("nan", "PTT Model"):
                models.append((val, cat))
    # 長的優先比對
    models.sort(key=lambda x: -len(x[0]))
    return models


def match_content(content, models):
    """回傳命中的型號列表 [{category, model, matched_keyword}]"""
    content = str(content)
    matched = []
    seen    = set()

    # 1. 先比對完整型號表
    for model_str, cat in models:
        if model_str in content:
            key = f"{cat}_{model_str}"
            if key not in seen:
                seen.add(key)
                matched.append({
                    "category":        cat,
                    "model":           model_str,
                    "matched_keyword": model_str,
                })

    # 2. 再比對別名/縮寫
    for alias, (cat, model_str) in ALIAS_MAP.items():
        if alias in content:
            key = f"{cat}_{model_str}"
            if key not in seen:
                seen.add(key)
                matched.append({
                    "category":        cat,
                    "model":           model_str,
                    "matched_keyword": alias,
                })

    return matched


def main():
    print("=" * 55)
    print("Step 3：型號比對（GA PTT 型號表）")
    print("=" * 55)

    models = load_models(MODEL_FILE)
    print(f"型號表載入：{len(models)} 個型號")

    # 讀入資料
    print(f"\n讀入 {INPUT_FILE}...")
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    print(f"總筆數：{len(records)}")

    # 修正 label 簡繁混用
    for r in records:
        if r.get("label") in ["负评", "负評"]:
            r["label"] = "負評"
        if r.get("label") is None and r.get("relevant"):
            r["label"] = "中立"

    # 比對
    print("\n比對中...")
    matched_records   = []
    unmatched_records = []

    for r in records:
        content = r.get("content", "")
        hits    = match_content(content, models)

        if hits:
            for h in hits:
                new_r = dict(r)
                new_r["category"]        = h["category"]
                new_r["model"]           = h["model"]
                new_r["matched_keyword"] = h["matched_keyword"]
                matched_records.append(new_r)
        else:
            unmatched_records.append(r)

    # 統計
    unique_matched = len(set(r["_index"] for r in matched_records))
    print(f"\n=== 比對結果 ===")
    print(f"原始資料：{len(records)} 筆")
    print(f"成功比對：{unique_matched} 筆 ({unique_matched/len(records)*100:.1f}%)")
    print(f"展開後：  {len(matched_records)} 筆（一筆可對多個型號）")
    print(f"未比對：  {len(unmatched_records)} 筆 ({len(unmatched_records)/len(records)*100:.1f}%) → 送 GPT")

    print(f"\n類別分布（matched）：")
    cat_cnt = Counter(r["category"] for r in matched_records)
    for cat, cnt in sorted(cat_cnt.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    print(f"\nlabel 分布（matched）：")
    label_cnt = Counter(r.get("label") for r in matched_records)
    print(dict(label_cnt))

    # 儲存
    print(f"\n儲存...")
    with open(MATCHED_FILE, "w", encoding="utf-8") as f:
        for r in matched_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {MATCHED_FILE}：{len(matched_records)} 筆")

    with open(UNMATCHED_FILE, "w", encoding="utf-8") as f:
        for r in unmatched_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {UNMATCHED_FILE}：{len(unmatched_records)} 筆")

    # 範例
    print(f"\n=== 比對成功範例 ===")
    shown = defaultdict(int)
    for r in matched_records:
        cat = r["category"]
        if shown[cat] < 2:
            shown[cat] += 1
            print(f"\n【{cat}】model:{r['model']} 關鍵字:{r['matched_keyword']} label:{r.get('label')}")
            print(f"  {r['content'][:80]}")

    print(f"\n=== 未比對範例（5筆）===")
    for r in unmatched_records[:5]:
        print(f"  label:{r.get('label')} | {r['content'][:75]}")


if __name__ == "__main__":
    main()
