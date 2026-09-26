# check_quality.py — 阶段A：检查 GPT 标注品质
#
# 验证方法：
#   把 GPT 的「面向标注」聚合成整体情感，跟原本的「正/负/中立」比对
#   一致率高 → GPT 标注可信 → 可以全跑
#   一致率低 → 要调 prompt
#
# 用法：python check_quality.py absa_labeled.jsonl

import sys
import json
from collections import Counter


def aspect_to_overall(aspects: dict, extra: dict) -> str:
    """
    把面向标注聚合成整体情感（正/负/中立），用来跟原始 label 比对。
    逻辑：数正面面向 vs 负面面向，多者为整体倾向。
    """
    all_labels = list(aspects.values()) + list(extra.values())
    pos = sum(1 for v in all_labels if v == "正面")
    neg = sum(1 for v in all_labels if v == "負面")

    if pos > neg:
        return "正評"
    elif neg > pos:
        return "負評"
    else:
        return "中立"


def main(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"载入 {len(rows)} 则标注\n")

    # ── 1. 整体情感一致率（GPT聚合 vs 原始label）──
    match = 0
    total_comparable = 0
    confusion = Counter()

    for r in rows:
        orig = r.get("orig_label", "")
        if orig not in ("正評", "負評", "中立"):
            continue
        gpt_overall = aspect_to_overall(r.get("aspects", {}), r.get("extra_aspects", {}))
        total_comparable += 1
        if gpt_overall == orig:
            match += 1
        confusion[(orig, gpt_overall)] += 1

    if total_comparable > 0:
        agreement = match / total_comparable * 100
        print(f"=== 整体情感一致率 ===")
        print(f"  GPT聚合 vs 原始标注：{agreement:.1f}% ({match}/{total_comparable})")
        print()
        print(f"  混淆矩阵（原始 → GPT）：")
        for orig in ["正評", "負評", "中立"]:
            for gpt in ["正評", "負評", "中立"]:
                cnt = confusion.get((orig, gpt), 0)
                mark = "✓" if orig == gpt else " "
                if cnt > 0:
                    print(f"    {mark} {orig} → {gpt}: {cnt}")
        print()

    # ── 2. 面向覆盖率（每个面向被标注的比例）──
    print(f"=== 面向覆盖情况 ===")
    aspect_stats = {}
    for r in rows:
        for aspect, label in r.get("aspects", {}).items():
            if aspect not in aspect_stats:
                aspect_stats[aspect] = Counter()
            aspect_stats[aspect][label] += 1

    for aspect, counter in aspect_stats.items():
        total = sum(counter.values())
        mentioned = counter["正面"] + counter["負面"]
        coverage = mentioned / total * 100 if total > 0 else 0
        print(f"  {aspect:8s} 提及率{coverage:5.1f}%  "
              f"(正{counter['正面']:3d} 负{counter['負面']:3d} 未提及{counter['未提及']:3d})")
    print()

    # ── 3. 动态面向发现 ──
    print(f"=== 动态发现的额外面向 ===")
    extra_counter = Counter()
    for r in rows:
        for aspect in r.get("extra_aspects", {}):
            extra_counter[aspect] += 1
    if extra_counter:
        for aspect, cnt in extra_counter.most_common(10):
            print(f"  {aspect}: {cnt} 次")
    else:
        print(f"  （无）")
    print()

    # ── 4. 雜訊统计 ──
    not_about = sum(1 for r in rows if not r.get("is_about_product", True))
    not_subst = sum(1 for r in rows if not r.get("is_substantive", True))
    print(f"=== 雜訊统计（不删除，仅参考）===")
    print(f"  主体非本产品：{not_about} 则")
    print(f"  非实质评价：{not_subst} 则")
    print()

    # ── 判断 ──
    print(f"=== 品质判断 ===")
    if total_comparable > 0:
        if agreement >= 80:
            print(f"  ✓ 一致率 {agreement:.0f}% ≥ 80%，GPT 标注可信，可以全跑")
        elif agreement >= 70:
            print(f"  △ 一致率 {agreement:.0f}%，尚可，建议微调 prompt 后全跑")
        else:
            print(f"  ✗ 一致率 {agreement:.0f}% < 70%，需要调整 prompt")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "absa_labeled.jsonl"
    main(path)
