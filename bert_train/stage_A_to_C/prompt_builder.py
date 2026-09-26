# prompt_builder.py — 阶段A：粗粒度面向标注 prompt
#
# 与 absa_v2 的差别：
#   absa_v2  : 输出 0~1 分数（路线二，即时用）
#   阶段A     : 输出 正面/负面/未提及（路线三，给 BERT 学）
#
# 粗粒度标注的好处：
#   1. BERT 擅长分类，正/负/未提及正中强项
#   2. 标注一致性高，训练资料品质好
#   3. 分数由「比例聚合」产生，有统计依据

from aspects import CORE_ASPECTS


def build_labeling_prompt(category: str, model: str, contents: list[str]) -> str:
    """
    建立批次标注 prompt（一次标多则，省 API 成本）。
    输出：每则评论对各面向标 正面/负面/未提及
    """
    core = CORE_ASPECTS.get(category, ["效能", "CP值"])
    core_str = "、".join(core)

    reviews_block = "\n".join(
        f"[{i+1}] {c}" for i, c in enumerate(contents)
    )

    prompt = f"""你是專業的電腦硬體評論標註員。以下是關於「{model}」（類別：{category}）的多則評論。

請針對每則評論，判斷它對下列各「面向」的態度：

核心面向：{core_str}

標註規則（每個面向只能填三選一）：
- "正面"：評論明確表達對這個面向的正面評價
- "負面"：評論明確表達對這個面向的負面評價
- "未提及"：評論完全沒有談到這個面向

額外判斷：
- is_about_product：這則評論的主體是否真的是 {model}？（若在講別的零件填 false）
- is_substantive：是否為有實質評價的評論？（純詢問、未來式「等之後買」填 false）
- extra_aspects：若提到核心面向以外的重要面向（如 VRAM、包裝、客服），列出並標正面/負面

評論列表：
{reviews_block}

只回傳 JSON 陣列，每則格式如下（不要任何其他文字）：
{{
  "id": 評論編號,
  "is_about_product": true/false,
  "is_substantive": true/false,
  "aspects": {{
    "{core[0]}": "正面/負面/未提及",
    ...（每個核心面向都要有）
  }},
  "extra_aspects": {{
    "面向名": "正面/負面"
  }}
}}

重要：
- 沒提到的面向一律填 "未提及"，不要勉強判斷
- 主體不是 {model} 就標 is_about_product=false
- 寧可標「未提及」也不要亂猜"""

    return prompt
