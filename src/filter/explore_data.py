import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================
# 設定
# ============================================================

DATA_PATH = "output/relevant.jsonl"
CSV_PATH  = "output/relevant.csv"
PLOT_PATH = "output/distribution.png"

# 中文字體（Mac / Windows / Linux 各自找）
def get_chinese_font():
    candidates = [
        "PingFang TC", "Microsoft JhengHei", "Noto Sans CJK TC",
        "WenQuanYi Micro Hei", "Heiti TC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

font_name = get_chinese_font()
if font_name:
    plt.rcParams["font.family"] = font_name
else:
    print("⚠️  找不到中文字體，圖表標題可能顯示亂碼")

# ============================================================
# 1. 讀 JSONL → DataFrame → 存 CSV
# ============================================================

records = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")  # utf-8-sig 讓 Excel 開不亂碼
print(f"✅ CSV 已存到 {CSV_PATH}，共 {len(df)} 筆")

# ============================================================
# 2. 基本統計
# ============================================================

print("\n--- Label 分布 ---")
print(df["label"].value_counts())
print("\n--- Label 比例 ---")
print(df["label"].value_counts(normalize=True).round(3))
print("\n--- 內文長度統計 ---")
df["content_len"] = df["content"].str.len()
print(df["content_len"].describe().round(1))

# ============================================================
# 3. 畫圖
# ============================================================

label_order = ["正評", "中立", "負評"]
colors = {"正評": "#4CAF50", "中立": "#90A4AE", "負評": "#EF5350"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PTT CPU 評論資料分布", fontsize=16)

# (1) Label 整體分布
label_counts = df["label"].value_counts().reindex(label_order)
axes[0, 0].bar(label_counts.index, label_counts.values,
               color=[colors[l] for l in label_counts.index])
axes[0, 0].set_title("Label 分布")
for i, (l, v) in enumerate(zip(label_counts.index, label_counts.values)):
    axes[0, 0].text(i, v + 20, f"{v}\n({v/len(df)*100:.1f}%)", ha="center")

# (2) Top 10 CPU 型號各 label 堆疊
top_models = df["model"].value_counts().head(10).index
model_label = (
    df[df["model"].isin(top_models)]
    .groupby(["model", "label"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=label_order, fill_value=0)
)
model_label.plot(kind="barh", stacked=True,
                 color=[colors[l] for l in label_order],
                 ax=axes[0, 1])
axes[0, 1].set_title("Top 10 CPU 型號 Label 分布")

# (3) 推文符號 vs Label
tag_label = (
    df.groupby(["tag", "label"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=label_order, fill_value=0)
)
tag_label.plot(kind="bar", color=[colors[l] for l in label_order], ax=axes[1, 0])
axes[1, 0].set_title("推文符號 vs Label")
axes[1, 0].set_xticklabels(tag_label.index, rotation=0)

# (4) 內文長度分布
for label in label_order:
    axes[1, 1].hist(df[df["label"] == label]["content_len"],
                    bins=20, alpha=0.6, label=label, color=colors[label])
axes[1, 1].set_title("內文長度分布")
axes[1, 1].set_xlabel("字數")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"\n✅ 圖表已存到 {PLOT_PATH}")
plt.show()