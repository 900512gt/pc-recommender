from retriever import Retriever
r = Retriever()
for q in ["RTX5070 值得買嗎", "5070 公車幾點", "今天天氣如何", "中階顯卡推薦"]:
    res = r.retrieve(q)
    print(q, "→", [(c["model"], round(c["score"], 3)) for c in res])