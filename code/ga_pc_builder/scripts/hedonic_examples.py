#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
報告用的實際例子：每個方法選擇與對比各拉幾個具體零件／配單出來。
輸出 results/hedonic_examples.json，執行：PYTHONHASHSEED=0 python scripts/hedonic_examples.py
"""
import io, sys, json, random, contextlib, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from config import DB_PATH, MATCHED_FILES  # noqa: E402
from data.catalog import PartCatalog  # noqa: E402
from data.sentiment import SentimentScorer  # noqa: E402
from advisor.compatibility import CompatibilityChecker  # noqa: E402
from core.ga_engine import GARecommender  # noqa: E402
import data.hedonic as H  # noqa: E402

out = {}
with contextlib.redirect_stdout(io.StringIO()):
    cat = PartCatalog(DB_PATH)
    H.attach_spec_values(cat, verbose=False)
    H.attach_efficiency(cat, "vrs")


def short(n, k=34):
    return n[:k]


# 1. 規格→效能驗證：GPU 同晶片、跨晶片；CPU 排名
def validation():
    res = {}
    parts = [p for p in cat.get("GPU") if p.price > 0]
    Q, _ = H.fit_hedonic(parts, H.VALIDATION_MODELS["GPU"])
    by = collections.defaultdict(list)
    for q, p in zip(Q, parts):
        by[p.specs["benchmark_chip"]].append((p, float(q)))
    same = {}
    for chip in ["RTX 5070 12G", "RTX 5070 Ti 16G"]:
        L = sorted(by[chip], key=lambda t: t[0].price)
        same[chip] = {"benchmark": float(L[0][0].specs["benchmark"]), "cards": len(L),
                      "examples": [{"name": short(p.name, 40), "price": p.price, "Q": round(q)}
                                   for p, q in (L[0], L[len(L) // 2], L[-1])],
                      "price_spread": round(L[-1][0].price / L[0][0].price - 1, 3),
                      "Q_spread": round(max(q for _, q in L) / min(q for _, q in L) - 1, 3)}
    res["gpu_same_chip"] = same
    chips = []
    for chip, L in by.items():
        chips.append({"chip": chip, "benchmark": float(L[0][0].specs["benchmark"]),
                      "median_price": float(np.median([p.price for p, _ in L])),
                      "median_Q": round(float(np.median([q for _, q in L])))})
    res["gpu_chips"] = sorted(chips, key=lambda r: r["benchmark"])
    parts = [p for p in cat.get("CPU") if p.price > 0]
    Q, _ = H.fit_hedonic(parts, H.VALIDATION_MODELS["CPU"])
    from scipy.stats import rankdata
    b = np.array([float(p.specs["benchmark"]) for p in parts])
    rb, rq = rankdata(-b, method="min"), rankdata(-Q, method="min")
    pick = ["9950X3D", "285K", "265K【", "9800X3D", "245KF", "7500F", "i5-14400F", "3400G"]
    res["cpu_rank"] = [{"name": short(p.name, 26), "benchmark": b[i], "rank_bench": int(rb[i]),
                        "Q": round(float(Q[i])), "rank_Q": int(rq[i]), "price": p.price}
                       for i, p in enumerate(parts) if any(k in p.name for k in pick)]
    res["cpu_rank"].sort(key=lambda r: r["rank_bench"])
    return res


# 2. 品牌／尺寸控制：同規格不同品牌的電源
def brand_control():
    rows = []
    for p in cat.get("電源"):
        if any(k in p.name for k in ["LEADEX III 850W 黑色", "ROG LOKI 850W 雙8", "FOCUS GX-850", "LEADEX III 750W 黑色", "ROG LOKI 750W"]):
            rows.append({"name": short(p.name, 40), "price": p.price, "Q": round(p.specs["spec_value"]),
                         "theta": round(p.specs["efficiency"], 2)})
    seen, uniq = set(), []
    for r in rows:
        if r["name"] not in seen:
            seen.add(r["name"]); uniq.append(r)
    return uniq


# 3. 函數形式與分組 CV：同一類別不同形式、隨機切分 vs 分組切分
def cv_examples():
    res = {}
    for c in ["主機板", "電源", "機殼", "風冷"]:
        parts = [p for p in cat.get(c) if p.price > 0]
        _, rep = H.fit_hedonic(parts, H.SPEC_MODELS[c])
        dz = H._Design(parts, H.SPEC_MODELS[c])
        X, _, _ = dz.matrix(rep["lambda_x"])
        yt = H.boxcox(dz.y, rep["lambda_y"])
        rng = np.random.default_rng(0)
        folds = rng.integers(0, 10, len(yt))
        pred = np.empty(len(yt))
        for f in range(10):
            tr, te = folds != f, folds == f
            if te.any():
                pred[te] = X[te] @ np.linalg.lstsq(X[tr], yt[tr], rcond=None)[0]
        r_rand = H._r2(np.log(dz.y), np.log(H.inv_boxcox(pred, rep["lambda_y"])))
        res[c] = {"in_sample_r2": rep["r2_lnprice"], "random_cv_r2": round(float(r_rand), 3),
                  "group_cv_r2": rep["r2_groupcv_lnprice"], "by_form": rep["groupcv_by_form"],
                  "lr_tests": rep["lr_tests"], "form": rep["form_chosen"]}
    return res


# 4. 機殼：分組 CV 下預測錯得最多的幾款（為什麼不採用）
def case_worst():
    parts = [p for p in cat.get("機殼") if p.price > 0]
    _, rep = H.fit_hedonic(parts, H.SPEC_MODELS["機殼"])
    dz = H._Design(parts, H.SPEC_MODELS["機殼"])
    X, _, _ = dz.matrix(rep["lambda_x"])
    yt = H.boxcox(dz.y, rep["lambda_y"])
    g = np.unique(dz.group); rng = np.random.default_rng(0); rng.shuffle(g)
    fo = {x: i % 10 for i, x in enumerate(g)}; f = np.array([fo[x] for x in dz.group])
    pred = np.empty(len(yt))
    for k in range(10):
        pred[f == k] = X[f == k] @ np.linalg.lstsq(X[f != k], yt[f != k], rcond=None)[0]
    pp = H.inv_boxcox(pred, rep["lambda_y"])
    err = np.abs(np.log(pp) - np.log(dz.y))
    o = np.argsort(-err)
    rows, seen = [], set()
    for i in o:
        n = short(dz.names[i], 30)
        if n in seen:
            continue
        seen.add(n)
        rows.append({"name": n, "price": float(dz.y[i]), "predicted_out_of_sample": round(float(pp[i]))})
        if len(rows) == 4:
            break
    return rows


# 5. 風冷、機殼的效率
def cooler_case():
    co = {}
    for p in cat.get("風冷"):
        for k in ["Peerless Assassin 120 V3 黑", "ASSASSIN IV 阿薩辛4 黑", "NH-D15S /6", "Hyper 212 Spectrum V3", "ASSASSIN IV VC VISION"]:
            if k in p.name and k not in co:
                from data.cooler_capacity import get_cooler_capacity
                co[k] = {"name": short(p.name, 36), "price": p.price, "capacity_w": get_cooler_capacity(p.name, p.specs),
                         "theta": round(p.specs["efficiency"], 2)}
    ca = {}
    for p in cat.get("機殼"):
        for k in ["Montech X1 黑", "NZXT H5 Flow(2024) 黑", "Fractal Design Meshify 3", "HYTE Y70"]:
            if k in p.name and k not in ca:
                ca[k] = {"name": short(p.name, 36), "price": p.price, "theta": round(p.specs["efficiency"], 2)}
    return {"cooler": list(co.values()), "case": list(ca.values())}


# 6. VRS / CRS / FDH：GPU 每顆晶片最便宜的卡
def frontiers():
    parts = [p for p in cat.get("GPU") if p.price > 0]
    x = np.array([p.price for p in parts], float)
    y = np.array([float(p.specs["benchmark"]) for p in parts])
    vrs = H.efficiency_scores(x, y, "vrs"); fdh = H.efficiency_scores(x, y, "fdh")
    crs = (y / x) / (y / x).max()
    best = {}
    for i, p in enumerate(parts):
        k = p.specs["benchmark_chip"]
        if k not in best or p.price < parts[best[k]].price:
            best[k] = i
    rows = [{"chip": k, "price": parts[i].price, "benchmark": y[i], "vrs": round(float(vrs[i]), 2),
             "crs": round(float(crs[i]), 2), "fdh": round(float(fdh[i]), 2)} for k, i in best.items()]
    rows.sort(key=lambda r: r["benchmark"])
    return [r for r in rows if r["benchmark"] > 10000]


# 7. DEA vs residual：同一個零件，兩種「划不划算」說法不同
def dea_vs_residual():
    rows = []
    for c in ["主機板", "電源", "SSD"]:
        for p in cat.get(c):
            if p.specs.get("spec_value") and p.specs.get("efficiency") is not None:
                rows.append((c, p, p.specs["spec_value"] / p.price, p.specs["efficiency"]))
    # 價格約等於規格行情（residual 比值 ≈ 1，看起來「合理」）但 DEA 效率低
    a = sorted([r for r in rows if 0.97 <= r[2] <= 1.03], key=lambda r: r[3])[:3]
    # residual 比值 < 1（看起來偏貴）但 DEA 效率 = 1（同品質沒有更便宜的）
    b = sorted([r for r in rows if r[3] > 0.999], key=lambda r: r[2])[:2]
    f = lambda r: {"category": r[0], "name": short(r[1].name, 36), "price": r[1].price,
                   "Q": round(r[1].specs["spec_value"]), "residual_ratio": round(r[2], 2), "theta": round(r[3], 2)}
    return {"fair_by_residual_but_inefficient": [f(r) for r in a],
            "pricey_by_residual_but_efficient": [f(r) for r in b]}


# 8. 同三張 8 萬配單，在五種 CP 定義下各得幾分
def cp_on_builds():
    with contextlib.redirect_stdout(io.StringIO()):
        sc = SentimentScorer(MATCHED_FILES, DB_PATH); ck = CompatibilityChecker()
    builds = {}
    for label, basis, cp in [("原本（價格 + ratio）", "price", "ratio"),
                             ("hedonic + noUnder", "hedonic", "noUnder"),
                             ("hedonic + DEA", "hedonic", "dea")]:
        random.seed(0); np.random.seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            ga = GARecommender(catalog=cat, scorer=sc, checker=ck, usage="遊戲", budget=80000,
                               pop_size=300, generations=300, perf_basis=basis, cp_mode=cp)
            builds[label] = ga.run(verbose=False)[0]
    res = []
    for label, b in builds.items():
        row = {"build": label, "total": b.total_price,
               "parts": {c: f"{short(p.name, 22)} {p.price}" for c, p in b.parts.items()}}
        for cp in ["ratio", "noUnder", "residual", "dea"]:
            with contextlib.redirect_stdout(io.StringIO()):
                g = GARecommender(catalog=cat, scorer=sc, checker=ck, usage="遊戲", budget=80000,
                                  perf_basis="hedonic", cp_mode=cp)
            row[cp] = round(g._cp_score(b), 3)
        res.append(row)
    return res


out["validation"] = validation()
out["brand_control"] = brand_control()
out["cv"] = cv_examples()
out["case_worst"] = case_worst()
out["cooler_case"] = cooler_case()
out["frontiers_gpu"] = frontiers()
out["dea_vs_residual"] = dea_vs_residual()
out["cp_on_builds"] = cp_on_builds()
(HERE / "results").mkdir(exist_ok=True)
json.dump(out, open(HERE / "results" / "hedonic_examples.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
