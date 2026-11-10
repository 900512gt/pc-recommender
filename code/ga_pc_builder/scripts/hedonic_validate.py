#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
驗證「hedonic 能把規格換算成效能」：CPU / GPU 有實測跑分，故意不把跑分放進迴歸，
只用規格（核心、執行緒、時脈、快取、TDP；GPU 為著色器數、時脈、VRAM、TGP、架構）
估計品質指數 Q，再看 Q 和 PassMark 跑分有多一致。

輸出：results/hedonic_validation.json
執行（在 code/ga_pc_builder/ 底下）：python scripts/hedonic_validate.py
"""
import io, sys, json, contextlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from config import DB_PATH  # noqa: E402
from data.catalog import PartCatalog  # noqa: E402
import data.hedonic as H  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    cat = PartCatalog(DB_PATH)
out = {}
for c, fn in H.VALIDATION_MODELS.items():
    parts = [p for p in cat.get(c) if p.price > 0]
    Q, rep = H.fit_hedonic(parts, fn)
    b = np.array([float(p.specs["benchmark"]) for p in parts])
    pr = np.array([p.price for p in parts], float)
    chips = {}
    for q, p in zip(Q, parts):
        chips.setdefault(p.specs.get("benchmark_chip"), []).append((q, p.price, float(p.specs["benchmark"])))
    multi = [v for v in chips.values() if len(v) > 2]
    res = {
        "n": len(parts), "chips": len(chips), "form": rep["form_chosen"],
        "lambda": [rep["lambda_y"], rep["lambda_x"]], "groupcv_lnprice": rep["r2_groupcv_lnprice"],
        "spearman_Q_bench": round(float(spearmanr(Q, b)[0]), 3),
        "spearman_price_bench": round(float(spearmanr(pr, b)[0]), 3),
        "pearson_lnQ_lnbench": round(float(pearsonr(np.log(Q), np.log(b))[0]), 3),
        "pearson_lnprice_lnbench": round(float(pearsonr(np.log(pr), np.log(b))[0]), 3),
        "chip_spearman_Q_bench": round(float(spearmanr([np.median([x[0] for x in v]) for v in chips.values()],
                                                       [v[0][2] for v in chips.values()])[0]), 3),
        "chip_spearman_price_bench": round(float(spearmanr([np.median([x[1] for x in v]) for v in chips.values()],
                                                           [v[0][2] for v in chips.values()])[0]), 3),
    }
    if multi:   # 同一顆晶片（跑分相同）的不同卡：Q 的離散度 vs 價格的離散度
        res["within_chip_sd_lnQ"] = round(float(np.mean([np.std(np.log([x[0] for x in v])) for v in multi])), 4)
        res["within_chip_sd_lnprice"] = round(float(np.mean([np.std(np.log([x[1] for x in v])) for v in multi])), 4)
    res["coef_functional"] = rep["coef_functional"]
    out[c] = res
    print(c, {k: v for k, v in res.items() if k != "coef_functional"})
(HERE / "results").mkdir(exist_ok=True)
json.dump(out, open(HERE / "results" / "hedonic_validation.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
