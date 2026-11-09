"""
Hedonic 規格價值：給沒有跑分的零件（主機板、SSD、電源、機殼、風冷、記憶體）一個「規格值多少錢」的分數，
取代原本直接拿價格當效能。

做法（Rosen 1974 的 hedonic price model）：
  ln(價格) = 功能規格 + 非功能溢價（品牌、白色、RGB、玻璃側板）+ 誤差
  規格價值 = exp(只用功能規格的預測值)，非功能變數固定在 0（沒有溢價的基準）

所以兩塊規格一樣的板子，一塊因為品牌或白色版貴 2,000 元，它們的規格價值相同；
GA 的效能分數不會再因為「比較貴」而變高。

用法：attach_spec_values(catalog) 會在每個零件的 specs 寫入 "spec_value"（元）。
只用目錄本身的價格與規格估計，不需要外部資料；目錄更新後重新載入就會重估。
"""
from __future__ import annotations

import re
import numpy as np

# 目錄內部類別名稱 → 規格擷取函式
_SPEC_FUNCS = {}


def _num(x, d=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def _maker(name: str) -> str:
    return name.split()[0] if name else ""


def _white(name: str) -> int:
    return int(bool(re.search(r"白|White|WHITE|雪", name)))


def _rgb(name: str) -> int:
    return int(bool(re.search(r"ARGB|RGB|燈", name)))


def _reg(cat):
    def deco(fn):
        _SPEC_FUNCS[cat] = fn
        return fn
    return deco


@_reg("主機板")
def _mb(p):
    s, n = p.specs, p.name
    ph = re.search(r"(\d+)(?:\+(\d+))*相", n)
    phases = sum(int(x) for x in re.findall(r"\d+", ph.group(0))) if ph else np.nan
    lan = 10 if re.search(r"10G", n, re.I) else 5 if re.search(r"5G", n) else 2.5 if "2.5G" in n else 1
    f = {"chip_" + str(s.get("series")): 1, "ff_" + str(s.get("form_factor")): 1,
         "ddr5": int(s.get("memory_type") == "DDR5"),
         "m2": _num(s.get("m2_slots"), 2), "wifi": int(s.get("wifi") == "有"),
         "wifi7": int(bool(re.search(r"Wi-?Fi ?7|WIFI7|無線7", n, re.I))),
         "lnlan": np.log(lan), "phases": phases / 10}
    return f, {"mk_" + _maker(n): 1, "white": _white(n)}


@_reg("SSD")
def _ssd(p):
    s, n = p.specs, p.name
    f = {"lncap": np.log(_num(s.get("capacity_gb"), 1000)), "if_" + str(s.get("interface")): 1,
         "lnread": np.log(_num(s.get("read_mbs"), 500)),
         "m2230": int("2230" in str(s.get("form_factor"))),
         "qlc": int("QLC" in n), "dram": int("DRAM" in n and "無DRAM" not in n)}
    return f, {"mk_" + _maker(n): 1}


@_reg("電源")
def _psu(p):
    s, n = p.specs, p.name
    f = {"lnwatt": np.log(_num(s.get("wattage"), 750)), "rt_" + str(s.get("rating")): 1,
         "fullmod": int(s.get("modular") == "全模"),
         "atx31": int(s.get("atx_version") == "ATX3.1"),
         "pcie5": int(s.get("pcie5") == "是"), "ff_" + str(s.get("form_factor")): 1,
         "warranty": _num(s.get("warranty_yr"), 10)}
    return f, {"mk_" + _maker(n): 1, "white": _white(n)}


@_reg("機殼")
def _case(p):
    s, n = p.specs, p.name
    f = {"sz_" + str(s.get("series")): 1, "gpu_mm": _num(s.get("max_gpu_mm"), 350) / 100,
         "cooler_mm": _num(s.get("max_cooler_mm"), 165) / 100,
         "water_mm": _num(s.get("max_water_mm"), 360) / 100}
    return f, {"mk_" + _maker(n): 1, "white": _white(n),
               "glass": int(s.get("tempered_glass") == "是"), "rgb": _rgb(n)}


@_reg("風冷")
def _air(p):
    s, n = p.specs, p.name
    f = {"tdp": _num(s.get("max_tdp"), 180) / 100, "pipes": _num(s.get("heat_pipes"), 5),
         "ty_" + str(s.get("cooler_type")): 1}
    return f, {"mk_" + _maker(n): 1, "white": _white(n), "rgb": _rgb(n)}


@_reg("記憶體")
def _ram(p):
    s, n = p.specs, p.name
    f = {"lncap": np.log(_num(s.get("capacity_gb"), 16)), "ddr5": int(s.get("type") == "DDR5"),
         "speed": _num(s.get("speed"), 3200) / 1000, "dual": int("雙通" in str(s.get("kit"))),
         "cl": _num(s.get("cl"))}
    return f, {"mk_" + _maker(n): 1, "white": _white(n), "rgb": _rgb(n)}


def _fit_category(parts, fn, alpha: float = 1.0):
    """回傳每個零件的規格價值（元）與模型摘要"""
    feats = [fn(p) for p in parts]
    fcols = sorted({k for f, _ in feats for k in f})
    ncols = sorted({k for _, nf in feats for k in nf})
    cols = [("F", c) for c in fcols] + [("N", c) for c in ncols]
    X = np.array([[(f if kind == "F" else nf).get(c, 0.0) for kind, c in cols]
                  for f, nf in feats], dtype=float)
    # 缺值：dummy 補 0，數值補中位數
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = 0.0 if np.isnan(col).all() else np.nanmedian(col)
    keep = [j for j in range(X.shape[1]) if np.unique(X[:, j]).size > 1]
    X, cols = X[:, keep], [cols[j] for j in keep]
    y = np.log(np.array([p.price for p in parts], dtype=float))

    # Ridge（標準化後），alpha 小，只是為了 dummy 很多時解得穩
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    beta = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    pred = y.mean() + Xs @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    # 非功能欄固定在 0 → 只剩功能規格的貢獻
    Xf = X.copy()
    for j, (kind, _) in enumerate(cols):
        if kind == "N":
            Xf[:, j] = 0.0
    spec_ln = y.mean() + ((Xf - mu) / sd) @ beta
    return np.exp(spec_ln), {"n": len(parts), "r2": round(float(r2), 3), "features": len(cols)}


def attach_spec_values(catalog, min_parts: int = 15, verbose: bool = True) -> dict:
    """在 catalog 每個非跑分零件的 specs 寫入 spec_value。回傳各類別的模型摘要。"""
    report = {}
    for cat, fn in _SPEC_FUNCS.items():
        parts = [p for p in catalog.get(cat) if p.price > 0]
        if len(parts) < min_parts:
            continue  # 樣本太少不估，_perf_score 會退回用價格
        values, info = _fit_category(parts, fn)
        for p, v in zip(parts, values):
            p.specs["spec_value"] = float(v)
        report[cat] = info
        if verbose:
            print(f"[hedonic] {cat}: n={info['n']} R²={info['r2']}")
    return report
