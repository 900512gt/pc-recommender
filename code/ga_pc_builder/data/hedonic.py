"""
Hedonic 品質指數：把零件規格換算成一個「規格價值」分數（元），給沒有跑分的零件當效能用。

方法依據
--------
- Rosen (1974, JPE 82(1)) ：產品是一組特性的組合，hedonic 價格函數 p = f(z) 的斜率是各特性的
  隱含價格（implicit price）。
- Triplett (2004/2006, OECD Handbook on Hedonic Indexes) ：迴歸變數用買方在意、廠商生產的
  「產出特性」；一個型號的品質 = 它的特性組合在隱含價格下的價值；觀察單位是不同的 variety。
- 函數形式：Box & Cox (1964) 轉換，λ 用最大概似估計；Halvorsen & Pollakowski (1981) 以概似比
  檢定在 Box-Cox 家族中選擇函數形式；Cropper, Deck & McConnell (1988) 模擬發現有遺漏變數時
  「linear Box-Cox」（價格與連續特性做 Box-Cox、虛擬變數不轉換）最穩健，故採用此形式。
- 標準誤：同一型號家族的不同版本（顏色、容量）誤差相關，用 cluster-robust 標準誤
  （Cameron & Miller 2015）；驗證用依型號家族分組的交叉驗證，避免同家族同時在訓練與測試。
- 影響點：Cook (1977) 距離 > 4/n 標記，重估後比較品質指數排序，不自動刪除（Triplett 建議
  先檢查而不是直接刪）。
- 品牌：放進迴歸當控制變數以免遺漏變數偏誤，但計算品質指數時固定在樣本平均，
  所以品牌溢價不算品質。Pakes (2003) 指出 hedonic 係數混合了邊際成本與加成，
  不等於消費者的邊際願付價格；因此這裡的「規格價值」解讀為市場上該規格組合的平均價格水準，
  用來排序規格高低，不解讀為效用。

輸出
----
attach_spec_values(catalog) 在每個零件的 specs 寫入 "spec_value"（元，品牌溢價為樣本平均時的
預測價格），並回傳各類別的模型報告（λ、R²、分組 CV、LR 檢定、影響點穩定度等）。
"""
from __future__ import annotations

import re
import numpy as np

# ─────────────────────────────────────────────────────────────
# 變數定義：每類回傳 (連續功能特性, 虛擬功能特性, 控制變數)
#   連續功能特性 → Box-Cox 轉換（需 > 0）
#   虛擬功能特性 → 0/1，不轉換
#   控制變數     → 品牌、白色、RGB 等非功能溢價；算品質指數時固定在樣本平均
# ─────────────────────────────────────────────────────────────


def _num(x, d=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def _maker(name: str) -> str:
    n = re.sub(r"^\[[^\]]*\]\s*", "", name or "")   # 去掉「[搭機優惠]」這類前綴
    return n.split()[0] if n else ""


def _white(name: str) -> int:
    return int(bool(re.search(r"白|White|WHITE|雪", name)))


def _rgb(name: str) -> int:
    return int(bool(re.search(r"ARGB|RGB|燈", name)))


def _mb(p):
    s, n = p.specs, p.name
    ph = re.search(r"(\d+)(?:\+(\d+))*相", n)
    phases = sum(int(x) for x in re.findall(r"\d+", ph.group(0))) if ph else np.nan
    lan = 10 if re.search(r"10G", n, re.I) else 5 if re.search(r"\b5G", n) else 2.5 if "2.5G" in n else 1
    cont = {"m2_slots": _num(s.get("m2_slots")), "lan_gbps": lan, "vrm_phases": phases}
    dum = {"chip_" + str(s.get("series")): 1,
           "ddr5": int(s.get("memory_type") == "DDR5"), "wifi": int(s.get("wifi") == "有"),
           "wifi7": int(bool(re.search(r"Wi-?Fi ?7|WIFI7|無線7", n, re.I)))}
    # 板型（ITX/M-ATX/ATX）放控制變數：小板的溢價是「體積」不是效能，裝不裝得下由相容性檢查處理
    return cont, dum, {"mk_" + _maker(n): 1, "white": _white(n), "ff_" + str(s.get("form_factor")): 1}


def _ssd(p):
    s, n = p.specs, p.name
    cont = {"capacity_gb": _num(s.get("capacity_gb")), "read_mbs": _num(s.get("read_mbs"))}
    dum = {"if_" + str(s.get("interface")): 1, "qlc": int("QLC" in n),
           "dram": int("DRAM" in n and "無DRAM" not in n),
           "m2230": int("2230" in str(s.get("form_factor")))}
    return cont, dum, {"mk_" + _maker(n): 1}


def _psu(p):
    s, n = p.specs, p.name
    # 保固年數幾乎全是 10 年（58/63），與品牌高度共線（VIF 27.8），不放
    cont = {"wattage": _num(s.get("wattage"))}
    dum = {"rt_" + str(s.get("rating")): 1, "full_modular": int(s.get("modular") == "全模"),
           "atx31": int(s.get("atx_version") == "ATX3.1"), "pcie5": int(s.get("pcie5") == "是")}
    # 尺寸（ATX/SFX/SFX-L）放控制變數：SFX 的溢價是「體積」不是效能
    return cont, dum, {"mk_" + _maker(n): 1, "white": _white(n), "ff_" + str(s.get("form_factor")): 1}


def _case(p):
    s, n = p.specs, p.name
    cont = {"max_gpu_mm": _num(s.get("max_gpu_mm")), "max_cooler_mm": _num(s.get("max_cooler_mm")),
            "max_water_mm": _num(s.get("max_water_mm"))}
    dum = {"size_" + str(s.get("series")): 1}
    return cont, dum, {"mk_" + _maker(n): 1, "white": _white(n),
                       "glass": int(s.get("tempered_glass") == "是"), "rgb": _rgb(n)}


def _air(p):
    s, n = p.specs, p.name
    cont = {"rated_tdp": _num(s.get("max_tdp")), "heat_pipes": _num(s.get("heat_pipes"))}
    dum = {"type_" + str(s.get("cooler_type")): 1}
    return cont, dum, {"mk_" + _maker(n): 1, "white": _white(n), "rgb": _rgb(n)}


def _ram(p):
    s, n = p.specs, p.name
    cont = {"capacity_gb": _num(s.get("capacity_gb")), "speed_mts": _num(s.get("speed")),
            "cas_latency": _num(s.get("cl"))}
    dum = {"ddr5": int(s.get("type") == "DDR5"), "dual": int("雙通" in str(s.get("kit")))}
    return cont, dum, {"mk_" + _maker(n): 1, "white": _white(n), "rgb": _rgb(n)}


# CPU / GPU：效能分數直接用跑分；這裡的模型只給 CP（成本效率）用，特性是跑分本身
def _cpu_bench(p):
    s, n = p.specs, p.name
    cont = {"benchmark": _num(s.get("benchmark"))}
    dum = {"igpu": int(s.get("igpu") == "有")}
    return cont, dum, {"box": int("盒裝" in n)}


def _gpu_bench(p):
    s, n = p.specs, p.name
    cont = {"benchmark": _num(s.get("benchmark")), "vram_gb": _num(s.get("vram_gb"))}
    return cont, {}, {"mk_" + _maker(n): 1, "white": _white(n), "rgb": _rgb(n),
                      "oc": int(bool(re.search(r"\bOC\b", n)))}


# 驗證用：CPU / GPU 只用規格（不含跑分），看算出來的品質指數跟 PassMark 有多一致
def _cpu_spec(p):
    s, n = p.specs, p.name
    cont = {"cores": _num(s.get("cores")), "threads": _num(s.get("threads")),
            "boost_ghz": _num(s.get("boost_ghz")), "cache_mb": _num(s.get("cache_mb")),
            "tdp_w": _num(s.get("tdp"))}
    dum = {"igpu": int(s.get("igpu") == "有"), "sock_" + str(s.get("socket")): 1}
    return cont, dum, {"box": int("盒裝" in n)}


def _gpu_spec(p):
    s, n = p.specs, p.name
    cont = {"shaders": _num(s.get("cuda_sp")), "boost_mhz": _num(s.get("boost_mhz")),
            "vram_gb": _num(s.get("vram_gb")), "tgp_w": _num(s.get("tgp_watts"))}
    chip = str(s.get("gpu_chip", ""))
    dum = {"arch_" + str(s.get("series")): 1, "gddr7": int("GDDR7" in chip)}
    return cont, dum, {"mk_" + _maker(n): 1, "white": _white(n), "rgb": _rgb(n),
                       "oc": int(bool(re.search(r"\bOC\b", n)))}


# 樣本外門檻：依型號家族分組交叉驗證，規格至少要解釋一半的 ln 價格變異，品質指數才採用
CV_MIN = 0.5

# 給 GA 用的模型（效能 + CP）
SPEC_MODELS = {"主機板": _mb, "SSD": _ssd, "電源": _psu, "機殼": _case, "風冷": _air,
               "記憶體": _ram, "CPU": _cpu_bench, "GPU": _gpu_bench}
# 驗證用模型
VALIDATION_MODELS = {"CPU": _cpu_spec, "GPU": _gpu_spec}


# ─────────────────────────────────────────────────────────────
# 估計
# ─────────────────────────────────────────────────────────────

def boxcox(x, lam):
    x = np.asarray(x, dtype=float)
    return np.log(x) if abs(lam) < 1e-9 else (x ** lam - 1.0) / lam


def inv_boxcox(y, lam):
    y = np.asarray(y, dtype=float)
    if abs(lam) < 1e-9:
        return np.exp(y)
    return np.maximum(lam * y + 1.0, 1e-12) ** (1.0 / lam)


class _Design:
    """把零件清單轉成設計矩陣（去重、補值、去掉無變異與共線欄位）"""

    def __init__(self, parts, fn):
        rows = [fn(p) for p in parts]
        self.cont = sorted({k for c, _, _ in rows for k in c})
        self.dum = sorted({k for _, d, _ in rows for k in d})
        self.ctrl = sorted({k for _, _, t in rows for k in t})
        C = np.array([[r[0].get(k, np.nan) for k in self.cont] for r in rows], float).reshape(len(rows), -1)
        D = np.array([[r[1].get(k, 0.0) for k in self.dum] for r in rows], float).reshape(len(rows), -1)
        T = np.array([[r[2].get(k, 0.0) for k in self.ctrl] for r in rows], float).reshape(len(rows), -1)
        # 連續變數缺值：補中位數，另加缺值指標（不讓缺值被當成真的中位數）
        self.missing_cols = []
        for j, k in enumerate(self.cont):
            miss = np.isnan(C[:, j]) | (C[:, j] <= 0)
            if miss.any():
                C[miss, j] = np.nanmedian(np.where(miss, np.nan, C[:, j]))
                D = np.c_[D, miss.astype(float)]
                self.dum.append(f"missing_{k}")
                self.missing_cols.append(k)
        self.C, self.D, self.T = C, D, T
        self.y = np.array([p.price for p in parts], float)
        # 交叉驗證與群聚標準誤的「家族」：同品牌、功能規格完全相同的版本（只差顏色、RGB、包裝），
        # 它們其實是同一個產品，不能一個在訓練、一個在測試。
        def fam(r):
            mk = [k for k in r[2] if k.startswith("mk_")]
            return repr((mk, sorted((k, round(v, 3)) for k, v in r[0].items() if np.isfinite(v)),
                         sorted(r[1].items())))
        self.group = np.array([fam(r) for r in rows])
        self.names = [p.name for p in parts]

    def matrix(self, lam_x):
        """回傳 [常數, BoxCox(連續), 虛擬, 控制] 與欄位名稱、哪些欄是控制變數"""
        Cx = boxcox(self.C, lam_x) if self.C.size else self.C
        X = np.c_[np.ones(len(self.y)), Cx, self.D, self.T]
        names = ["const"] + self.cont + self.dum + self.ctrl
        is_ctrl = np.array([False] * (1 + len(self.cont) + len(self.dum)) + [True] * len(self.ctrl))
        # 去掉無變異欄（常數除外）與完全共線欄（QR 判斷）
        keep = [0] + [j for j in range(1, X.shape[1]) if np.ptp(X[:, j]) > 0]
        X, names, is_ctrl = X[:, keep], [names[j] for j in keep], is_ctrl[keep]
        q, r = np.linalg.qr(X)
        indep = np.abs(np.diag(r)) > 1e-8 * np.abs(np.diag(r)).max()
        return X[:, indep], [n for n, k in zip(names, indep) if k], is_ctrl[indep]


def _profile_loglik(dz: _Design, lam_y, lam_x):
    """Box-Cox 集中概似（含 Jacobian）：-n/2·ln(SSR/n) + (λy−1)·Σln y"""
    X, _, _ = dz.matrix(lam_x)
    yt = boxcox(dz.y, lam_y)
    beta, *_ = np.linalg.lstsq(X, yt, rcond=None)
    ssr = float(((yt - X @ beta) ** 2).sum())
    n = len(yt)
    return -n / 2 * np.log(ssr / n) + (lam_y - 1) * np.log(dz.y).sum()


def _fit_ols(X, yt):
    beta, *_ = np.linalg.lstsq(X, yt, rcond=None)
    return beta


def _cluster_se(X, resid, groups):
    """Cluster-robust 共變異數（CR1，Cameron & Miller 2015）"""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    ug = np.unique(groups)
    for g in ug:
        idx = groups == g
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    G = len(ug)
    c = G / max(G - 1, 1) * (n - 1) / max(n - k, 1)
    V = c * XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.clip(np.diag(V), 0, None))


def _quality_index(X, is_ctrl, beta, lam_y):
    """品質指數：控制變數固定在樣本平均，其餘照實際規格，轉回價格尺度（元）"""
    Xq = X.copy()
    Xq[:, is_ctrl] = X[:, is_ctrl].mean(axis=0)
    return inv_boxcox(Xq @ beta, lam_y)


def _r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_hedonic(parts, fn, lam_grid=None, cv_folds=10, seed=0):
    """估計一個類別的 hedonic 模型，回傳 (品質指數 array, 報告 dict)"""
    if lam_grid is None:
        lam_grid = np.round(np.arange(-1.0, 1.01, 0.1), 2)
    dz = _Design(parts, fn)
    n = len(dz.y)

    # 1. 函數形式：(λy, λx) 二維格點最大概似（Halvorsen & Pollakowski 1981 的做法）
    ll = {(ly, lx): _profile_loglik(dz, ly, lx) for ly in lam_grid for lx in lam_grid}
    (ml_y, ml_x), ll_max = max(ll.items(), key=lambda kv: kv[1])
    # 概似比檢定（自由度 2）：常見的三種固定形式相對於最大概似形式是否被拒絕
    def lr(ly, lx):
        stat = 2 * (ll_max - ll[(ly, lx)])
        return round(float(stat), 2), round(float(np.exp(-stat / 2)), 4)   # χ²(2) 存活函數 = e^(−x/2)
    tests = {"log-log (0,0)": lr(0.0, 0.0), "semi-log (0,1)": lr(0.0, 1.0), "linear (1,1)": lr(1.0, 1.0)}

    # 2. 樣本外檢查：依型號家族分組交叉驗證，在 ln 價格尺度比較（不同 λ 才可比）
    groups = np.unique(dz.group)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    k = min(cv_folds, len(groups))
    fold_of = {g: i % k for i, g in enumerate(groups)}
    folds = np.array([fold_of[g] for g in dz.group])

    def group_cv(ly, lx):
        X_, _, _ = dz.matrix(lx)
        yt_ = boxcox(dz.y, ly)
        pred = np.empty(n)
        for f in range(k):
            tr, te = folds != f, folds == f
            pred[te] = X_[te] @ _fit_ols(X_[tr], yt_[tr])
        return float(_r2(np.log(dz.y), np.log(inv_boxcox(pred, ly)))), float(_r2(yt_, pred))

    cv_ml = group_cv(ml_y, ml_x)
    # 最大概似形式若無法外推（分組 CV R² < CV_MIN），改用通過檢查、CV 最好的標準形式；
    # 都不通過時這個類別判定為「規格無法可靠解釋價格」，不輸出品質指數（valid=False）。
    candidates = {"ML": (ml_y, ml_x, cv_ml)}
    for nm, (ly, lx) in {"log-log": (0.0, 0.0), "semi-log": (0.0, 1.0), "linear": (1.0, 1.0)}.items():
        candidates[nm] = (ly, lx, group_cv(ly, lx))
    if cv_ml[0] >= CV_MIN:
        chosen = "ML"
    else:
        ok = {nm: v for nm, v in candidates.items() if v[2][0] >= CV_MIN}
        chosen = max(ok, key=lambda nm: ok[nm][2][0]) if ok else max(candidates, key=lambda nm: candidates[nm][2][0])
    lam_y, lam_x, (r2_cv_p, r2_cv_t) = candidates[chosen]
    valid = r2_cv_p >= CV_MIN

    # 3. 估計（選定形式）
    X, names, is_ctrl = dz.matrix(lam_x)
    yt = boxcox(dz.y, lam_y)
    beta = _fit_ols(X, yt)
    resid = yt - X @ beta
    se = _cluster_se(X, resid, dz.group)
    yhat = inv_boxcox(X @ beta, lam_y)

    # 4. 影響點：Cook's distance
    H_diag = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X)
    p_ = X.shape[1]
    s2 = (resid ** 2).sum() / max(n - p_, 1)
    cook = resid ** 2 / (p_ * s2) * H_diag / np.maximum((1 - H_diag) ** 2, 1e-12)
    flagged = cook > 4 / n
    Q = _quality_index(X, is_ctrl, beta, lam_y)
    if flagged.any() and (~flagged).sum() > p_ + 2:
        b2 = _fit_ols(X[~flagged], yt[~flagged])
        Q2 = _quality_index(X, is_ctrl, b2, lam_y)
        rank_stab = _spearman(Q, Q2)
    else:
        rank_stab = 1.0

    # 5. 多重共線：連續特性的 VIF
    vif = {}
    for j, nm in enumerate(names):
        if nm in dz.cont:
            others = np.delete(X, j, axis=1)
            b = _fit_ols(others, X[:, j])
            r2j = _r2(X[:, j], others @ b)
            vif[nm] = round(1 / max(1 - r2j, 1e-9), 1)

    coef = {nm: (round(float(b_), 4), round(float(s_), 4))
            for nm, b_, s_, c in zip(names, beta, se, is_ctrl) if not c and nm != "const"}
    unique_varieties = len({(tuple(r), y) for r, y in zip(X.round(6).tolist(), dz.y)})
    report = {
        "valid": bool(valid), "form_chosen": chosen,
        "lambda_ml": (float(ml_y), float(ml_x)), "groupcv_ml_lnprice": round(cv_ml[0], 3),
        "groupcv_by_form": {nm: round(v[2][0], 3) for nm, v in candidates.items()},
        "n": n, "varieties": unique_varieties, "families": len(groups), "params": p_,
        "lambda_y": float(lam_y), "lambda_x": float(lam_x), "lr_tests": tests,
        "r2_transformed": round(float(_r2(yt, X @ beta)), 3),
        "r2_lnprice": round(float(_r2(np.log(dz.y), np.log(yhat))), 3),
        "r2_groupcv_transformed": round(float(r2_cv_t), 3),
        "r2_groupcv_lnprice": round(float(r2_cv_p), 3),
        "cook_flagged": int(flagged.sum()), "rank_stability_without_flagged": round(float(rank_stab), 3),
        "vif": vif, "missing_imputed": dz.missing_cols,
        "coef_functional": coef,
    }
    return Q, report


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def attach_spec_values(catalog, min_parts: int = 15, verbose: bool = True) -> dict:
    """在每個零件的 specs 寫入 spec_value（元）。回傳各類別的模型報告。
    CPU/GPU 的 spec_value 由跑分換算，只給 CP 用；效能分數仍直接用跑分。"""
    reports = {}
    for cat, fn in SPEC_MODELS.items():
        parts = [p for p in catalog.get(cat) if p.price > 0]
        if len(parts) < min_parts:
            continue
        Q, rep = fit_hedonic(parts, fn)
        reports[cat] = rep
        for p, q in zip(parts, Q):
            if rep["valid"]:
                p.specs["spec_value"] = float(q)
            else:
                p.specs.pop("spec_value", None)
        if verbose:
            print(f"[hedonic] {cat}: n={rep['n']} 形式={rep['form_chosen']} λy={rep['lambda_y']} "
                  f"λx={rep['lambda_x']} 分組CV R²(ln價格)={rep['r2_groupcv_lnprice']} "
                  f"{'採用' if rep['valid'] else '不採用（規格無法可靠解釋價格）'}")
    return reports


# ─────────────────────────────────────────────────────────────
# 價格—品質效率（CP 值用）
# ─────────────────────────────────────────────────────────────
# Farrell (1957) 投入導向效率：產品 i 的效率 θ_i = 達到「至少同樣品質」所需的最低價格 / 實際價格。
# 參考集合是同類別的所有零件；前緣用 BCC 變動規模報酬（Banker, Charnes & Cooper 1984），
# 一個投入（價格）、一個產出（品質）。消費品的價格—品質效率用 DEA 衡量見
# Kamakura, Ratchford & Agrawal (1988)、Doyle & Green (1991)。
# 不用固定規模報酬（CCR）：單一投入單一產出時 CCR 等於「品質/價格」最大者為唯一標竿，
# 高階零件每單位效能本來就比較貴，CCR 會把所有高階品都判成沒效率（等同鼓勵買低階）。
# 另提供 FDH（Deprins, Simar & Tulkens 1984）：只跟「真實存在、品質不低於它」的零件比，
# 不假設可以混合兩個產品，作為敏感度分析。

def efficiency_scores(prices, quality, frontier: str = "vrs"):
    """回傳每個產品的投入導向效率 θ ∈ (0, 1]"""
    x = np.asarray(prices, float)
    y = np.asarray(quality, float)
    n = len(x)
    theta = np.ones(n)
    if frontier == "fdh":
        for i in range(n):
            theta[i] = x[y >= y[i] - 1e-12].min() / x[i]
        return theta
    # VRS，單投入單產出：前緣是 (品質, 價格) 點雲的下凸包，且價格隨品質不遞減
    # 最低成本 = min Σλj xj  s.t. Σλj yj ≥ yi, Σλj = 1, λ ≥ 0
    from scipy.optimize import linprog
    for i in range(n):
        res = linprog(c=x, A_ub=-y[None, :], b_ub=[-y[i]], A_eq=np.ones((1, n)), b_eq=[1.0],
                      bounds=(0, None), method="highs")
        theta[i] = min(res.fun / x[i], 1.0) if res.success else 1.0
    return theta


def attach_efficiency(catalog, frontier: str = "vrs") -> dict:
    """在每個零件的 specs 寫入 "efficiency"（Farrell 投入導向效率）與 "eff_output"（用了哪個產出）。
    產出（品質）的選擇，優先用直接量測的效能，其次用 hedonic 品質指數：
      CPU / GPU     → PassMark 跑分（實測效能）
      風冷          → 解熱瓦數（data.cooler_capacity，廠商標示 + 已知錯誤校正）。
                      hedonic 驗證不通過，但解熱瓦數本身就是效能規格，DEA 可以直接用原始屬性
                      當產出（Doyle & Green 1991）。
      主機板/SSD/電源/記憶體 → hedonic 品質指數 spec_value（驗證通過）
      其他（機殼）  → 沒有可靠的品質指標：視為品質相同，標竿是同類最低價，θ = 最低價 / 價格。
                      理由：無法證明貴的機殼提供更多價值時，多花的錢就不算換到品質。"""
    from data.cooler_capacity import get_cooler_capacity
    summary = {}
    cats = list(SPEC_MODELS) + ["機殼"]
    for cat in cats:
        parts = [p for p in catalog.get(cat) if p.price > 0]
        if not parts:
            continue
        if cat in ("CPU", "GPU"):
            q, src = [_num(p.specs.get("benchmark"), np.nan) for p in parts], "benchmark"
        elif cat == "風冷":
            q, src = [get_cooler_capacity(p.name, p.specs) for p in parts], "cooler_capacity_w"
        elif any(p.specs.get("spec_value") for p in parts):
            q, src = [p.specs.get("spec_value", np.nan) for p in parts], "hedonic_spec_value"
        else:
            q, src = [1.0] * len(parts), "equal_quality"
        q = np.array(q, float)
        ok = np.isfinite(q) & (q > 0)
        for p in parts:
            p.specs.pop("efficiency", None)
            p.specs.pop("quality", None)
        if ok.sum() < 5:
            continue
        sub = [p for p, k in zip(parts, ok) if k]
        th = efficiency_scores([p.price for p in sub], q[ok], frontier)
        for p, t in zip(sub, th):
            p.specs["efficiency"] = float(t)
            p.specs["eff_output"] = src
            # 給 GA 效能分數用的品質值（perf_basis="hedonic"）；品質相同的類別不寫，效能分數給中性值
            if src in ("hedonic_spec_value", "cooler_capacity_w"):
                p.specs["quality"] = float(q[parts.index(p)])
        summary[cat] = {"output": src, "n": int(ok.sum()), "on_frontier": int((th > 0.999).sum()),
                        "mean_eff": round(float(th.mean()), 3)}
    return summary
