"""
FastAPI backend — wraps the GA recommendation system.
Run from this directory:  uvicorn api:app --reload --port 8000
"""
import os
import sys
from pathlib import Path

# Make bare `from config import ...` style imports work
sys.path.insert(0, str(Path(__file__).parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import DB_PATH, MATCHED_FILES
from data.catalog import PartCatalog
from data.sentiment import SentimentScorer
from advisor.compatibility import CompatibilityChecker
from advisor.upgrade_advisor import UpgradeAdvisor
from core.ga_engine import GARecommender

_HERE = Path(__file__).parent


# ── startup: load heavy resources once ───────────────────────────────────────

class _Resources:
    catalog: PartCatalog
    scorer: SentimentScorer
    checker: CompatibilityChecker
    advisor: UpgradeAdvisor

res = _Resources()


@asynccontextmanager
async def lifespan(app: FastAPI):
    res.catalog = PartCatalog(DB_PATH)
    res.scorer   = SentimentScorer(MATCHED_FILES, DB_PATH)
    res.checker  = CompatibilityChecker()
    res.advisor  = UpgradeAdvisor(res.catalog, res.scorer, res.checker)
    yield


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="GA PC Recommender", lifespan=lifespan)

# 正式前端網域固定放行；沒設 CORS_ORIGIN_REGEX 時行為跟以前完全一樣。
# 本機開發要讓瀏覽器打得到這支 API，啟動前設：
#   export CORS_ORIGIN_REGEX='http://(localhost|127\.0\.0\.1):[0-9]+'
# 用 regex 而不是列舉 port，是因為 Next.js dev 遇到 3000 被佔用會自動換號。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pc-recommender.vercel.app"],
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX") or None,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 單次 /api/recommend 要跑 90,000 次 fitness 評估（約 8 秒），限制每 IP 頻率避免運算成本被打爆
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_HERE / "static" / "index.html")


# ── schema ───────────────────────────────────────────────────────────────────

VALID_USAGES = {"工作", "遊戲", "一般文書"}

class UsageWeights(BaseModel):
    """效能/口碑/CP值三個滑桿，各 0~100，代表「相對重要程度」，不是絕對佔比。
    三者之間的比例會被重新分配進該 usage 預設的效能+口碑+CP值總份額裡，
    budget/compat 這兩個約束完全不受滑桿影響。全部留預設值(50/50/50)等同
    交給系統判斷（維持該 usage 原本的權重比例）。"""
    perf: float = Field(50, ge=0, le=100, description="效能相對重要程度")
    sent: float = Field(50, ge=0, le=100, description="口碑相對重要程度")
    cp:   float = Field(50, ge=0, le=100, description="CP值相對重要程度")


class RecommendRequest(BaseModel):
    budget: int = Field(..., ge=10000, description="Total budget in NT$")
    cooling_prefer: str = Field("auto", description="auto | 風冷 | 水冷")
    usage: str = Field("工作", description="工作 | 遊戲 | 一般文書")
    weights: Optional[UsageWeights] = Field(
        None, description="效能/口碑/CP值滑桿，留空則使用該用途的預設權重"
    )


# ── endpoint ─────────────────────────────────────────────────────────────────

@app.post("/api/recommend")
@limiter.limit("5/minute")
def recommend(request: Request, req: RecommendRequest):
    if req.usage not in VALID_USAGES:
        raise HTTPException(status_code=400, detail=f"不支援的用途：{req.usage}")
    ga = GARecommender(
        catalog=res.catalog,
        scorer=res.scorer,
        checker=res.checker,
        usage=req.usage,
        budget=req.budget,
        pop_size=300,
        generations=300,
        elite_k=2,
        crossover_rate=0.8,
        mutation_rate=0.30,
        cooling_prefer=req.cooling_prefer,
        psu_tier="standard",
        custom_weights=req.weights.dict() if req.weights else None,
    )
    # 實際套用到 fitness 的權重（滑桿換算後的結果），給前端／本機測試頁顯示用，
    # 方便肉眼確認滑桿真的有正確換算成 fitness 權重，不是只是介面上動一動。
    resolved_weights = {
        "perf": round(ga.weights["w_perf"], 4),
        "sent": round(ga.weights["w_sent"], 4),
        "cp": round(ga.weights["w_cp"], 4),
        "budget": round(ga.weights["w_budget"], 4),
        "compat": round(ga.weights["w_compat"], 4),
    }

    top5 = ga.run(verbose=False)
    if not top5:
        raise HTTPException(status_code=500, detail="GA 未能產生配置結果")

    best = top5[0]
    remaining = req.budget - best.total_price

    # Serialize parts (only what the frontend needs)
    parts = [
        {
            "category": cat,
            "name": part.name,
            "price": part.price,
            "score": round(float(res.scorer.get(cat, part.short_name)), 3),
        }
        for cat, part in sorted(best.parts.items())
    ]

    # Compatibility
    penalty, issues = res.checker.check(best)

    # Upgrade recommendations，依加價幅度分級
    def _serialize(r: dict) -> dict:
        return {
            "priority": r["priority"],
            "category": r["category"],
            "current_name": r["current"].name,
            "current_price": r["current"].price,
            "upgrade_name": r["upgrade"].name,
            "upgrade_price": r["upgrade"].price,
            "cost": r["cost"],
            # 只有 CPU/GPU 有 PassMark 跑分，其餘類別是 null，前端不顯示百分比
            "benchmark_gain_pct": r["benchmark_gain_pct"],
            "sentiment_delta": r["sentiment_delta"],
            "spec_changes": r["spec_changes"],
        }

    upgrade_tiers = [
        {
            "extra_budget": t["extra_budget"],
            "extra_ratio": t["extra_ratio"],
            "available": t["available"],
            "spent": t["spent"],
            "new_total_price": t["new_total_price"],
            "upgrades": [_serialize(r) for r in t["upgrades"]],
        }
        for t in res.advisor.recommend_tiers(best, req.budget, remaining, req.usage)
    ]

    return {
        "parts": parts,
        "total_price": best.total_price,
        "budget": req.budget,
        "remaining": remaining,
        "compatibility": {
            "ok": len(issues) == 0,
            "penalty": round(penalty, 3),
            "issues": issues,
        },
        "upgrade_tiers": upgrade_tiers,
        "resolved_weights": resolved_weights,
    }


# ── dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
