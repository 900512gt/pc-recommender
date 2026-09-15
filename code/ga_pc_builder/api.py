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

VALID_USAGES = {"工作", "遊戲"}

class RecommendRequest(BaseModel):
    budget: int = Field(..., ge=10000, description="Total budget in NT$")
    cooling_prefer: str = Field("auto", description="auto | 風冷 | 水冷")
    usage: str = Field("工作", description="工作 | 遊戲")


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
    )

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
    }


# ── dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
