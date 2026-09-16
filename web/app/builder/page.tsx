"use client";

import { useState, type FormEvent } from "react";
import {
  GaApiError,
  recommend,
  type CoolingPreference,
  type RecommendResponse,
  type Usage,
} from "../lib/ga-api";

const MIN_BUDGET = 10000;

export default function BuilderPage() {
  // 存字串而不是數字：存數字的話清空欄位會讓 Number("") 變成 0，畫面重新渲染出
  // 一個刪不掉的「0」——想刪它又觸發同一次轉換，使用者永遠清不乾淨。
  const [budget, setBudget] = useState("40000");
  const [usage, setUsage] = useState<Usage>("工作");
  const [cooling, setCooling] = useState<CoolingPreference>("auto");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecommendResponse | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const parsedBudget = Number(budget);
    // 空欄位會是 Number("") === 0，一樣被這道檢查擋下來
    if (parsedBudget < MIN_BUDGET) {
      setError(`預算至少需要 NT$${MIN_BUDGET.toLocaleString()}`);
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await recommend({ budget: parsedBudget, usage, cooling_prefer: cooling });
      setResult(res);
    } catch (err) {
      setError(err instanceof GaApiError ? err.message : "發生未知錯誤");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-360 flex-1 px-6 py-16">
      <h1 className="text-4xl font-semibold tracking-tight">配置建置器</h1>
      <p className="mt-2 text-text-muted">
        輸入預算與需求，透過遺傳演算法從零件資料庫中挑出最佳組合。
      </p>

      <form
        onSubmit={onSubmit}
        className="mt-10 grid max-w-xl gap-6 rounded-lg border border-border p-6"
      >
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">預算（NT$）</span>
          <input
            type="number"
            min={MIN_BUDGET}
            step={1000}
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
            className="rounded-sm border border-border-strong px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">用途</span>
          <select
            value={usage}
            onChange={(e) => setUsage(e.target.value as Usage)}
            className="rounded-sm border border-border-strong px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="工作">工作</option>
            <option value="遊戲">遊戲</option>
          </select>
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium">散熱偏好</span>
          <select
            value={cooling}
            onChange={(e) => setCooling(e.target.value as CoolingPreference)}
            className="rounded-sm border border-border-strong px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="auto">自動</option>
            <option value="風冷">風冷</option>
            <option value="水冷">水冷</option>
          </select>
        </label>

        <button
          type="submit"
          disabled={loading}
          className="rounded-sm bg-accent px-4 py-2 text-sm font-medium text-accent-inverse transition-opacity hover:opacity-90 disabled:opacity-50"
        >
          {loading ? "生成中…" : "生成配置"}
        </button>
      </form>

      {error && (
        <div className="mt-6 max-w-xl rounded-sm border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {result && <ResultView result={result} />}
    </main>
  );
}

function ResultView({ result }: { result: RecommendResponse }) {
  return (
    <section className="mt-12 max-w-3xl">
      <div className="grid grid-cols-3 gap-px overflow-hidden rounded-lg border border-border bg-border">
        <Stat label="總價" value={`NT$${result.total_price.toLocaleString()}`} />
        <Stat label="預算" value={`NT$${result.budget.toLocaleString()}`} />
        <Stat label="剩餘" value={`NT$${result.remaining.toLocaleString()}`} />
      </div>

      <div
        className={`mt-6 rounded-sm border px-4 py-3 text-sm ${
          result.compatibility.ok
            ? "border-border text-text-muted"
            : "border-red-300 bg-red-50 text-red-700"
        }`}
      >
        {result.compatibility.ok ? (
          "相容性檢查通過"
        ) : (
          <div>
            <p className="font-medium">相容性檢查發現問題</p>
            <ul className="mt-1 list-disc pl-5">
              {result.compatibility.issues.map((issue, i) => (
                <li key={i}>{issue}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <table className="mt-8 w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-text text-left">
            <th className="py-2 font-medium">類別</th>
            <th className="py-2 font-medium">品項</th>
            <th className="py-2 font-medium text-right">價格</th>
            <th className="py-2 font-medium text-right">口碑分數</th>
          </tr>
        </thead>
        <tbody>
          {result.parts.map((part) => (
            <tr key={part.category} className="border-b border-border">
              <td className="py-2 text-text-muted">{part.category}</td>
              <td className="py-2">{part.name}</td>
              <td className="py-2 text-right">
                NT${part.price.toLocaleString()}
              </td>
              <td className="py-2 text-right">{part.score.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <UpgradeSection result={result} />
    </section>
  );
}

/**
 * 升級建議依加價幅度分級。GA 會把預算用到只剩幾個百分點，光靠剩餘預算幾乎只換得動
 * 電源，所以額外提供「多花 10% / 20% 能換到什麼」讓使用者自己權衡。
 */
function UpgradeSection({ result }: { result: RecommendResponse }) {
  const [selected, setSelected] = useState(0);
  const tiers = result.upgrade_tiers;
  if (tiers.length === 0) return null;

  // 換了預算重新搜尋後級距數量可能變少，夾住索引避免落在不存在的級距上
  const tier = tiers[Math.min(selected, tiers.length - 1)];

  return (
    <div className="mt-10">
      <h2 className="text-lg font-semibold">升級建議</h2>
      <p className="mt-1 text-sm text-text-muted">
        效能提升依 PassMark 跑分計算，沒有跑分資料的類別只列規格差異，不換算成百分比。
      </p>

      <div className="mt-4 flex gap-2">
        {tiers.map((t, i) => (
          <button
            key={t.extra_ratio}
            onClick={() => setSelected(i)}
            className={`rounded-sm border px-3 py-1.5 text-sm transition-colors ${
              t === tier
                ? "border-accent bg-accent text-accent-inverse"
                : "border-border-strong text-text-muted hover:text-text"
            }`}
          >
            {t.extra_ratio === 0 ? "不加價" : `加價 ${(t.extra_ratio * 100).toFixed(0)}%`}
          </button>
        ))}
      </div>

      <p className="mt-3 text-sm text-text-muted">
        多花 NT${tier.spent.toLocaleString()}，總價 NT$
        {result.total_price.toLocaleString()} → NT${tier.new_total_price.toLocaleString()}
        <span className="ml-1 text-text-dim">（這一級的項目可以一起買）</span>
      </p>

      <div className="mt-4 grid gap-4">
        {tier.upgrades.map((u, i) => (
          <div key={i} className="rounded-lg border border-border p-4 text-sm">
            <p className="font-medium">
              {u.category}：{u.current_name} → {u.upgrade_name}
            </p>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-text-muted">
              <span>加價 NT${u.cost.toLocaleString()}</span>
              {u.benchmark_gain_pct !== null && (
                <span className="font-medium text-text">
                  效能 {formatGain(u.benchmark_gain_pct)}
                </span>
              )}
              {Math.abs(u.sentiment_delta) >= 0.01 && (
                <span>
                  論壇口碑 {u.sentiment_delta > 0 ? "+" : ""}
                  {u.sentiment_delta.toFixed(2)}
                </span>
              )}
            </div>
            {u.spec_changes.length > 0 && (
              <ul className="mt-2 list-disc pl-5 text-text-dim">
                {u.spec_changes.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * 跑分差距大的時候百分比會失去可讀性——GT710 換 RTX5080 是 +5667%，
 * 寫成「約 58 倍」才看得懂。倍數門檻設在 100%（也就是兩倍）。
 */
function formatGain(pct: number): string {
  if (pct >= 100) return `約 ${(1 + pct / 100).toFixed(1)} 倍`;
  return `+${pct}%`;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-bg p-4">
      <p className="text-xs text-text-muted">{label}</p>
      <p className="mt-1 text-xl font-semibold">{value}</p>
    </div>
  );
}
