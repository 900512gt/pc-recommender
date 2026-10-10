"use client";

import { useEffect, useId, useState, type CSSProperties, type FormEvent } from "react";
import Link from "next/link";
import {
  fetchSelectableAspects,
  GaApiError,
  recommend,
  type AspectEvidence,
  type AspectPrefs,
  type CoolingPreference,
  type PreferenceWeights,
  type RecommendedPart,
  type RecommendResponse,
  type SelectableAspects,
  type Usage,
} from "../lib/ga-api";
import styles from "./preference-slider.module.css";

const MIN_BUDGET = 10000;
const INITIAL_USAGE: Usage = "工作";

/**
 * 各用途的系統預設比重，對應 code/ga_pc_builder/config.py 的 USAGE_WEIGHTS
 * （w_perf / w_sent / w_cp）。那邊改權重時這裡要一起改。
 *
 * 目前三項預設等權：系統不預設使用者比較重視哪一項，用途只影響後端「哪些零件的
 * 效能比較重要」。滑桿從這個位置出發，沒調整過就不送 weights。
 */
const USAGE_DEFAULT_WEIGHTS: Record<Usage, PreferenceWeights> = {
  遊戲: { perf: 1, sent: 1, cp: 1 },
  工作: { perf: 1, sent: 1, cp: 1 },
};

const PREFERENCES = [
  { key: "perf", label: "效能" },
  { key: "sent", label: "口碑" },
  { key: "cp", label: "CP 值" },
] as const;

/** 把用途預設比重換算成滑桿位置（0~100，三項加總約 100）。等權時是 33 / 33 / 33。 */
function defaultSliders(usage: Usage): PreferenceWeights {
  const w = USAGE_DEFAULT_WEIGHTS[usage];
  const total = w.perf + w.sent + w.cp;
  return {
    perf: Math.round((w.perf / total) * 100),
    sent: Math.round((w.sent / total) * 100),
    cp: Math.round((w.cp / total) * 100),
  };
}

function isCustomized(usage: Usage, weights: PreferenceWeights): boolean {
  const defaults = defaultSliders(usage);
  return PREFERENCES.some((p) => weights[p.key] !== defaults[p.key]);
}

export default function BuilderPage() {
  // 存字串而不是數字：存數字的話清空欄位會讓 Number("") 變成 0，畫面重新渲染出
  // 一個刪不掉的「0」——想刪它又觸發同一次轉換，使用者永遠清不乾淨。
  const [budget, setBudget] = useState("40000");
  const [usage, setUsage] = useState<Usage>(INITIAL_USAGE);
  const [weights, setWeights] = useState<PreferenceWeights>(() => defaultSliders(INITIAL_USAGE));
  const [cooling, setCooling] = useState<CoolingPreference>("auto");
  const [selectable, setSelectable] = useState<SelectableAspects[]>([]);
  const [aspectPrefs, setAspectPrefs] = useState<AspectPrefs>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecommendResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchSelectableAspects().then((list) => {
      if (!cancelled) setSelectable(list);
    });
    return () => {
      cancelled = true;
    };
  }, []);

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
      const total = weights.perf + weights.sent + weights.cp;
      const res = await recommend({
        budget: parsedBudget,
        usage,
        cooling_prefer: cooling,
        // 沒調整過（或三項全是 0）就不送，讓後端完整沿用該用途的預設比重
        ...(isCustomized(usage, weights) && total > 0 ? { weights } : {}),
        ...(Object.keys(aspectPrefs).length > 0 ? { aspect_prefs: aspectPrefs } : {}),
      });
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

      {/* 寬螢幕左邊放表單、右邊放結果，結果不必捲到表單底下才看得到；窄螢幕照原本上下排 */}
      <div className="mt-10 grid items-start gap-10 lg:grid-cols-[24rem_minmax(0,1fr)]">
        <form
          onSubmit={onSubmit}
          className="grid gap-6 rounded-lg border border-border p-6"
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
              onChange={(e) => {
                const next = e.target.value as Usage;
                setUsage(next);
                // 滑桿跟著跳到新用途的預設位置，不然會把上一個用途的比重帶過來
                setWeights(defaultSliders(next));
              }}
              className="rounded-sm border border-border-strong px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
            >
              <option value="工作">工作</option>
              <option value="遊戲">遊戲</option>
            </select>
          </label>

          <PreferenceSliders usage={usage} value={weights} onChange={setWeights} />

          {selectable.length > 0 && (
            <AspectPicker options={selectable} value={aspectPrefs} onChange={setAspectPrefs} />
          )}

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

        <div className="min-w-0">
          {error && (
            <div className="rounded-sm border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {result && <ResultView result={result} />}

          {!result && !error && (
            <div className="rounded-lg border border-dashed border-border-strong px-6 py-16 text-center text-sm text-text-muted">
              {loading
                ? "正在搜尋最佳組合，大約需要 5~10 秒…"
                : "設定好條件後按「生成配置」，推薦的零件、口碑依據與升級建議會顯示在這裡。"}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

/**
 * 效能／口碑／CP 值的偏好比重。右邊顯示的是三項之間的相對佔比而不是滑桿原始值，
 * 讓使用者看得出「拉到 100」不等於「100%」。
 *
 * 版面沿用表單其他欄位「標籤在上、控制項在下」的排法：每列上方左邊是名稱、右邊是
 * 佔比，下方是整條寬度的滑桿。這樣三條軌道等長、左右兩端對齊，手機上也不必把
 * 名稱、軌道、數字硬擠在同一行。佔比數字沿用結果區 Stat 的「數值用 semibold」。
 */
function PreferenceSliders({
  usage,
  value,
  onChange,
}: {
  usage: Usage;
  value: PreferenceWeights;
  onChange: (next: PreferenceWeights) => void;
}) {
  const id = useId();
  const total = value.perf + value.sent + value.cp;
  const customized = isCustomized(usage, value);

  return (
    <div role="group" aria-labelledby={`${id}-title`} className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between gap-4">
        <span id={`${id}-title`} className="text-sm font-medium">
          偏好比重
        </span>
        {customized ? (
          <button
            type="button"
            onClick={() => onChange(defaultSliders(usage))}
            className="text-xs text-text-muted underline-offset-4 transition-colors hover:text-text hover:underline"
          >
            恢復「{usage}」預設
          </button>
        ) : (
          <span className="text-xs text-text-dim">「{usage}」的系統預設</span>
        )}
      </div>

      {PREFERENCES.map((p) => {
        const share = total > 0 ? Math.round((value[p.key] / total) * 100) : null;
        const inputId = `${id}-${p.key}`;
        return (
          <div key={p.key} className="flex flex-col gap-1">
            <div className="flex items-baseline justify-between gap-4">
              <label htmlFor={inputId} className="text-sm text-text-muted">
                {p.label}
              </label>
              {/* 右對齊＋等寬數字，拖動時數字改變不會讓位置左右跳動 */}
              <output htmlFor={inputId} className="tabular-nums">
                {share === null ? (
                  <span className="text-base font-semibold text-text-dim">—</span>
                ) : (
                  <>
                    <span className="text-base font-semibold">{share}</span>
                    <span className="ml-0.5 text-xs text-text-muted">%</span>
                  </>
                )}
              </output>
            </div>
            <input
              id={inputId}
              type="range"
              min={0}
              max={100}
              step={1}
              value={value[p.key]}
              onChange={(e) => onChange({ ...value, [p.key]: Number(e.target.value) })}
              // 螢幕閱讀器預設會唸滑桿原始值（例如 50），但畫面上顯示的是佔比，改唸佔比
              aria-valuetext={share === null ? undefined : `${share}%`}
              className={styles.range}
              style={{ "--value": value[p.key] / 100 } as CSSProperties}
            />
          </div>
        );
      })}

      <p className="text-xs text-text-dim">
        {total === 0
          ? "三項都是 0 時會使用系統預設比重。"
          : "百分比是三項之間的相對比重。預算與零件相容性的把關不受影響。"}
      </p>
    </div>
  );
}

/**
 * 口碑裡特別在意的面向（選填）。選了的面向佔該零件口碑的一半，其餘面向平均分另一半，
 * 所以是「加重」而不是「只看」。選項只列後端認定評論資料夠的面向。
 *
 * 用可複選的切換鈕而不是下拉選單：每類只有 2~4 個選項，一眼看得完，也不必多點一下。
 * 樣式與升級建議的級距按鈕相同（選取＝黑底），不另外發明新的選取狀態。
 */
function AspectPicker({
  options,
  value,
  onChange,
}: {
  options: SelectableAspects[];
  value: AspectPrefs;
  onChange: (next: AspectPrefs) => void;
}) {
  const id = useId();
  const picked = Object.keys(value).length > 0;

  function toggle(category: string, aspect: string) {
    const current = value[category] ?? [];
    const nextList = current.includes(aspect)
      ? current.filter((a) => a !== aspect)
      : [...current, aspect];
    const next = { ...value };
    // 清空的類別整個拿掉，「沒選任何面向」才能用空物件判斷
    if (nextList.length > 0) next[category] = nextList;
    else delete next[category];
    onChange(next);
  }

  return (
    <div role="group" aria-labelledby={`${id}-title`} className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between gap-4">
        <span id={`${id}-title`} className="text-sm font-medium">
          特別在意的口碑面向
          <span className="ml-2 text-xs font-normal text-text-dim">選填</span>
        </span>
        {picked && (
          <button
            type="button"
            onClick={() => onChange({})}
            className="text-xs text-text-muted underline-offset-4 transition-colors hover:text-text hover:underline"
          >
            全部清除
          </button>
        )}
      </div>

      {options.map((opt) => (
        <div key={opt.category} className="flex flex-col gap-2">
          <span className="text-sm text-text-muted">{opt.category}</span>
          <div className="flex flex-wrap gap-2">
            {opt.aspects.map((aspect) => {
              const on = value[opt.category]?.includes(aspect) ?? false;
              return (
                <button
                  key={aspect}
                  type="button"
                  aria-pressed={on}
                  onClick={() => toggle(opt.category, aspect)}
                  className={`rounded-sm border px-3 py-1.5 text-sm outline-none transition-colors focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 ${
                    on
                      ? "border-accent bg-accent text-accent-inverse"
                      : "border-border-strong text-text-muted hover:text-text"
                  }`}
                >
                  {aspect}
                </button>
              );
            })}
          </div>
        </div>
      ))}

      <p className="text-xs text-text-dim">
        選了的面向會佔該零件口碑的一半，其餘面向仍會列入。只列出論壇評論夠多、分得出型號差異的面向。
      </p>
    </div>
  );
}

function ResultView({ result }: { result: RecommendResponse }) {

  return (
    <section>
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
            <th className="py-2 pl-4 font-medium whitespace-nowrap text-right">口碑分數</th>
          </tr>
        </thead>
        <tbody>
          {result.parts.map((part) => (
            <tr key={part.category} className="border-b border-border">
              <td className="py-2 pr-4 whitespace-nowrap text-text-muted">{part.category}</td>
              <td className="py-2">{part.name}</td>
              <td className="py-2 pl-4 whitespace-nowrap text-right">
                NT${part.price.toLocaleString()}
              </td>
              <td className="py-2 text-right">{part.score.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <SentimentEvidence parts={result.parts} />

      <UpgradeSection result={result} />
    </section>
  );
}

/**
 * 口碑分數的依據：每個零件各面向的名次與正負評價則數，並連到該型號的原始評論。
 * 讓使用者能自己核對「指定了 CP值，選到的這顆 CP值口碑到底好不好」，而不是只能相信系統。
 *
 * 名次照實顯示：指定的面向只佔整體評分的一小部分，選到的零件可能只是中段，
 * 這時候就該顯示中段，不能只挑好看的講。
 */
function SentimentEvidence({ parts }: { parts: RecommendedPart[] }) {
  // 對不到評論的零件只有一句說明，排到最後，不要佔掉第一格
  const withEvidence = parts
    .filter((p) => p.sentiment)
    .sort((a, b) => Number(a.sentiment!.model === null) - Number(b.sentiment!.model === null));
  if (withEvidence.length === 0) return null;
  const anyPicked = withEvidence.some((p) => p.sentiment!.aspects.some((a) => a.picked));

  return (
    <div className="mt-10">
      <h2 className="text-lg font-semibold">口碑依據</h2>
      <p className="mt-1 text-sm text-text-muted">
        {anyPicked ? "標示「已加重」的是你指定的面向。" : ""}
        名次是在同類別、該面向論壇評價達 10 則的型號之間排的；評價不足的面向不排名，系統以類別平均計。
      </p>

      <div className="mt-4 grid items-start gap-4 md:grid-cols-2 2xl:grid-cols-3">
        {withEvidence.map((part) => {
          const { model, aspects } = part.sentiment!;
          if (model === null) {
            return (
              <div key={part.category} className="rounded-lg border border-border p-4 text-sm">
                <p className="font-medium text-text-muted">{part.category}</p>
                <p className="mt-3 text-text-muted">
                  這個型號在論壇上找不到對應的評論，口碑以{part.category}的類別平均計，
                  指定的面向對它沒有作用。
                </p>
              </div>
            );
          }
          // 指定的面向排最前面，其餘維持後端的順序
          const rows = [...aspects.filter((a) => a.picked), ...aspects.filter((a) => !a.picked)];
          return (
            <div key={part.category} className="rounded-lg border border-border p-4 text-sm">
              <div className="flex items-baseline justify-between gap-4">
                <p className="font-medium">
                  <span className="text-text-muted">{part.category}</span>
                  <span className="ml-2">{model}</span>
                </p>
                <Link
                  href={`/parts/${encodeURIComponent(model)}`}
                  className="shrink-0 text-xs text-text-muted underline-offset-4 transition-colors hover:text-text hover:underline"
                >
                  看原始評論
                </Link>
              </div>
              <ul className="mt-3 divide-y divide-border">
                {rows.map((a) => (
                  <AspectRow key={a.aspect} aspect={a} />
                ))}
              </ul>
            </div>
          );
        })}
      </div>

      <p className="mt-3 text-xs text-text-dim">
        指定的面向佔該零件口碑的一半，口碑又只是整體評分的一部分，預算與效能仍會影響選擇，
        所以選到的零件不一定是該面向的第一名。面向裡的「CP值」是論壇網友的評價，
        與偏好比重的「CP 值」（系統依價格與跑分計算）是兩回事。
      </p>
    </div>
  );
}

function AspectRow({ aspect: a }: { aspect: AspectEvidence }) {
  const total = a.positive + a.negative;
  const diff = a.score - a.category_avg;
  return (
    <li className={`py-2 ${a.picked ? "" : "text-text-muted"}`}>
      <div className="flex items-baseline justify-between gap-4">
        <span className={a.picked ? "font-medium text-text" : ""}>
          {a.aspect}
          {a.picked && (
            <span className="ml-2 rounded-sm bg-accent px-1.5 py-0.5 text-xs font-normal text-accent-inverse">
              已加重
            </span>
          )}
        </span>
        <span className="tabular-nums">
          {a.rank === null ? (
            <span className="text-text-dim">評價不足</span>
          ) : (
            <>
              第 <span className="font-semibold text-text">{a.rank}</span> / {a.ranked_total} 名
            </>
          )}
        </span>
      </div>
      <p className="mt-0.5 text-xs tabular-nums text-text-dim">
        {total === 0 ? "沒有評價" : `正面 ${a.positive} 則・負面 ${a.negative} 則`}
        {a.rank !== null && (
          <>
            ・分數 {a.score.toFixed(2)}（
            {Math.abs(diff) < 0.005 ? "與類別平均相同" : `${diff > 0 ? "高於" : "低於"}類別平均 ${a.category_avg.toFixed(2)}`}
            ）
          </>
        )}
      </p>
    </li>
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

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
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
