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
  // 16-pin 顯卡能不能用轉接線接在沒有原生 16-pin 的電源上；預設不允許（後端只推薦有原生接頭的電源）
  const [allowAdapter, setAllowAdapter] = useState(false);
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
        allow_16pin_adapter: allowAdapter,
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
    <main className="mx-auto w-full max-w-360 flex-1 px-6 py-12">
      <h1 className="text-3xl font-semibold tracking-tight">配置建置器</h1>
      <p className="mt-2 text-sm text-text-muted">
        輸入預算與需求，透過遺傳演算法從零件資料庫中挑出最佳組合。
      </p>

      {/* 寬螢幕左邊放表單、右邊放結果，結果不必捲到表單底下才看得到；窄螢幕照原本上下排 */}
      <div className="mt-8 grid items-start gap-8 lg:grid-cols-[22rem_minmax(0,1fr)]">
        {/*
          大螢幕把設定欄釘在畫面上，捲動右側長報表時仍能調整條件。設定欄比視窗高時在
          面板內捲動，「生成配置」用 sticky 黏在面板底部，不會被捲出畫面。
        */}
        <form
          onSubmit={onSubmit}
          className="flex flex-col rounded-lg border border-border lg:sticky lg:top-6 lg:max-h-[calc(100vh-3rem)] lg:overflow-y-auto"
        >
          <div className="grid gap-6 p-5">
            <div className="grid grid-cols-2 gap-3">
              <label className="col-span-2 flex flex-col gap-1.5">
                <FieldLabel>預算</FieldLabel>
                <div className="flex items-center rounded-sm border border-border-strong focus-within:ring-2 focus-within:ring-accent">
                  <span className="pl-3 font-mono text-sm text-text-dim">NT$</span>
                  <input
                    type="number"
                    min={MIN_BUDGET}
                    step={1000}
                    value={budget}
                    onChange={(e) => setBudget(e.target.value)}
                    className="w-full min-w-0 bg-transparent px-2 py-2 font-mono text-sm tabular-nums outline-none"
                  />
                </div>
              </label>

              <label className="flex flex-col gap-1.5">
                <FieldLabel>用途</FieldLabel>
                <select
                  value={usage}
                  onChange={(e) => {
                    const next = e.target.value as Usage;
                    setUsage(next);
                    // 滑桿跟著跳到新用途的預設位置，不然會把上一個用途的比重帶過來
                    setWeights(defaultSliders(next));
                  }}
                  className="rounded-sm border border-border-strong bg-bg px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
                >
                  <option value="工作">工作</option>
                  <option value="遊戲">遊戲</option>
                </select>
              </label>

              <label className="flex flex-col gap-1.5">
                <FieldLabel>散熱偏好</FieldLabel>
                <select
                  value={cooling}
                  onChange={(e) => setCooling(e.target.value as CoolingPreference)}
                  className="rounded-sm border border-border-strong bg-bg px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
                >
                  <option value="auto">自動</option>
                  <option value="風冷">風冷</option>
                </select>
              </label>
            </div>

            <Divider />

            <PreferenceSliders usage={usage} value={weights} onChange={setWeights} />

            {selectable.length > 0 && (
              <>
                <Divider />
                <AspectPicker options={selectable} value={aspectPrefs} onChange={setAspectPrefs} />
              </>
            )}

            <Divider />

            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={allowAdapter}
                  onChange={(e) => setAllowAdapter(e.target.checked)}
                  className="size-4 accent-current"
                />
                接受 16-pin 顯卡使用轉接線
              </label>
              <Note>
                不勾選時只推薦有原生 16-pin 接頭的電源。轉接線曾有燒熔案例，廠商建議搭配 ATX 3.x 電源。
              </Note>
            </div>
          </div>

          <div className="sticky bottom-0 mt-auto border-t border-border bg-bg p-5">
            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-sm bg-accent px-4 py-2 text-sm font-medium text-accent-inverse transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {loading ? "生成中…" : "生成配置"}
            </button>
          </div>
        </form>

        <div className="min-w-0">
          {error && (
            <div className="mb-6 rounded-sm border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {result && <ResultView result={result} />}

          {!result && !error && <EmptyState loading={loading} />}
        </div>
      </div>
    </main>
  );
}

// ── 共用的小元件 ─────────────────────────────────────────────────────────────

const MONO = "font-mono tabular-nums";

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <span className="text-xs font-medium text-text-muted">{children}</span>;
}

function Divider() {
  return <hr className="border-border" />;
}

/** Vercel 風格的說明區塊：細邊框、極淺灰底，用來放補充說明，不跟主要控制項搶視線。 */
function Note({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded-sm border border-border bg-bg-alt px-3 py-2 text-xs leading-relaxed text-text-muted">
      {children}
    </p>
  );
}

/** 等寬的小標籤：名次、加價、效能變化這類要能並排比較的數字。emphasis 用黑框加粗強調（設計系統不用彩色）。 */
function MonoBadge({ children, emphasis = false }: { children: React.ReactNode; emphasis?: boolean }) {
  return (
    <span
      className={`inline-flex items-center rounded-sm border px-1.5 py-0.5 text-xs whitespace-nowrap ${MONO} ${
        emphasis ? "border-accent font-semibold text-text" : "border-border text-text-muted"
      }`}
    >
      {children}
    </span>
  );
}

/** 0~1 分數的迷你橫條：底色淺灰、填色黑，用來快速掃出高分與低分的零件。 */
function MiniBar({ value, width = "w-10" }: { value: number; width?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <span className={`inline-block h-1 ${width} overflow-hidden rounded-full bg-border`} aria-hidden>
      <span className="block h-full bg-text" style={{ width: `${pct}%` }} />
    </span>
  );
}

/**
 * 商品名稱降噪：原價屋的品名把型號和規格寫在一起，例如
 * 「華碩 DUAL-RX9060XT-8G(3150MHz/21cm/雙風扇/註冊四年保)金屬背板」、
 * 「致態 ZhiTai TiPlus7100s 1TB/Gen4/PCIe 4.0/讀7400/寫6900/TLC」。
 * 把 ()、（）、【】、[] 裡的內容、結尾的「*最後優惠」這類促銷字樣，以及第一個「/」之後
 * 用斜線分隔的規格拆出來，剩下的當主名稱。只影響顯示，原始字串不變；
 * 拆完沒有主名稱時退回原字串。
 */
function splitPartName(name: string): { main: string; specs: string[] } {
  const specs: string[] = [];
  const stripped = name
    .replace(/[(（【\[]([^)）】\]]*)[)）】\]]/g, (_, inner: string) => {
      specs.push(...inner.split("/").map((s) => s.trim()).filter(Boolean));
      return " ";
    })
    .replace(/\*([^*]+)$/, (_, promo: string) => {
      specs.push(promo.trim());
      return " ";
    })
    .replace(/\s+/g, " ")
    .trim();
  const [main, ...slashSpecs] = stripped.split("/").map((s) => s.trim());
  const allSpecs = [...slashSpecs.filter(Boolean), ...specs];
  return main ? { main, specs: allSpecs } : { main: name, specs: [] };
}

function money(n: number): string {
  return `NT$${n.toLocaleString()}`;
}

// ── 左側設定欄 ───────────────────────────────────────────────────────────────

/**
 * 效能／口碑／CP 值的偏好比重。右邊顯示的是三項之間的相對佔比而不是滑桿原始值，
 * 讓使用者看得出「拉到 100」不等於「100%」。上方的分段條把三者佔比畫成一條，
 * 用黑、深灰、中灰三階區分（設計系統沒有彩色，靠灰階層次分辨）。
 */
const PREFERENCE_TONES = ["bg-text", "bg-text-muted", "bg-text-dim"] as const;

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
    <div role="group" aria-labelledby={`${id}-title`} className="flex flex-col gap-3">
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
            恢復預設
          </button>
        ) : (
          <span className="text-xs text-text-dim">「{usage}」系統預設</span>
        )}
      </div>

      {/* 三項佔比的分段條；三項都是 0 時顯示一條空的底色 */}
      <div className="flex h-1.5 gap-px overflow-hidden rounded-full bg-border" aria-hidden>
        {total > 0 &&
          PREFERENCES.map((p, i) =>
            value[p.key] > 0 ? (
              <span
                key={p.key}
                className={`h-full transition-[flex-grow] ${PREFERENCE_TONES[i]}`}
                style={{ flexGrow: value[p.key] }}
              />
            ) : null,
          )}
      </div>

      {PREFERENCES.map((p, i) => {
        const share = total > 0 ? Math.round((value[p.key] / total) * 100) : null;
        const inputId = `${id}-${p.key}`;
        return (
          <div key={p.key} className="flex flex-col gap-0.5">
            <div className="flex items-baseline justify-between gap-4">
              <label htmlFor={inputId} className="flex items-center gap-2 text-sm text-text-muted">
                <span className={`size-2 rounded-[2px] ${PREFERENCE_TONES[i]}`} aria-hidden />
                {p.label}
              </label>
              {/* 右對齊＋等寬數字，拖動時數字改變不會讓位置左右跳動 */}
              <output htmlFor={inputId} className={MONO}>
                {share === null ? (
                  <span className="text-sm font-semibold text-text-dim">—</span>
                ) : (
                  <>
                    <span className="text-sm font-semibold">{share}</span>
                    <span className="ml-0.5 text-xs text-text-dim">%</span>
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
 * 口碑裡特別在意的面向（選填）。指定的面向從口碑比重中分出一半單獨計分，
 * 所以是「加重」而不是「只看」。選項只列後端認定評論資料夠的面向。
 *
 * 用可複選的切換鈕而不是下拉選單：每類只有 1~3 個選項，一眼看得完，也不必多點一下。
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
    <div role="group" aria-labelledby={`${id}-title`} className="flex flex-col gap-3">
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

      <dl className="grid grid-cols-[3.5rem_minmax(0,1fr)] items-center gap-x-3 gap-y-2">
        {options.map((opt) => (
          <div key={opt.category} className="contents">
            <dt className={`text-xs text-text-dim ${MONO}`}>{opt.category}</dt>
            <dd className="flex flex-wrap gap-1.5">
              {opt.aspects.map((aspect) => {
                const on = value[opt.category]?.includes(aspect) ?? false;
                return (
                  <button
                    key={aspect}
                    type="button"
                    aria-pressed={on}
                    onClick={() => toggle(opt.category, aspect)}
                    className={`inline-flex items-center gap-1 rounded-sm border px-2.5 py-1 text-sm outline-none transition-colors focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 ${
                      on
                        ? "border-accent bg-accent text-accent-inverse"
                        : "border-border-strong text-text-muted hover:border-text-dim hover:text-text"
                    }`}
                  >
                    {on && (
                      <svg viewBox="0 0 12 12" className="size-3" aria-hidden>
                        <path d="M2.5 6.5 5 9l4.5-5.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
                      </svg>
                    )}
                    {aspect}
                  </button>
                );
              })}
            </dd>
          </div>
        ))}
      </dl>

      <p className="text-xs text-text-dim">
        指定的面向佔口碑比重的一半，其餘面向仍會列入。只列出論壇評論夠多、分得出型號差異的面向。
      </p>
    </div>
  );
}

// ── 右側：未生成前 ───────────────────────────────────────────────────────────

/** 細格線背景用 SVG pattern 畫（設計系統不用漸層），線條顏色跟著主題的邊框色。 */
function EmptyState({ loading }: { loading: boolean }) {
  const id = useId();
  return (
    <div className="relative overflow-hidden rounded-lg border border-border">
      <svg className="absolute inset-0 size-full text-border" aria-hidden>
        <defs>
          <pattern id={`${id}-grid`} width="24" height="24" patternUnits="userSpaceOnUse">
            <path d="M24 0H0V24" fill="none" stroke="currentColor" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill={`url(#${id}-grid)`} />
      </svg>
      <div className="relative flex flex-col items-center px-6 py-20 text-center">
        <div className="flex size-12 items-center justify-center rounded-lg border border-border-strong bg-bg">
          <svg viewBox="0 0 24 24" className={`size-6 text-text ${loading ? "animate-pulse" : ""}`} aria-hidden>
            <rect x="6" y="6" width="12" height="12" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <rect x="9.5" y="9.5" width="5" height="5" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <path
              d="M9 3v3M12 3v3M15 3v3M9 18v3M12 18v3M15 18v3M3 9h3M3 12h3M3 15h3M18 9h3M18 12h3M18 15h3"
              stroke="currentColor"
              strokeWidth="1.5"
            />
          </svg>
        </div>
        {loading ? (
          <>
            <p className="mt-5 text-base font-semibold">正在搜尋最佳組合</p>
            <p className="mt-1 text-sm text-text-muted">族群 300 × 演化 300 代，大約需要 10~15 秒。</p>
          </>
        ) : (
          <>
            <p className="mt-5 text-base font-semibold">尚未生成配置</p>
            <p className="mt-1 max-w-sm text-sm text-text-muted">
              在左側設定條件後按「生成配置」，推薦的零件與口碑依據會顯示在這裡。
            </p>
            <ol className="mt-6 flex flex-wrap justify-center gap-2 text-xs text-text-muted">
              {["輸入預算與用途", "調整偏好比重（選填）", "生成配置"].map((step, i) => (
                <li key={step} className="flex items-center gap-1.5 rounded-sm border border-border bg-bg px-2 py-1">
                  <span className={`text-text-dim ${MONO}`}>{i + 1}</span>
                  {step}
                </li>
              ))}
            </ol>
          </>
        )}
      </div>
    </div>
  );
}

// ── 右側：結果 ───────────────────────────────────────────────────────────────

function ResultView({ result }: { result: RecommendResponse }) {
  const used = result.budget > 0 ? result.total_price / result.budget : 0;
  const { ok, issues } = result.compatibility;

  return (
    <section className="flex flex-col gap-10">
      {/* 預算摘要：1px 共用邊框的網格，底部一條預算使用率 */}
      <div className="overflow-hidden rounded-lg border border-border">
        <div className="grid grid-cols-3 divide-x divide-border">
          <Stat label="總價" value={money(result.total_price)} strong />
          <Stat label="預算" value={money(result.budget)} />
          {/* GA 允許超支 20% 以內（會扣分），超支時改標「超支」，不顯示負的剩餘 */}
          <Stat
            label={result.remaining < 0 ? "超支" : "剩餘"}
            value={money(Math.abs(result.remaining))}
          />
        </div>
        <div className="flex items-center gap-3 border-t border-border px-4 py-2.5">
          <span className="h-1 flex-1 overflow-hidden rounded-full bg-border" aria-hidden>
            <span className="block h-full bg-text" style={{ width: `${Math.min(used, 1) * 100}%` }} />
          </span>
          <span className={`text-xs text-text-muted ${MONO}`}>預算使用 {(used * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-semibold">推薦配置</h2>
          <span className="inline-flex items-center gap-2 rounded-sm border border-border px-2 py-1 text-xs text-text-muted">
            <span className={`size-1.5 rounded-full ${ok ? "bg-emerald-500" : "bg-red-500"}`} aria-hidden />
            {ok ? "相容性檢查通過" : "相容性檢查發現問題"}
          </span>
        </div>

        {/* 允許轉接線時，「需使用轉接線」會以不扣分的提醒出現 */}
        {issues.length > 0 && (
          <ul
            className={`mt-3 list-disc rounded-sm border px-4 py-2 pl-8 text-xs leading-relaxed ${
              ok ? "border-border bg-bg-alt text-text-muted" : "border-red-300 bg-red-50 text-red-700"
            }`}
          >
            {issues.map((issue, i) => (
              <li key={i}>{issue}</li>
            ))}
          </ul>
        )}

        <PartsTable parts={result.parts} />
      </div>

      <SentimentEvidence parts={result.parts} />
    </section>
  );
}

function Stat({ label, value, strong = false }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="min-w-0 px-3 py-3 sm:px-4">
      <p className="text-xs text-text-muted">{label}</p>
      <p className={`mt-1 truncate text-sm sm:text-lg ${MONO} ${strong ? "font-semibold text-text" : "font-medium text-text-muted"}`}>
        {value}
      </p>
    </div>
  );
}

function PartsTable({ parts }: { parts: RecommendedPart[] }) {
  return (
    <table className="mt-3 w-full border-collapse text-sm">
      <thead>
        <tr className="border-b border-text text-left text-xs text-text-muted">
          <th className="hidden py-2 pr-4 font-medium sm:table-cell">類別</th>
          <th className="py-2 font-medium">品項</th>
          <th className="py-2 pl-4 text-right font-medium">價格</th>
          <th className="py-2 pl-6 text-right font-medium whitespace-nowrap">口碑分數</th>
        </tr>
      </thead>
      <tbody>
        {parts.map((part) => {
          const { main, specs } = splitPartName(part.name);
          return (
            <tr key={part.category} className="border-b border-border align-top">
              <td className="hidden py-2.5 pr-4 whitespace-nowrap text-text-muted sm:table-cell">{part.category}</td>
              <td className="py-2.5">
                {/* 窄螢幕不放類別欄，改成品名上方的小字 */}
                <p className="text-xs text-text-dim sm:hidden">{part.category}</p>
                <p className="font-medium text-text">{main}</p>
                {specs.length > 0 && (
                  <p className="mt-0.5 text-xs leading-relaxed text-text-dim">{specs.join(" · ")}</p>
                )}
              </td>
              <td className={`py-2.5 pl-4 text-right whitespace-nowrap ${MONO}`}>{money(part.price)}</td>
              <td className="py-2.5 pl-6">
                <span className="flex items-center justify-end gap-2">
                  <span className="hidden sm:inline-flex">
                    <MiniBar value={part.score} />
                  </span>
                  <span className={`w-9 text-right ${MONO}`}>{part.score.toFixed(2)}</span>
                </span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/**
 * 口碑分數的依據：每個零件各面向的名次與正負評價則數，並連到該型號的原始評論。
 * 讓使用者能自己核對「指定了某個面向，選到的這顆在該面向口碑到底好不好」，而不是只能相信系統。
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
    <div>
      <h2 className="text-base font-semibold">口碑依據</h2>
      <p className="mt-1 text-sm text-text-muted">
        {anyPicked ? "標示「已加重」的是你指定的面向。" : ""}
        名次是在同類別、該面向論壇評價達 10 則的型號之間排的；評價不足的面向不排名，系統以類別平均計。
      </p>

      <div className="mt-4 grid items-start gap-4 md:grid-cols-2 2xl:grid-cols-3">
        {withEvidence.map((part) => {
          const { model, aspects } = part.sentiment!;
          // 指定的面向排最前面，其餘維持後端的順序
          const rows = [...aspects.filter((a) => a.picked), ...aspects.filter((a) => !a.picked)];
          return (
            <div key={part.category} className="flex flex-col rounded-lg border border-border text-sm">
              <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
                <div className="flex min-w-0 items-center gap-2">
                  <span className="shrink-0 rounded-sm border border-border px-1.5 py-0.5 text-xs text-text-muted">
                    {part.category}
                  </span>
                  <span className={`truncate font-semibold ${MONO}`}>{model ?? "—"}</span>
                </div>
                {model && (
                  <Link
                    href={`/parts/${encodeURIComponent(model)}`}
                    className="shrink-0 text-xs text-text-muted underline-offset-4 transition-colors hover:text-text hover:underline"
                  >
                    看原始評論 ↗
                  </Link>
                )}
              </div>

              {model === null ? (
                <div className="m-4 rounded-sm border border-dashed border-border-strong bg-bg-alt px-4 py-6 text-center">
                  <p className="text-sm font-medium text-text-muted">找不到對應的論壇評論</p>
                  <p className="mt-1 text-xs leading-relaxed text-text-dim">
                    口碑以{part.category}的類別平均計，指定的面向對它沒有作用。
                  </p>
                </div>
              ) : (
                <ul className="divide-y divide-border px-4">
                  {rows.map((a) => (
                    <AspectRow key={a.aspect} aspect={a} category={part.category} />
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>

      <p className="mt-3 text-xs text-text-dim">
        指定的面向佔口碑比重的一半，口碑又只是整體評分的一部分，預算與效能仍會影響選擇，
        所以選到的零件不一定是該面向的第一名。
      </p>
    </div>
  );
}

/**
 * 一個面向的口碑：名次、正負評價則數與比例條。評價不足（沒有名次）的面向整列淡化，
 * 讓有實質排名的面向跳出來。比例條正面用黑、負面用淺灰，不用紅綠（設計系統不用彩色）。
 */
function AspectRow({ aspect: a, category }: { aspect: AspectEvidence; category: string }) {
  const total = a.positive + a.negative;
  const diff = a.score - a.category_avg;
  const ranked = a.rank !== null;
  return (
    <li className={`py-2.5 ${ranked || a.picked ? "" : "opacity-50"}`}>
      <div className="flex items-center justify-between gap-3">
        <span className={`flex items-center gap-2 ${a.picked ? "font-semibold text-text" : "text-text"}`}>
          {a.aspect}
          {a.picked && (
            <span className="rounded-sm bg-accent px-1.5 py-0.5 text-xs font-normal text-accent-inverse">已加重</span>
          )}
        </span>
        {ranked ? (
          <MonoBadge emphasis={a.rank === 1}>
            第 {a.rank} / {a.ranked_total} 名
          </MonoBadge>
        ) : (
          <span className="text-xs text-text-dim">評價不足</span>
        )}
      </div>

      {total > 0 && (
        <span className="mt-2 flex h-1 gap-px overflow-hidden rounded-full" aria-hidden>
          {a.positive > 0 && <span className="h-full bg-text" style={{ flexGrow: a.positive }} />}
          {a.negative > 0 && <span className="h-full bg-border-strong" style={{ flexGrow: a.negative }} />}
        </span>
      )}

      <p className={`mt-1.5 flex flex-wrap justify-between gap-x-3 text-xs text-text-dim ${MONO}`}>
        <span>{total === 0 ? "沒有評價" : `正面 ${a.positive}・負面 ${a.negative}`}</span>
        {ranked && (
          <span>
            {a.score.toFixed(2)}
            <span className="ml-1">
              {Math.abs(diff) < 0.005
                ? "＝ 類別平均"
                : `${diff > 0 ? "▲" : "▼"} 類別平均 ${a.category_avg.toFixed(2)}`}
            </span>
          </span>
        )}
      </p>

      {a.picked && <PickedHint aspect={a} category={category} />}
    </li>
  );
}

/**
 * 使用者指定的面向「看不出效果」的兩種情況，直接說明原因，不然會以為功能壞了：
 *   - 評價不足：這個零件在該面向不到 10 則評價，系統給類別平均，指定對它不加分也不扣分。
 *     常見於低預算：該面向評價較好的型號要換 CPU 平台或超出預算，權衡後不划算。
 *   - 第 1 名：已經是同類最好的，換不到更好的。
 */
function PickedHint({ aspect: a, category }: { aspect: AspectEvidence; category: string }) {
  if (a.rank === null) {
    return (
      <p className="mt-2 rounded-sm border border-border bg-bg-alt px-2.5 py-1.5 text-xs leading-relaxed text-text-muted">
        這款{category}的「{a.aspect}」評價不足 10 則，系統以類別平均計，指定的面向對它不加分也不扣分。
        「{a.aspect}」評價較好的型號在這個預算下需要換 CPU 平台或超出預算，權衡後沒有被選上。
      </p>
    );
  }
  if (a.rank === 1) {
    return (
      <p className="mt-2 rounded-sm border border-border bg-bg-alt px-2.5 py-1.5 text-xs leading-relaxed text-text-muted">
        已是同類「{a.aspect}」第 1 名，沒有更好的選擇。
      </p>
    );
  }
  return null;
}
