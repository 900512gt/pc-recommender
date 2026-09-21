"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Swatch } from "../components/charts";
import { fetchModelIndex, type ModelIndexRow } from "../lib/rag-api";

/**
 * 正負分布條的三段，依序由左到右。標題旁的圖例與長條本身都讀這一份，
 * 改顏色或順序只要動這裡，圖例不會跟長條對不上。
 * 順序與詳情頁的口碑走勢圖一致（負評在最前面）。
 */
const LABEL_SEGMENTS = [
  { key: "負評", short: "負", color: "var(--text)" },
  { key: "正評", short: "正", color: "var(--text-dim)" },
  { key: "中立", short: "中", color: "var(--border-strong)" },
] as const;

export default function PartsPage() {
  const [rows, setRows] = useState<ModelIndexRow[] | null>(null);
  const [keyword, setKeyword] = useState("");
  const [category, setCategory] = useState("全部");

  useEffect(() => {
    fetchModelIndex().then(setRows);
  }, []);

  const categories = useMemo(() => {
    if (!rows) return [];
    return ["全部", ...Array.from(new Set(rows.map((r) => r.category))).sort()];
  }, [rows]);

  const filtered = useMemo(() => {
    if (!rows) return [];
    const kw = keyword.trim().toLowerCase();
    return rows.filter(
      (r) =>
        (category === "全部" || r.category === category) &&
        (kw === "" || r.model.toLowerCase().includes(kw)),
    );
  }, [rows, keyword, category]);

  return (
    <main className="mx-auto w-full max-w-360 flex-1 px-6 py-16">
      <h1 className="text-4xl font-semibold tracking-tight">零件口碑</h1>
      <p className="mt-2 max-w-2xl text-text-muted">
        161 個型號的 PTT 與巴哈姆特討論分析。點進任一型號可看口碑走勢、面向評分與原始評論。
      </p>

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <input
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
          placeholder="搜尋型號…"
          className="w-56 rounded-sm border border-border-strong px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
        />
        <div className="flex flex-wrap gap-1.5">
          {categories.map((c) => (
            <button
              key={c}
              onClick={() => setCategory(c)}
              className={`rounded-sm border px-2.5 py-1 text-sm transition-colors ${
                c === category
                  ? "border-accent bg-accent text-accent-inverse"
                  : "border-border-strong text-text-muted hover:text-text"
              }`}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      {rows === null ? (
        <p className="mt-10 text-sm text-text-muted">載入中…</p>
      ) : (
        <>
          <p className="mt-6 text-sm text-text-dim">
            共 {filtered.length} 個型號 · 點任一列查看完整分析
          </p>
          <table className="mt-3 w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-text text-left">
                <th className="py-2 font-medium">型號</th>
                <th className="py-2 font-medium">類別</th>
                <th className="py-2 pr-8 font-medium text-right">討論則數</th>
                <th className="py-2 font-medium">
                  <span className="flex items-center gap-2.5">
                    正負分布
                    {/* 長條只有顏色沒有文字，不附圖例的話看不出哪段是哪種評價 */}
                    <span className="flex items-center gap-2 text-xs font-normal text-text-dim">
                      {LABEL_SEGMENTS.map((s) => (
                        <Swatch key={s.key} color={s.color} label={s.short} />
                      ))}
                    </span>
                  </span>
                </th>
                <th className="py-2 font-medium text-right">報價單價格</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.model} className="group border-b border-border transition-colors hover:bg-bg-alt">
                  <td className="py-2">
                    {/* 整列都是連結：點擊目標大、hover 有底色，一眼看得出可以點 */}
                    <Link
                      href={`/parts/${encodeURIComponent(r.model)}`}
                      className="flex items-center gap-1.5 font-medium underline decoration-border-strong underline-offset-4 group-hover:decoration-text"
                    >
                      {r.model}
                      <span className="text-text-dim opacity-0 transition-opacity group-hover:opacity-100">
                        →
                      </span>
                    </Link>
                  </td>
                  <td className="py-2 text-text-muted">{r.category}</td>
                  <td className="py-2 pr-8 text-right text-text-muted">
                    {r.review_count.toLocaleString()}
                  </td>
                  <td className="py-2">
                    <LabelBar distribution={r.label_distribution} />
                  </td>
                  <td className="py-2 text-right text-text-muted">
                    {r.price ? `NT$${r.price.toLocaleString()}` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="mt-4 max-w-2xl text-xs text-text-dim">
            價格取自原價屋報價單的單次快照，不是即時價格；快照裡沒有收錄的型號顯示為「—」。
            上一代顯卡等未收錄的型號可能仍在市面上流通，「—」只代表這份報價單上查不到，
            不代表已經停產。
          </p>
        </>
      )}
    </main>
  );
}

/** 正負中立的比例條。顏色與順序見 LABEL_SEGMENTS。 */
function LabelBar({ distribution }: { distribution: Record<string, number> }) {
  const segments = LABEL_SEGMENTS.map((s) => ({
    value: distribution[s.key] ?? 0,
    color: s.color,
    label: s.key,
  }));
  const total = segments.reduce((sum, s) => sum + s.value, 0);
  if (total === 0) return <span className="text-text-dim">—</span>;

  return (
    <span
      className="flex h-2 w-28 overflow-hidden rounded-xs"
      title={segments.map((s) => `${s.label} ${s.value}`).join("・")}
    >
      {segments.map((s) => (
        <span
          key={s.label}
          style={{ width: `${(s.value / total) * 100}%`, background: s.color }}
        />
      ))}
    </span>
  );
}
