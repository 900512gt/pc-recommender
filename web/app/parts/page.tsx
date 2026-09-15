"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { fetchModelIndex, type ModelIndexRow } from "../lib/rag-api";

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
                <th className="py-2 font-medium">正負分布</th>
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

/** 正負中立的比例條。灰階：負評黑、正評中灰、中立淺灰，與各圖表一致。 */
function LabelBar({ distribution }: { distribution: Record<string, number> }) {
  const negative = distribution["負評"] ?? 0;
  const positive = distribution["正評"] ?? 0;
  const neutral = distribution["中立"] ?? 0;
  const total = negative + positive + neutral;
  if (total === 0) return <span className="text-text-dim">—</span>;

  const segments = [
    { value: negative, color: "var(--text)", label: "負評" },
    { value: positive, color: "var(--text-dim)", label: "正評" },
    { value: neutral, color: "var(--border-strong)", label: "中立" },
  ];

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
