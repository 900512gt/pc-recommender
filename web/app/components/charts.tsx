"use client";

import type { ModelAspects, Timeline } from "../lib/rag-api";

/**
 * 口碑走勢圖。主圖是「比例」堆疊而不是數量堆疊——各月討論量相差可到 60 倍
 * （RTX5080 上市前每月十幾則、上市當月 728 則），照數量畫的話低量月份會細到看不見。
 * 比例圖看得出風向變化，下方另外附一條討論量帶補回聲量資訊，兩者共用同一組欄位對齊。
 *
 * SVG 用 preserveAspectRatio="none" 只在水平方向拉伸；裡面全是矩形，拉伸不會變形，
 * 文字標籤都放在 SVG 外面用 HTML 排。
 */
export function TimelineChart({ data, height = "h-40" }: { data: Timeline; height?: string }) {
  const { months } = data;
  const counts = months.map((m) => m.positive + m.negative + m.neutral);
  const peak = Math.max(...counts);

  return (
    <div className="flex flex-col gap-1.5">
      <svg
        viewBox={`0 0 ${months.length} 100`}
        preserveAspectRatio="none"
        className={`${height} w-full`}
        role="img"
        aria-label={`${data.model} 逐月正負評比例`}
      >
        {months.map((m, i) => {
          const total = counts[i];
          if (total === 0) return null;
          const neg = (m.negative / total) * 100;
          const pos = (m.positive / total) * 100;
          const neu = (m.neutral / total) * 100;
          return (
            <g key={m.month}>
              <title>{`${m.month}　負評 ${m.negative}・正評 ${m.positive}・中立 ${m.neutral}`}</title>
              <rect x={i + 0.08} y={0} width={0.84} height={neg} fill="var(--text)" />
              <rect x={i + 0.08} y={neg} width={0.84} height={pos} fill="var(--text-dim)" />
              <rect x={i + 0.08} y={neg + pos} width={0.84} height={neu} fill="var(--border-strong)" />
            </g>
          );
        })}
      </svg>

      <svg
        viewBox={`0 0 ${months.length} 100`}
        preserveAspectRatio="none"
        className="h-4 w-full"
        role="img"
        aria-label="每月討論量"
      >
        {months.map((m, i) => {
          const h = peak ? (counts[i] / peak) * 100 : 0;
          return (
            <rect key={m.month} x={i + 0.08} y={100 - h} width={0.84} height={h} fill="var(--border-strong)">
              <title>{`${m.month}　共 ${counts[i]} 則`}</title>
            </rect>
          );
        })}
      </svg>

      <div className="flex justify-between text-xs text-text-dim">
        <span>{months[0].month}</span>
        <span>{months[months.length - 1].month}</span>
      </div>

      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-text-dim">
        <Swatch color="var(--text)" label="負評" />
        <Swatch color="var(--text-dim)" label="正評" />
        <Swatch color="var(--border-strong)" label="中立" />
        <span className="ml-auto">下方細帶為每月討論量</span>
      </div>
    </div>
  );
}

// 灰階設計系統不能用顏色區分型號，改用線條樣式：實線 → 虛線 → 點線
const LINE_STYLES = [undefined, "5 3", "1.5 2.5"] as const;

const RADAR_SIZE = 220;
const RADAR_CENTER = RADAR_SIZE / 2;
const RADAR_RADIUS = 72;

export const DIMENSION_ORDER = ["效能", "溫控", "噪音", "保固", "CP值"] as const;

/**
 * 面向口碑雷達圖。分數來自 fine-tune 過的中研院 BERT 跑完 50,593 則評論的結果，
 * 不是 LLM 即時推估的。
 *
 * 只畫「每個被比較的型號都有足夠樣本」的面向：某個型號的噪音只有三則評論時，
 * 把它跟另一款有兩百則的放在同一張圖上比較會誤導，寧可整個軸不畫。
 */
export function RadarChart({ models }: { models: ModelAspects[] }) {
  const shared = DIMENSION_ORDER.filter((name) =>
    models.every((m) => m.aspects.find((a) => a.name === name)?.sufficient),
  );
  if (shared.length < 3) return null;

  const angleOf = (i: number) => (i / shared.length) * 2 * Math.PI - Math.PI / 2;
  const pointAt = (i: number, ratio: number) => {
    const angle = angleOf(i);
    return [
      RADAR_CENTER + Math.cos(angle) * RADAR_RADIUS * ratio,
      RADAR_CENTER + Math.sin(angle) * RADAR_RADIUS * ratio,
    ] as const;
  };
  const polygon = (ratios: number[]) => ratios.map((r, i) => pointAt(i, r).join(",")).join(" ");

  return (
    <div className="flex flex-col gap-2">
      <svg viewBox={`0 0 ${RADAR_SIZE} ${RADAR_SIZE}`} className="w-full max-w-64" role="img" aria-label="面向口碑比較">
        {[0.25, 0.5, 0.75, 1].map((ring) => (
          <polygon
            key={ring}
            points={polygon(shared.map(() => ring))}
            fill="none"
            stroke="var(--border)"
            strokeWidth="0.6"
          />
        ))}
        {shared.map((name, i) => {
          const [x, y] = pointAt(i, 1);
          const [lx, ly] = pointAt(i, 1.2);
          return (
            <g key={name}>
              <line x1={RADAR_CENTER} y1={RADAR_CENTER} x2={x} y2={y} stroke="var(--border)" strokeWidth="0.6" />
              <text x={lx} y={ly} fill="var(--text-dim)" fontSize="10" textAnchor="middle" dominantBaseline="middle">
                {name}
              </text>
            </g>
          );
        })}
        {models.map((m, mi) => (
          <polygon
            key={m.model}
            points={polygon(shared.map((name) => m.aspects.find((a) => a.name === name)?.score ?? 0))}
            fill="var(--text)"
            fillOpacity={0.06}
            stroke="var(--text)"
            strokeWidth="1.4"
            strokeDasharray={LINE_STYLES[mi % LINE_STYLES.length]}
          />
        ))}
      </svg>

      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-text-dim">
        {models.map((m, mi) => (
          <span key={m.model} className="flex items-center gap-1">
            <svg width="14" height="6" aria-hidden="true">
              <line
                x1="0"
                y1="3"
                x2="14"
                y2="3"
                stroke="var(--text)"
                strokeWidth="1.4"
                strokeDasharray={LINE_STYLES[mi % LINE_STYLES.length]}
              />
            </svg>
            {m.model}
          </span>
        ))}
        <span className="ml-auto">0～1，越外圈評價越好</span>
      </div>
    </div>
  );
}

function Swatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="inline-block h-2 w-2 rounded-xs" style={{ background: color }} />
      {label}
    </span>
  );
}
