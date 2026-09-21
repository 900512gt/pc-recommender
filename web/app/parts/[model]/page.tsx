"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { RadarChart, TimelineChart } from "../../components/charts";
import {
  fetchAspects,
  fetchModelComments,
  fetchModelDetail,
  fetchTimeline,
  type ModelAspects,
  type ModelDetail,
  type SourceComment,
  type Timeline,
} from "../../lib/rag-api";

const SOURCE_LABEL: Record<string, string> = { ptt: "PTT", bahamut: "巴哈" };

export default function ModelDetailPage() {
  const params = useParams<{ model: string }>();
  const model = decodeURIComponent(params.model);

  // 四支端點的結果連同「這是哪個型號的資料」一起存。這樣 loading 可以從 state 推導
  // 而不必在 effect 裡同步 setLoading(true)（那會多觸發一輪重繪，eslint 也會擋），
  // 而且從 A 型號跳到 B 型號時，不會有一瞬間把 A 的資料掛在 B 的標題底下。
  const [loaded, setLoaded] = useState<{
    model: string;
    detail: ModelDetail | null;
    timeline: Timeline | null;
    aspects: ModelAspects | null;
    comments: SourceComment[];
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    // 四支端點互相獨立，同時發出去；走勢與面向資料不足時會回 null，該區塊就不顯示
    Promise.all([
      fetchModelDetail(model),
      fetchTimeline(model),
      fetchAspects(model),
      fetchModelComments(model),
    ]).then(([detail, timeline, aspects, comments]) => {
      if (cancelled) return;
      setLoaded({ model, detail, timeline, aspects, comments });
    });
    return () => {
      cancelled = true;
    };
  }, [model]);

  const ready = loaded?.model === model ? loaded : null;
  const detail = ready?.detail ?? null;
  const timeline = ready?.timeline ?? null;
  const aspects = ready?.aspects ?? null;
  const comments = ready?.comments ?? [];

  if (!ready) {
    return <Shell><p className="text-sm text-text-muted">載入中…</p></Shell>;
  }
  if (!detail) {
    return (
      <Shell>
        <p className="text-sm text-text-muted">找不到「{model}」這個型號。</p>
        <Link href="/parts" className="mt-4 inline-block text-sm hover:underline">
          ← 回到零件口碑列表
        </Link>
      </Shell>
    );
  }

  return (
    <Shell>
      <Link href="/parts" className="text-sm text-text-muted hover:text-text">
        ← 零件口碑
      </Link>

      <div className="mt-4 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h1 className="text-4xl font-semibold tracking-tight">{detail.model}</h1>
        <span className="text-text-muted">{detail.category}</span>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-4">
        <Stat label="論壇討論" value={`${detail.review_count.toLocaleString()} 則`} />
        <Stat
          label="資料期間"
          value={
            detail.data_start_date && detail.data_end_date
              ? `${detail.data_start_date.slice(0, 7)} ~ ${detail.data_end_date.slice(0, 7)}`
              : "—"
          }
        />
        <Stat
          label="報價單價格"
          value={detail.listing?.price ? `NT$${detail.listing.price.toLocaleString()}` : "—"}
        />
        <Stat
          label="PassMark"
          value={detail.listing?.benchmark ? detail.listing.benchmark.toLocaleString() : "—"}
        />
      </div>

      {!detail.listing && (
        <p className="mt-3 max-w-3xl text-xs text-text-dim">
          原價屋報價單快照裡沒有收錄這個型號，所以沒有價格與跑分。這不代表它已經停產
          ——上一代零件常常只是從這份報價單上下架，市面上仍買得到。口碑資料不受影響。
        </p>
      )}

      {detail.summary && (
        <section className="mt-10">
          <h2 className="text-lg font-semibold">論壇整體印象</h2>
          <p className="mt-3 max-w-3xl leading-relaxed text-text-muted">{detail.summary}</p>
        </section>
      )}

      {(timeline || aspects) && (
        <section className="mt-10 grid gap-8 lg:grid-cols-2">
          {timeline && (
            <div>
              <h2 className="text-lg font-semibold">口碑走勢</h2>
              <p className="mt-1 text-sm text-text-muted">
                近 {timeline.months.length} 個月、{timeline.total.toLocaleString()} 則討論的正負評比例變化。
              </p>
              <div className="mt-4">
                <TimelineChart data={timeline} />
              </div>
            </div>
          )}
          {aspects && (
            <div>
              <h2 className="text-lg font-semibold">面向評分</h2>
              <p className="mt-1 text-sm text-text-muted">
                由 fine-tune 過的中研院 BERT 分析全部評論得出，非 LLM 推估。
              </p>
              <div className="mt-4">
                <RadarChart models={[aspects]} />
              </div>
            </div>
          )}
        </section>
      )}

      {detail.aspects.length > 0 && (
        <section className="mt-10">
          <h2 className="text-lg font-semibold">各面向討論重點</h2>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            {detail.aspects.map((a, i) => (
              <div key={i} className="rounded-lg border border-border p-4">
                <p className="text-sm font-medium">{a.aspect}</p>
                <p className="mt-1.5 text-sm leading-relaxed text-text-muted">{a.text}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {detail.pros_cons.length > 0 && (
        <section className="mt-10">
          <h2 className="text-lg font-semibold">優缺點整理</h2>
          {detail.pros_cons.map((text, i) => (
            <p key={i} className="mt-3 max-w-3xl whitespace-pre-wrap leading-relaxed text-text-muted">
              {text}
            </p>
          ))}
        </section>
      )}

      {detail.comparisons.length > 0 && (
        <section className="mt-10">
          <h2 className="text-lg font-semibold">常被拿來比較的型號</h2>
          {detail.comparisons.map((text, i) => (
            <p key={i} className="mt-3 max-w-3xl whitespace-pre-wrap leading-relaxed text-text-muted">
              {text}
            </p>
          ))}
        </section>
      )}

      {comments.length > 0 && (
        <section className="mt-10">
          <h2 className="text-lg font-semibold">原始論壇評論</h2>
          <p className="mt-1 text-sm text-text-muted">
            正負評交錯取樣的 {comments.length} 則，點擊可開啟論壇原文。
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {comments.map((c, i) => (
              <a
                key={i}
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex flex-col gap-1.5 rounded-lg border border-border p-3 transition-colors hover:border-border-strong"
              >
                <div className="flex items-center gap-1.5 text-xs text-text-dim">
                  <span
                    className={`rounded-sm px-1 py-px ${
                      c.label === "負評"
                        ? "bg-accent text-accent-inverse"
                        : c.label === "正評"
                          ? "border border-border-strong text-text"
                          : "text-text-dim"
                    }`}
                  >
                    {c.label ?? "未分類"}
                  </span>
                  <span>{SOURCE_LABEL[c.source] ?? c.source}</span>
                  {c.date && <span>{c.date}</span>}
                </div>
                <p className="text-sm leading-relaxed text-text-muted">{c.content}</p>
              </a>
            ))}
          </div>
        </section>
      )}
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <main className="mx-auto w-full max-w-360 flex-1 px-6 py-16">{children}</main>;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-bg px-4 py-3">
      <p className="text-xs text-text-muted">{label}</p>
      <p className="mt-0.5 text-lg font-medium">{value}</p>
    </div>
  );
}
