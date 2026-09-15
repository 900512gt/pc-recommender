"use client";

import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import {
  fetchAspects,
  RagApiError,
  streamChat,
  type ChatMessage,
  type ModelAspects,
  type SourceComment,
  type SourceGroup,
} from "../lib/rag-api";

const EXAMPLES = [
  "RTX5070 值得買嗎？",
  "7800X3D 有什麼缺點？",
  "中階顯示卡推薦",
];

/** 畫面上的訊息比送回後端的 ChatMessage 多帶佐證與面向分數，送出前必須剝掉（見 send()）。 */
type DisplayMessage = ChatMessage & { sources?: SourceGroup[]; aspects?: ModelAspects[] };

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  async function send(query: string) {
    const q = query.trim();
    if (!q || busy) return;

    setError(null);
    // 只送 role/content：後端會把 history 原封不動塞進 OpenAI 的 messages，
    // 夾帶 sources 這種多餘欄位會讓 API 直接回 400。
    const historySnapshot = messages.map(({ role, content }) => ({ role, content }));
    // 面向分數是串流結束後才非同步補上的，這時可能已經有新訊息，所以先記住這則
    // 助理訊息的位置（送出被 busy 擋著序列化，不會有兩則同時在寫）。
    const assistantIndex = messages.length + 1;
    setMessages((m) => [...m, { role: "user", content: q }, { role: "assistant", content: "" }]);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setBusy(true);

    let models: string[] = [];
    try {
      for await (const event of streamChat(q, historySnapshot)) {
        if (event.type === "sources") models = event.groups.map((g) => g.model);
        setMessages((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          copy[copy.length - 1] =
            event.type === "text"
              ? { ...last, content: event.text }
              : { ...last, sources: event.groups };
          return copy;
        });
      }

      // 可信面向不足的型號會回 null，全部落空就不畫圖。
      const aspects = (await Promise.all(models.map(fetchAspects))).filter(
        (a): a is ModelAspects => a !== null,
      );
      if (aspects.length > 0) {
        setMessages((m) => {
          const copy = [...m];
          if (copy[assistantIndex]) copy[assistantIndex] = { ...copy[assistantIndex], aspects };
          return copy;
        });
      }
    } catch (err) {
      const msg = err instanceof RagApiError ? err.message : "發生未知錯誤";
      setError(msg);
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { role: "assistant", content: `發生錯誤：${msg}` };
        return copy;
      });
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send(input);
    }
  }

  function onInput() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
      {open && (
        <div
          className="flex w-96 flex-col overflow-hidden rounded-lg border border-border bg-bg"
          style={{ height: 520, boxShadow: "0 1px 2px rgba(0,0,0,0.06)" }}
        >
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <div>
              <p className="text-sm font-semibold">零件口碑問答</p>
              <p className="text-xs text-text-muted">PTT · 巴哈姆特 論壇評價</p>
            </div>
            <button
              onClick={() => setOpen(false)}
              aria-label="關閉"
              className="rounded-sm p-1 text-text-muted transition-colors hover:text-text"
            >
              ✕
            </button>
          </div>

          <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-4">
            {messages.length === 0 && (
              <div className="flex flex-col gap-2">
                <span className="text-xs text-text-muted">試試看這些問題：</span>
                <div className="flex flex-wrap gap-1.5">
                  {EXAMPLES.map((ex) => (
                    <button
                      key={ex}
                      onClick={() => send(ex)}
                      disabled={busy}
                      className="rounded-full border border-border px-2.5 py-1 text-xs text-text-muted transition-colors hover:border-border-strong hover:text-text disabled:opacity-50"
                    >
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div
                key={i}
                className={`flex flex-col gap-1.5 ${m.role === "user" ? "items-end" : "items-start"}`}
              >
                <div
                  className={`max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm leading-relaxed ${
                    m.role === "user"
                      ? "bg-accent text-accent-inverse"
                      : "border border-border text-text"
                  }`}
                >
                  {m.content || (busy && i === messages.length - 1 ? <TypingDots /> : "")}
                </div>
                {m.role === "assistant" && m.aspects && m.aspects.length > 0 && (
                  <div className="w-full rounded-md border border-border p-2.5 text-xs">
                    <RadarChart models={m.aspects} />
                  </div>
                )}
                {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                  <SourcePanel groups={m.sources} />
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          {error && (
            <div className="border-t border-red-300 bg-red-50 px-4 py-2 text-xs text-red-700">
              {error}
            </div>
          )}

          <div className="flex items-end gap-2 border-t border-border p-3">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onInput={onInput}
              onKeyDown={onKeyDown}
              placeholder="輸入型號或問題…"
              rows={1}
              className="max-h-30 flex-1 resize-none rounded-sm border border-border-strong px-2.5 py-1.5 text-sm outline-none focus:ring-2 focus:ring-accent"
            />
            <button
              onClick={() => send(input)}
              disabled={busy || !input.trim()}
              className="rounded-sm bg-accent px-3 py-1.5 text-sm font-medium text-accent-inverse transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              送出
            </button>
          </div>
        </div>
      )}

      <button
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "關閉聊天" : "開啟零件口碑問答"}
        className="flex h-14 w-14 items-center justify-center rounded-full bg-accent text-accent-inverse transition-opacity hover:opacity-90"
        style={{ boxShadow: "0 4px 16px rgba(0,0,0,0.18)" }}
      >
        {open ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        )}
      </button>
    </div>
  );
}

// 灰階設計系統不能用顏色區分型號，改用線條樣式：實線 → 虛線 → 點線
const LINE_STYLES = [undefined, "5 3", "1.5 2.5"] as const;

const RADAR_SIZE = 200;
const RADAR_CENTER = RADAR_SIZE / 2;
const RADAR_RADIUS = 58;

/**
 * 面向口碑雷達圖。分數來自 fine-tune 過的中研院 BERT 跑完 50,593 則評論的結果，
 * 不是 LLM 即時推估的。
 *
 * 只畫「每個被比較的型號都有足夠樣本」的面向：某個型號的噪音只有三則評論時，
 * 把它跟另一款有兩百則的放在同一張圖上比較會誤導，寧可整個軸不畫。
 */
function RadarChart({ models }: { models: ModelAspects[] }) {
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
  const polygon = (ratios: number[]) =>
    ratios.map((r, i) => pointAt(i, r).join(",")).join(" ");

  return (
    <div className="flex flex-col gap-1.5">
      <svg viewBox={`0 0 ${RADAR_SIZE} ${RADAR_SIZE}`} className="w-full" role="img" aria-label="面向口碑比較">
        {[0.25, 0.5, 0.75, 1].map((ring) => (
          <polygon
            key={ring}
            points={polygon(shared.map(() => ring))}
            fill="none"
            stroke="var(--border)"
            strokeWidth="0.5"
          />
        ))}
        {shared.map((name, i) => {
          const [x, y] = pointAt(i, 1);
          const [lx, ly] = pointAt(i, 1.22);
          return (
            <g key={name}>
              <line x1={RADAR_CENTER} y1={RADAR_CENTER} x2={x} y2={y} stroke="var(--border)" strokeWidth="0.5" />
              <text
                x={lx}
                y={ly}
                fill="var(--text-dim)"
                fontSize="9"
                textAnchor="middle"
                dominantBaseline="middle"
              >
                {name}
              </text>
            </g>
          );
        })}
        {models.map((m, mi) => (
          <polygon
            key={m.model}
            points={polygon(
              shared.map((name) => m.aspects.find((a) => a.name === name)?.score ?? 0),
            )}
            fill="var(--text)"
            fillOpacity={0.06}
            stroke="var(--text)"
            strokeWidth="1.2"
            strokeDasharray={LINE_STYLES[mi % LINE_STYLES.length]}
          />
        ))}
      </svg>

      <div className="flex flex-wrap gap-x-3 gap-y-1 text-text-dim">
        {models.map((m, mi) => (
          <span key={m.model} className="flex items-center gap-1">
            <svg width="14" height="6" aria-hidden="true">
              <line
                x1="0"
                y1="3"
                x2="14"
                y2="3"
                stroke="var(--text)"
                strokeWidth="1.2"
                strokeDasharray={LINE_STYLES[mi % LINE_STYLES.length]}
              />
            </svg>
            {m.model}
          </span>
        ))}
        <span className="ml-auto">分數為 0～1，越外圈評價越好</span>
      </div>
    </div>
  );
}

const DIMENSION_ORDER = ["效能", "溫控", "噪音", "保固", "CP值"] as const;

const SOURCE_LABEL = { ptt: "PTT", bahamut: "巴哈" } as const;

/**
 * 佐證面板：回答依據了哪些真實論壇評論。
 * 預設收合——大部分使用者只要答案，想查證的人才會展開。
 */
function SourcePanel({ groups }: { groups: SourceGroup[] }) {
  const [open, setOpen] = useState(false);
  const total = groups.reduce((sum, g) => sum + g.total, 0);

  return (
    <div className="max-w-[85%] text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-text-muted transition-colors hover:text-text"
      >
        <span>依據 {total} 則論壇評論</span>
        <svg
          width="10"
          height="10"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`transition-transform ${open ? "rotate-180" : ""}`}
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div className="mt-2 flex flex-col gap-3">
          {groups.map((g) => (
            <div key={g.model} className="flex flex-col gap-1.5">
              <p className="text-text-dim">
                {g.model}
                <span className="ml-1.5">
                  節錄 {g.comments.length} / {g.total} 則
                </span>
              </p>
              {g.comments.map((c, i) => (
                <SourceCard key={i} comment={c} />
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SourceCard({ comment }: { comment: SourceComment }) {
  // 灰階設計系統不用紅綠：負評用黑底反白凸顯，正評用外框，中立不加框只用暗色文字。
  const labelStyle =
    comment.label === "負評"
      ? "bg-accent text-accent-inverse"
      : comment.label === "正評"
        ? "border border-border-strong text-text"
        : "text-text-dim";

  return (
    <a
      href={comment.url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex flex-col gap-1 rounded-md border border-border p-2 transition-colors hover:border-border-strong"
    >
      <div className="flex items-center gap-1.5 text-text-dim">
        <span className={`rounded-sm px-1 py-px ${labelStyle}`}>{comment.label ?? "未分類"}</span>
        <span>{SOURCE_LABEL[comment.source]}</span>
        {comment.date && <span>{comment.date}</span>}
      </div>
      <p className="line-clamp-3 leading-relaxed text-text-muted">{comment.content}</p>
    </a>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1">
      <Dot delay="0ms" />
      <Dot delay="150ms" />
      <Dot delay="300ms" />
    </span>
  );
}

function Dot({ delay }: { delay: string }) {
  return (
    <span
      className="h-1.5 w-1.5 animate-bounce rounded-full bg-text-dim"
      style={{ animationDelay: delay }}
    />
  );
}
