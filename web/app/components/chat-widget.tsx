"use client";

import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import Link from "next/link";
import {
  RagApiError,
  streamChat,
  type ChatMessage,
  type SourceComment,
  type SourceGroup,
} from "../lib/rag-api";

const EXAMPLES = [
  "RTX5070 值得買嗎？",
  "7800X3D 有什麼缺點？",
  "中階顯示卡推薦",
];

/** 畫面上的訊息比送回後端的 ChatMessage 多帶佐證，送出前必須剝掉（見 send()）。 */
type DisplayMessage = ChatMessage & { sources?: SourceGroup[] };

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
    setMessages((m) => [...m, { role: "user", content: q }, { role: "assistant", content: "" }]);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setBusy(true);

    try {
      for await (const event of streamChat(q, historySnapshot)) {
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
                {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                  <>
                    <ModelLinks models={m.sources.map((g) => g.model)} />
                    <SourcePanel groups={m.sources} />
                  </>
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

/**
 * 導向型號詳情頁。走勢圖與雷達圖刻意不畫在這裡——聊天視窗只有 384px 寬，
 * 18 根柱子的時間軸與五軸雷達圖在這個寬度下都看不清楚，那些留給有整頁空間的
 * /parts/[型號]。聊天室負責回答問題並指路。
 */
function ModelLinks({ models }: { models: string[] }) {
  if (models.length === 0) return null;
  return (
    <div className="flex max-w-[85%] flex-wrap gap-1.5">
      {models.map((model) => (
        <Link
          key={model}
          href={`/parts/${encodeURIComponent(model)}`}
          className="rounded-sm border border-border px-2 py-0.5 text-xs text-text-muted transition-colors hover:border-border-strong hover:text-text"
        >
          {model} 完整分析 →
        </Link>
      ))}
    </div>
  );
}

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
