const RAG_API_URL = process.env.NEXT_PUBLIC_RAG_API_URL ?? "http://localhost:8001";

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export class RagApiError extends Error {}

/** 回答依據的單則原始論壇評論。 */
export interface SourceComment {
  source: "ptt" | "bahamut";
  model: string;
  url: string;
  title: string;
  content: string;
  date: string | null;
  label: string;
  tag?: string;
  floor?: string;
  author?: string;
}

/** 佐證評論依型號分組。total 是該型號的全部佐證數，comments 只是其中的取樣。 */
export interface SourceGroup {
  model: string;
  total: number;
  comments: SourceComment[];
}

export type ChatEvent =
  | { type: "sources"; groups: SourceGroup[] }
  | { type: "text"; text: string };

export interface TimelineMonth {
  month: string;
  positive: number;
  negative: number;
  neutral: number;
}

/** 某型號最近一段時間的逐月口碑走勢。total 是窗口內的評論數，不是歷史總數。 */
export interface Timeline {
  model: string;
  total: number;
  months: TimelineMonth[];
}

/**
 * 評論量不足以畫出有意義走勢的型號，後端會回 404——這是預期結果不是錯誤。
 * 走勢圖只是輔助資訊，任何失敗都回 null 讓畫面單純不顯示，不影響聊天。
 */
export async function fetchTimeline(model: string): Promise<Timeline | null> {
  try {
    const res = await fetch(`${RAG_API_URL}/api/model/${encodeURIComponent(model)}/timeline`);
    return res.ok ? ((await res.json()) as Timeline) : null;
  } catch {
    return null;
  }
}

/**
 * POST /api/chat 是 SSE 串流：先來一個 sources event（回答依據的原始評論），
 * 之後是一連串 text event。text 帶的是累積文字（不是 delta），呼叫端直接拿來
 * 覆蓋畫面上的內容即可。
 */
export async function* streamChat(
  query: string,
  history: ChatMessage[],
): AsyncGenerator<ChatEvent, void, unknown> {
  let res: Response;
  try {
    res = await fetch(`${RAG_API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history }),
    });
  } catch {
    throw new RagApiError("無法連線到聊天伺服器，請確認後端是否已啟動");
  }

  if (!res.ok || !res.body) {
    throw new RagApiError(`伺服器錯誤（${res.status}）`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (raw === "[DONE]") return;

      let parsed: { text?: string; error?: string; sources?: SourceGroup[] };
      try {
        parsed = JSON.parse(raw);
      } catch {
        continue;
      }

      if (parsed.error) throw new RagApiError(parsed.error);
      if (parsed.sources !== undefined) yield { type: "sources", groups: parsed.sources };
      if (parsed.text !== undefined) yield { type: "text", text: parsed.text };
    }
  }
}
