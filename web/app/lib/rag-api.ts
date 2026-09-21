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

export interface Aspect {
  name: string;
  score: number;
  positive: number;
  negative: number;
  samples: number;
  /** 樣本數足夠、分數才可信。不足時 score 仍有值但不該拿來畫圖。 */
  sufficient: boolean;
}

export interface ModelAspects {
  model: string;
  review_count: number;
  aspects: Aspect[];
}

/**
 * 五大面向口碑分數（效能／溫控／噪音／保固／CP值）。
 * 可信面向不到三個的型號後端會回 404——三個軸才畫得出多邊形。
 * 雷達圖只是輔助資訊，失敗一律回 null 讓畫面單純不顯示。
 */
export async function fetchAspects(model: string): Promise<ModelAspects | null> {
  try {
    const res = await fetch(`${RAG_API_URL}/api/model/${encodeURIComponent(model)}/aspects`);
    return res.ok ? ((await res.json()) as ModelAspects) : null;
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

// ── 型號頁 ────────────────────────────────────────────────────────────────────

export interface ModelIndexRow {
  model: string;
  category: string;
  review_count: number;
  label_distribution: Record<string, number>;
  confidence: string | null;
  price: number | null;
  benchmark: number | null;
}

export interface ModelDetail {
  model: string;
  category: string;
  summary: string | null;
  confidence: string | null;
  review_count: number;
  label_distribution: Record<string, number>;
  source_distribution: Record<string, number>;
  data_start_date: string | null;
  data_end_date: string | null;
  aspects: { aspect: string; text: string }[];
  pros_cons: string[];
  comparisons: string[];
  /**
   * 不在原價屋報價單快照裡的型號會是 null。這不代表停產（快照裡連 RTX40 系列都沒有），
   * 只代表這份報價單上查不到，所以畫面顯示「—」而不是「已停產」。
   */
  listing: {
    name: string | null;
    price: number | null;
    benchmark: number | null;
    brand: string | null;
  } | null;
}

async function getJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${RAG_API_URL}${path}`);
    return res.ok ? ((await res.json()) as T) : null;
  } catch {
    return null;
  }
}

export async function fetchModelIndex(): Promise<ModelIndexRow[]> {
  const data = await getJson<{ models: ModelIndexRow[] }>("/api/models");
  return data?.models ?? [];
}

export function fetchModelDetail(model: string): Promise<ModelDetail | null> {
  return getJson<ModelDetail>(`/api/model/${encodeURIComponent(model)}`);
}

export async function fetchModelComments(model: string): Promise<SourceComment[]> {
  const data = await getJson<{ comments: SourceComment[] }>(
    `/api/model/${encodeURIComponent(model)}/comments`,
  );
  return data?.comments ?? [];
}
