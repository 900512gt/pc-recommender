const RAG_API_URL = process.env.NEXT_PUBLIC_RAG_API_URL ?? "http://localhost:8001";

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export class RagApiError extends Error {}

/**
 * POST /api/chat 是 SSE 串流：每個 event 帶累積文字（不是 delta）。
 * yield 的也是累積後的完整字串，呼叫端直接拿來覆蓋畫面上的內容即可。
 */
export async function* streamChat(
  query: string,
  history: ChatMessage[],
): AsyncGenerator<string, void, unknown> {
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

      let parsed: { text?: string; error?: string };
      try {
        parsed = JSON.parse(raw);
      } catch {
        continue;
      }

      if (parsed.error) throw new RagApiError(parsed.error);
      if (parsed.text !== undefined) yield parsed.text;
    }
  }
}
