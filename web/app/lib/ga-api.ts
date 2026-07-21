const GA_API_URL = process.env.NEXT_PUBLIC_GA_API_URL ?? "http://localhost:8000";

export type Usage = "工作" | "遊戲";
export type CoolingPreference = "auto" | "風冷" | "水冷";

export interface RecommendRequest {
  budget: number;
  usage: Usage;
  cooling_prefer: CoolingPreference;
}

export interface RecommendedPart {
  category: string;
  name: string;
  price: number;
  score: number;
}

export interface Upgrade {
  priority: number;
  category: string;
  reason: string;
  current_name: string;
  current_price: number;
  upgrade_name: string;
  upgrade_price: number;
  cost: number;
  benefit: string;
  sentiment_improvement: number;
}

export interface RecommendResponse {
  parts: RecommendedPart[];
  total_price: number;
  budget: number;
  remaining: number;
  compatibility: {
    ok: boolean;
    penalty: number;
    issues: string[];
  };
  upgrades: Upgrade[];
}

export class GaApiError extends Error {}

export async function recommend(req: RecommendRequest): Promise<RecommendResponse> {
  let res: Response;
  try {
    res = await fetch(`${GA_API_URL}/api/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch {
    throw new GaApiError("無法連線到配置伺服器，請確認後端是否已啟動");
  }

  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new GaApiError(detail?.detail ?? `伺服器錯誤（${res.status}）`);
  }

  return res.json();
}
