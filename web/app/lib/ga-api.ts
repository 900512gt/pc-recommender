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
  current_name: string;
  current_price: number;
  upgrade_name: string;
  upgrade_price: number;
  cost: number;
  /** PassMark 跑分提升百分比。只有 CPU/GPU 有跑分資料，其餘類別是 null。 */
  benchmark_gain_pct: number | null;
  /** 論壇口碑變化量 [-1, 1]，與效能分開計算，不可混為一談。 */
  sentiment_delta: number;
  /** 實際變好的規格，例如「容量 8GB → 16GB」。 */
  spec_changes: string[];
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
