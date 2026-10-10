const GA_API_URL = process.env.NEXT_PUBLIC_GA_API_URL ?? "http://localhost:8000";

export type Usage = "工作" | "遊戲";
export type CoolingPreference = "auto" | "風冷" | "水冷";

/**
 * 效能／口碑／CP 值三個滑桿，各 0~100。是三者之間的相對比重，不是百分比：
 * 後端會照比例重新分配進該用途原本「效能＋口碑＋CP值」的總份額，
 * 預算與相容性的權重不受影響。
 */
export interface PreferenceWeights {
  perf: number;
  sent: number;
  cp: number;
}

export interface RecommendRequest {
  budget: number;
  usage: Usage;
  cooling_prefer: CoolingPreference;
  /**
   * 使用者沒調整過就不要送。只有「沒收到 weights」後端才會完整沿用該用途的預設比重；
   * 送 50/50/50 會被換算成三者等重，把遊戲偏重效能、工作偏重 CPU 這類用途差異抹掉。
   */
  weights?: PreferenceWeights;
  /** 使用者在意的口碑面向，沒選就不要送。只能送 fetchSelectableAspects() 列出的面向。 */
  aspect_prefs?: AspectPrefs;
}

/** 類別 → 在意的口碑面向，例如 { GPU: ["穩定"] }。 */
export type AspectPrefs = Record<string, string[]>;

/** 一個類別裡評論資料夠、可以讓使用者指定的口碑面向。 */
export interface SelectableAspects {
  category: string;
  aspects: string[];
}

/** 零件在一個口碑面向上的原始依據，用來核對「指定的面向是否真的比較好」。 */
export interface AspectEvidence {
  aspect: string;
  positive: number;
  negative: number;
  /** 面向口碑分數 [0,1]，已依評價則數往類別平均修正。 */
  score: number;
  category_avg: number;
  /** 在同類別、該面向評價夠多的型號之間的名次。評價不足時是 null，不可硬排。 */
  rank: number | null;
  ranked_total: number;
  /** 使用者是否指定了這個面向。 */
  picked: boolean;
}

export interface PartSentiment {
  /** 對應到的評論型號，同時是 /parts/[model] 的路徑參數。對不到任何評論時是 null。 */
  model: string | null;
  aspects: AspectEvidence[];
}

export interface RecommendedPart {
  category: string;
  name: string;
  price: number;
  score: number;
  /** 只有評論資料夠的類別（CPU、GPU、主機板）才有，其餘類別是 null。 */
  sentiment: PartSentiment | null;
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

/**
 * 一個加價級距的升級方案。extra_ratio 0 代表只花沒用完的剩餘預算，
 * 0.1 / 0.2 代表願意在原預算上多花 10% / 20%。
 * 該級距完全換不到更好的零件時後端不會回傳，所以陣列可能只有一兩項。
 */
export interface UpgradeTier {
  extra_budget: number;
  extra_ratio: number;
  available: number;
  spent: number;
  new_total_price: number;
  upgrades: Upgrade[];
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
  upgrade_tiers: UpgradeTier[];
  /** 實際套用到 fitness 的權重（滑桿換算後的結果），五項加總為 1。 */
  resolved_weights: {
    perf: number;
    sent: number;
    cp: number;
    budget: number;
    compat: number;
  };
  /** 實際套用的面向偏好。請求裡資料不夠的面向會被後端濾掉，所以可能比送出的少。 */
  applied_aspect_prefs: AspectPrefs;
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

/**
 * 各類別可指定的口碑面向。清單由後端依口碑資料決定，重建資料後可能會變。
 * 這是選填的進階選項，拿不到就回傳空陣列讓表單照常運作，不擋住主要流程。
 */
export async function fetchSelectableAspects(): Promise<SelectableAspects[]> {
  try {
    const res = await fetch(`${GA_API_URL}/api/aspects`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.categories ?? [];
  } catch {
    return [];
  }
}
