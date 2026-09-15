"""
evidence.py
把檢索到的 chunk 還原成「這個回答依據了哪些原始評論」，給前端的佐證面板顯示。

讀的是 build_evidence.py 產生的 data/rag_evidence.jsonl（離線抽好的精簡查表檔），
不是 src/database/input/ 底下的原始評論——那兩包在 API image 裡不存在。
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EVIDENCE_FILE = ROOT / "data" / "rag_evidence.jsonl"

# 每個型號在面板上最多列幾則。chunk 的 evidence 動輒五六十則，全部塞給前端
# 既拖慢串流也沒人會看完；真實總數另外用 total 欄位誠實回報。
MAX_PER_MODEL = 8


class EvidenceStore:
    """evidence_id → 原始評論的查表。檔案不存在時不擋掉聊天功能，只是查不到佐證。"""

    def __init__(self, path: Path = EVIDENCE_FILE):
        self._by_id: dict[str, dict] = {}
        self.available = path.exists()
        if not self.available:
            print(f"[RAG] 找不到 {path.name}，佐證面板停用（跑 build_evidence.py 產生）")
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self._by_id[rec.pop("evidence_id")] = rec
        print(f"[RAG] 載入 {len(self._by_id)} 則原始評論供佐證面板查詢")

    def for_model(self, model: str, limit: int = 30) -> list[dict]:
        """某型號的佐證評論，給型號詳情頁用。

        聊天視窗只塞得下幾則，詳情頁有整頁空間，所以 limit 放寬。取樣一樣是正負評
        交錯，避免整頁都是同一面倒的意見。
        """
        comments = [rec for rec in self._by_id.values() if rec.get("model") == model]
        return _pick(comments, limit)

    def collect(self, chunks: list[dict]) -> list[dict]:
        """依型號分組回傳佐證評論。

        同一個型號的 summary / aspect / pros_cons chunk 會引用到重疊的評論，
        所以要先跨 chunk 去重再取樣。
        """
        if not self._by_id:
            return []

        ids_by_model: dict[str, list[str]] = {}
        for chunk in chunks:
            model = chunk.get("model")
            if not model:
                continue
            seen = ids_by_model.setdefault(model, [])
            for eid in chunk.get("evidence_ids") or []:
                if eid not in seen:
                    seen.append(eid)

        groups = []
        for model, ids in ids_by_model.items():
            comments = [self._by_id[i] for i in ids if i in self._by_id]
            if not comments:
                continue
            groups.append({
                "model": model,
                "total": len(comments),
                "comments": _pick(comments, MAX_PER_MODEL),
            })
        return groups


def _pick(comments: list[dict], limit: int) -> list[dict]:
    """挑要顯示的評論：正負評交錯（讓使用者兩面都看得到，不會只看到一面倒的樣本），
    各自新的排前面，額度有剩才補中立評論。"""
    def newest_first(label: str) -> list[dict]:
        got = [c for c in comments if c.get("label") == label]
        return sorted(got, key=lambda c: c.get("date") or "", reverse=True)

    negative, positive = newest_first("負評"), newest_first("正評")

    picked = []
    for i in range(max(len(negative), len(positive))):
        if len(picked) >= limit:
            break
        if i < len(negative):
            picked.append(negative[i])
        if i < len(positive) and len(picked) < limit:
            picked.append(positive[i])

    if len(picked) < limit:
        picked.extend(newest_first("中立")[: limit - len(picked)])
    return picked


_store: EvidenceStore | None = None


def get_store() -> EvidenceStore:
    global _store
    if _store is None:
        _store = EvidenceStore()
    return _store
