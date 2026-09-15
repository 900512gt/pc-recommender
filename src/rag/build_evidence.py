"""
build_evidence.py
把 rag_chunks_v2.jsonl 的 evidence_ids 還原成原始評論，輸出精簡查表檔
data/rag_evidence.jsonl，給 server.py 的「評論佐證面板」在執行期查詢。

為什麼需要這一步：原始評論散在 src/database/input/ 底下（PTT 8MB + 巴哈 36MB），
而這兩個目錄都被 .dockerignore 排除在 API image 之外。chunk 實際引用到的評論只有
一萬多則，先離線抽出來存成一份小檔，容器裡就只要帶這份，不用打包整包原始資料。

evidence_id 的組成（實測兩邊都 100% 對得起來）：
  PTT   → f"{url}#{_index}"   _index 是 filtered_comments.jsonl 的全域列號
  巴哈  → f"{url}#{floor}"    floor 形如 "B3"，主文為 "1"

使用方式（從專案根目錄執行）：
  python src/rag/build_evidence.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CHUNKS_FILE = ROOT / "data" / "rag_chunks_v2.jsonl"
PTT_FILE = ROOT / "src" / "database" / "input" / "ptt_comment" / "relevant.jsonl"
BAHA_FILES = [
    ROOT / "src" / "database" / "input" / "baha_comment" / f"matched_part{i}.jsonl"
    for i in (1, 2, 3)
]
OUT_FILE = ROOT / "data" / "rag_evidence.jsonl"

# 標註資料裡混了簡體的「负評」，統一成繁體，避免前端要判斷兩種寫法
LABEL_FIX = {"负評": "負評"}


def _wanted_evidence_ids() -> set[str]:
    ids = set()
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            ids.update(json.loads(line).get("evidence_ids") or [])
    return ids


def _normalize_date(raw: str | None) -> str | None:
    """PTT 是 MM/DD/YYYY、巴哈是 YYYY-MM-DD HH:MM:SS，統一成 YYYY-MM-DD。
    PTT 有少數幾筆日期是 '????'，這種直接回 None 讓前端不顯示日期。"""
    if not raw:
        return None
    raw = raw.strip()
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            mm, dd, yyyy = parts
            return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}"
        return None
    head = raw.split(" ")[0]
    return head if head[:4].isdigit() else None


def _record(row: dict, source: str) -> dict:
    """只留前端顯示需要的欄位。原始 row 還有 brand/series/relevant 等，用不到就不帶，
    這是這份檔案能從 44MB 縮到幾 MB 的主因。"""
    label = row.get("label")
    out = {
        "source": source,
        "model": row.get("model"),
        "url": row.get("url"),
        "title": row.get("title"),
        "content": row.get("content"),
        "date": _normalize_date(row.get("date")),
        "label": LABEL_FIX.get(label, label),
    }
    if source == "ptt":
        # 推/噓/→，PTT 特有的推文態度標記
        out["tag"] = row.get("tag")
    else:
        out["floor"] = row.get("floor")
        out["author"] = row.get("author")
    return out


def main() -> None:
    wanted = _wanted_evidence_ids()
    print(f"chunk 共引用 {len(wanted)} 則不重複評論")

    found: dict[str, dict] = {}

    with PTT_FILE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = f"{row['url']}#{row['_index']}"
            if key in wanted:
                found[key] = _record(row, "ptt")
    print(f"  PTT 命中 {len(found)}")

    before = len(found)
    for path in BAHA_FILES:
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = f"{row['url']}#{row.get('floor')}"
                if key in wanted:
                    found[key] = _record(row, "bahamut")
    print(f"  巴哈命中 {len(found) - before}")

    missing = wanted - found.keys()
    if missing:
        print(f"⚠️  有 {len(missing)} 則找不到原始評論，範例：{list(missing)[:3]}")

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for key, rec in found.items():
            f.write(json.dumps({"evidence_id": key, **rec}, ensure_ascii=False) + "\n")

    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    print(f"已寫入 {OUT_FILE.relative_to(ROOT)}：{len(found)} 筆，{size_mb:.1f} MB")


if __name__ == "__main__":
    main()
