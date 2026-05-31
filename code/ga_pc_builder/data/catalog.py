"""
零件目錄：從 ga_database_v2.json 載入所有零件規格與價格
"""
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

from config import CAT_MAP


@dataclass
class Part:
    category: str
    name: str
    price: int
    specs: dict = field(default_factory=dict)

    @property
    def short_name(self) -> str:
        return self.name


class PartCatalog:
    def __init__(self, db_path: Path):
        self.parts: dict[str, list[Part]] = defaultdict(list)
        self._load(db_path)

    def _load(self, path: Path):
        with open(path, encoding="utf-8") as f:
            db = json.load(f)

        for db_cat, cat in CAT_MAP.items():
            for item in db.get(db_cat, []):
                name = item.get("name", "").strip()
                price = int(item.get("price", 0) or 0)
                if not name or price <= 0:
                    continue
                # 把所有欄位都存進 specs 供相容性檢查使用
                specs = {k: v for k, v in item.items()
                         if k not in ("name", "price")}
                self.parts[cat].append(Part(cat, name, price, specs))

        for cat, lst in self.parts.items():
            print(f"[PartCatalog] {cat}: {len(lst)} 筆")

    def get(self, category: str) -> list[Part]:
        return self.parts.get(category, [])
