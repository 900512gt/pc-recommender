#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查数据库 schema 和可用的规格字段
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

with open(DB_PATH, 'r', encoding='utf-8') as f:
    db = json.load(f)

categories = ['CPU', 'GPU', '記憶體', 'SSD', '電源', '主機板']

for cat in categories:
    if cat in db and len(db[cat]) > 0:
        item = db[cat][0]
        print(f'\n{"="*70}')
        print(f'【{cat}】')
        print(f'{"="*70}')
        print(f'字段: {list(item.keys())}')
        print(f'\n示例值:')
        for key, val in item.items():
            val_str = str(val)[:50] if len(str(val)) > 50 else str(val)
            print(f'  {key:20s}: {val_str}')
        print(f'\n类别数量: {len(db[cat])} 件')
