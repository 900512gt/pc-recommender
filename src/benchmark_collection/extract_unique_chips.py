"""
Step 1: Extract unique CPU/GPU chips from ga_database.json and emit two
CSV files for manual PassMark lookup.

Output CSVs (in benchmark_collection/):
  - chip_list_cpu.csv
  - chip_list_gpu.csv

Each row represents one unique chip you need to look up. After this script
runs, open the CSV files in Excel/Sheets and fill the 'benchmark' and
'source' columns by hand (or with a small fetch script).

Columns:
  normalized_name   : Canonical chip name (used as the JOIN key in step 4)
  count             : How many DB items use this chip (priority hint)
  passmark_query    : Suggested search string for PassMark
  passmark_url      : Leave blank, fill in the URL you used (audit trail)
  benchmark         : Leave blank, fill with the PassMark score
  source            : Leave blank, fill with 'passmark' / 'interpolated' / etc.
  notes             : Leave blank, optional remarks
  raw_chip_names    : All original DB strings that map to this normalized name
                      (useful for sanity-checking the merge)
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make `import normalize` work when run from anywhere
sys.path.insert(0, str(Path(__file__).parent))
from normalize import (
    normalize_cpu_chip,
    normalize_gpu_chip,
    passmark_query_cpu,
    passmark_query_gpu,
)

DB_PATH = Path(__file__).resolve().parents[2] / 'data' / 'ga_database.json'
OUT_DIR = Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

CSV_HEADER = [
    'normalized_name',
    'count',
    'passmark_query',
    'passmark_url',
    'benchmark',
    'source',
    'notes',
    'raw_chip_names',
]


def extract_cpu(db):
    """Return list of dicts, one per unique normalized CPU chip."""
    groups = defaultdict(lambda: {'count': 0, 'raws': set()})
    for item in db['CPU']:
        raw = item['ptt_model']
        norm = normalize_cpu_chip(raw)
        groups[norm]['count'] += 1
        groups[norm]['raws'].add(raw)

    rows = []
    for norm in sorted(groups.keys()):
        g = groups[norm]
        rows.append({
            'normalized_name': norm,
            'count': g['count'],
            'passmark_query': passmark_query_cpu(norm),
            'passmark_url': '',
            'benchmark': '',
            'source': '',
            'notes': '',
            'raw_chip_names': ' | '.join(sorted(g['raws'])),
        })
    return rows


def extract_gpu(db):
    """Return list of dicts, one per unique normalized GPU chip."""
    groups = defaultdict(lambda: {'count': 0, 'raws': set()})
    for item in db['GPU']:
        raw = item['gpu_chip']
        norm = normalize_gpu_chip(raw)
        groups[norm]['count'] += 1
        groups[norm]['raws'].add(raw)

    rows = []
    for norm in sorted(groups.keys()):
        g = groups[norm]
        rows.append({
            'normalized_name': norm,
            'count': g['count'],
            'passmark_query': passmark_query_gpu(norm),
            'passmark_url': '',
            'benchmark': '',
            'source': '',
            'notes': '',
            'raw_chip_names': ' | '.join(sorted(g['raws'])),
        })
    return rows


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        # 'utf-8-sig' adds a BOM so Excel opens the file with correct encoding
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with open(DB_PATH, encoding='utf-8') as f:
        db = json.load(f)

    cpu_rows = extract_cpu(db)
    gpu_rows = extract_gpu(db)

    cpu_csv = OUT_DIR / 'chip_list_cpu.csv'
    gpu_csv = OUT_DIR / 'chip_list_gpu.csv'
    write_csv(cpu_csv, cpu_rows)
    write_csv(gpu_csv, gpu_rows)

    # Console summary
    cpu_total_items = sum(r['count'] for r in cpu_rows)
    gpu_total_items = sum(r['count'] for r in gpu_rows)
    print(f'CPU: {len(cpu_rows)} unique chips covering {cpu_total_items} DB items')
    print(f'GPU: {len(gpu_rows)} unique chips covering {gpu_total_items} DB items')
    print(f'Total chips to look up: {len(cpu_rows) + len(gpu_rows)}')
    print()
    print(f'Wrote: {cpu_csv}')
    print(f'Wrote: {gpu_csv}')

    # Show how the GPU merge collapsed things
    print('\nGPU normalization merges (where multiple raw names mapped to one):')
    for r in gpu_rows:
        raws = r['raw_chip_names'].split(' | ')
        if len(raws) > 1:
            print(f'  {r["normalized_name"]!r}')
            for raw in raws:
                print(f'      <- {raw!r}')


if __name__ == '__main__':
    main()