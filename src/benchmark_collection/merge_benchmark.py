"""
Step 4: Broadcast benchmarks from chip_benchmark.json into ga_database.json,
producing ga_database_v2.json.

Pipeline:
  ga_database.json + chip_benchmark.json -> ga_database_v2.json

For every CPU and GPU item in the database, this script:
  1. Reads the raw chip identifier (CPU.ptt_model / GPU.gpu_chip)
  2. Normalizes it via normalize.py (same function step 1 used)
  3. Looks up chip_benchmark.json with the normalized name as key
  4. Adds three fields to the item:
       benchmark         : the score (or None if not found)
       benchmark_source  : 'chip_mapped' / 'missing' (step 5 may produce
                           'interpolated' / 'regressed' later)
       benchmark_chip    : the normalized chip name used as JOIN key
                           (kept for debug / auditability)

All other categories (RAM, MB, SSD, HDD, AIR_COOLER, WATER_COOLER, CASE,
PSU) are passed through unchanged.

The script never modifies the input v1 file. If anything looks wrong,
delete the v2 output and re-run.
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Make sibling modules importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize_cpu_chip, normalize_gpu_chip

ROOT       = Path(__file__).resolve().parent           # benchmark_collection/
DATA_DIR   = ROOT.parent.parent / 'data'               # project_root/data/

DB_IN_PATH        = DATA_DIR / 'ga_database.json'
DB_OUT_PATH       = DATA_DIR / 'ga_database_v2.json'
CHIP_BENCH_PATH   = DATA_DIR / 'raw' / 'chip_benchmark.json'


def annotate_item(item, raw_chip, normalize_fn, lookup_table):
    """
    Add benchmark fields to a single CPU or GPU item, in place.
    Returns the source tag that was assigned ('chip_mapped' or 'missing').
    """
    norm = normalize_fn(raw_chip)
    entry = lookup_table.get(norm)

    if entry is not None:
        item['benchmark'] = entry['benchmark']
        item['benchmark_source'] = 'chip_mapped'
        item['benchmark_chip'] = norm
        return 'chip_mapped'
    else:
        item['benchmark'] = None
        item['benchmark_source'] = 'missing'
        item['benchmark_chip'] = norm  # keep the attempted key for debugging
        return 'missing'


def process_category(db, category, raw_field, normalize_fn, lookup_table):
    """
    Annotate every item in db[category]. Returns (success_count,
    missing_list) where missing_list contains the unique normalized
    chip names that could not be looked up.
    """
    counter = Counter()
    missing_normalized = set()  # unique missing chips (not raw items)

    for item in db[category]:
        raw_chip = item[raw_field]
        tag = annotate_item(item, raw_chip, normalize_fn, lookup_table)
        counter[tag] += 1
        if tag == 'missing':
            missing_normalized.add(item['benchmark_chip'])

    return counter, missing_normalized


def main():
    # --- Load inputs ---
    if not DB_IN_PATH.exists():
        print(f'ERROR: {DB_IN_PATH} not found')
        sys.exit(1)
    if not CHIP_BENCH_PATH.exists():
        print(f'ERROR: {CHIP_BENCH_PATH} not found (run build_chip_benchmark.py first)')
        sys.exit(1)

    with open(DB_IN_PATH, encoding='utf-8') as f:
        db = json.load(f)
    with open(CHIP_BENCH_PATH, encoding='utf-8') as f:
        chip_bench = json.load(f)

    cpu_lookup = chip_bench['CPU']
    gpu_lookup = chip_bench['GPU']

    print(f'Loaded ga_database.json: {sum(len(v) for v in db.values())} items across {len(db)} categories')
    print(f'Loaded chip_benchmark.json: {len(cpu_lookup)} CPU chips + {len(gpu_lookup)} GPU chips')
    print()

    # --- Annotate CPU and GPU ---
    cpu_counter, cpu_missing = process_category(db, 'CPU', 'ptt_model',  normalize_cpu_chip, cpu_lookup)
    gpu_counter, gpu_missing = process_category(db, 'GPU', 'gpu_chip',   normalize_gpu_chip, gpu_lookup)

    # --- Report ---
    print('=== CPU ===')
    print(f'  Total items: {len(db["CPU"])}')
    print(f'  chip_mapped: {cpu_counter["chip_mapped"]}')
    print(f'  missing:     {cpu_counter["missing"]}')
    if cpu_missing:
        print('  Unmapped chips (sorted):')
        for name in sorted(cpu_missing):
            print(f'    - {name!r}')

    print('\n=== GPU ===')
    print(f'  Total items: {len(db["GPU"])}')
    print(f'  chip_mapped: {gpu_counter["chip_mapped"]}')
    print(f'  missing:     {gpu_counter["missing"]}')
    if gpu_missing:
        print('  Unmapped chips (sorted):')
        for name in sorted(gpu_missing):
            print(f'    - {name!r}')

    # --- Write output ---
    with open(DB_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f'\nWrote: {DB_OUT_PATH}')

    # --- Final advice ---
    total_missing = cpu_counter['missing'] + gpu_counter['missing']
    if total_missing == 0:
        print('All CPU/GPU items have benchmarks. v2 is ready for the GA / NSGA-II.')
    else:
        print(f'{total_missing} items remain without a benchmark.')
        print('  -> These are candidates for step 5 (interpolation / regression).')
        print('  -> The unique chips needing attention are listed above.')


if __name__ == '__main__':
    main()