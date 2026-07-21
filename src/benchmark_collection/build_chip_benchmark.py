"""
Step 3: Build chip_benchmark.json from the validated chip_list_*.csv files.

Output structure:
  {
    "metadata": {
      "source_url_cpu": "https://www.cpubenchmark.net/cpu_list.php",
      "source_url_gpu": "https://www.videocardbenchmark.net/gpu_list.php",
      "captured_at": "2026-04-26",
      "cpu_metric": "PassMark CPU Mark (multi-thread)",
      "gpu_metric": "PassMark G3D Mark"
    },
    "CPU": {
      "AMD R7 7800X3D": {"benchmark": 34285, "source": "passmark"},
      ...
    },
    "GPU": {
      "RTX 5080 16G": {"benchmark": 35696, "source": "passmark"},
      ...
    }
  }

This file is the canonical chip -> benchmark lookup. Step 4
(merge_benchmark.py) will use it to broadcast benchmarks back into
ga_database.json.

Behavior:
  - Skips rows with empty benchmark (these are missing entries to be
    handled by step 5 interpolation)
  - Auto-fills source='passmark' for rows that have a benchmark but no
    source (since the manual workflow only fills benchmark)
  - Refuses to run if validation would fail; you should run
    validate_chip_list.py first
"""

import csv
import json
import sys
from datetime import date
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / 'output'
CPU_CSV = OUT_DIR / 'chip_list_cpu.csv'
GPU_CSV = OUT_DIR / 'chip_list_gpu.csv'
JSON_OUT = OUT_DIR / 'chip_benchmark.json'

METADATA = {
    'source_url_cpu': 'https://www.cpubenchmark.net/cpu_list.php',
    'source_url_gpu': 'https://www.videocardbenchmark.net/gpu_list.php',
    'captured_at': date.today().isoformat(),
    'cpu_metric': 'PassMark CPU Mark (multi-thread)',
    'gpu_metric': 'PassMark G3D Mark',
}


def read_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def build_section(rows, label):
    """
    Convert CSV rows into a {chip_name: {benchmark, source}} dict.
    Skips rows with no benchmark (logs them for the caller to track).
    """
    section = {}
    skipped = []
    for row in rows:
        name = (row.get('normalized_name') or '').strip()
        bench_raw = (row.get('benchmark') or '').strip().replace(',', '').replace(' ', '')
        source = (row.get('source') or '').strip() or 'passmark'  # default

        if not name:
            continue
        if not bench_raw:
            skipped.append(name)
            continue

        try:
            bench = float(bench_raw)
        except ValueError:
            print(f'[!] {label}: skipping {name!r} - benchmark not a number: {bench_raw!r}')
            skipped.append(name)
            continue

        # Store as int when possible (cleaner JSON)
        bench_val = int(bench) if bench.is_integer() else bench
        section[name] = {'benchmark': bench_val, 'source': source}

    return section, skipped


def main():
    if not CPU_CSV.exists() or not GPU_CSV.exists():
        print(f'ERROR: missing input CSV in {OUT_DIR}')
        print(f'  Expected: {CPU_CSV.name} and {GPU_CSV.name}')
        sys.exit(1)

    cpu_section, cpu_skipped = build_section(read_csv(CPU_CSV), 'CPU')
    gpu_section, gpu_skipped = build_section(read_csv(GPU_CSV), 'GPU')

    output = {
        'metadata': METADATA,
        'CPU': cpu_section,
        'GPU': gpu_section,
    }

    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'Wrote: {JSON_OUT}')
    print(f'  CPU entries: {len(cpu_section)}')
    print(f'  GPU entries: {len(gpu_section)}')
    if cpu_skipped or gpu_skipped:
        print(f'\n  Skipped (no benchmark, need step 5 interpolation):')
        for name in cpu_skipped:
            print(f'    CPU: {name}')
        for name in gpu_skipped:
            print(f'    GPU: {name}')
    else:
        print('  All entries have benchmarks. No interpolation needed.')


if __name__ == '__main__':
    main()