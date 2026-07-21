"""
Step 2: Validate the manually-filled chip_list_*.csv files.

What it checks:
  - Required columns are present
  - Every row has a non-empty `normalized_name`
  - `benchmark` is a positive number (no commas, no junk)
  - `benchmark` falls inside a reasonable range for CPU/GPU
  - `source` is either empty or one of the allowed values
  - Reports rows that are missing `benchmark` (these go to step 5: interpolation)

What it does NOT do:
  - Modify the CSVs (read-only validation)
  - Build chip_benchmark.json (that's the next script)

Usage:
  python validate_chip_list.py
"""

import csv
import sys
from pathlib import Path
from statistics import median

OUT_DIR = Path(__file__).resolve().parent / 'output'
CPU_CSV = OUT_DIR / 'chip_list_cpu.csv'
GPU_CSV = OUT_DIR / 'chip_list_gpu.csv'

# Reasonable ranges based on PassMark's actual distribution.
# Anything outside these ranges is almost certainly a typo or wrong column.
CPU_MARK_RANGE = (500, 150_000)   # PassMark CPU Mark
G3D_MARK_RANGE = (100, 80_000)    # PassMark G3D Mark

ALLOWED_SOURCES = {'', 'passmark', 'interpolated', 'regressed', 'manual'}
REQUIRED_COLUMNS = ['normalized_name', 'count', 'benchmark', 'source']


class ValidationReport:
    def __init__(self, label):
        self.label = label
        self.errors = []     # blocking issues
        self.warnings = []   # non-blocking issues
        self.filled = 0
        self.missing = []    # rows without benchmark (need interpolation later)
        self.values = []     # for stats

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def print_report(self):
        print(f'\n=== {self.label} ===')
        if self.errors:
            print(f'  Errors ({len(self.errors)}):')
            for e in self.errors:
                print(f'    [X] {e}')
        if self.warnings:
            print(f'  Warnings ({len(self.warnings)}):')
            for w in self.warnings:
                print(f'    [!] {w}')

        if self.values:
            vals = sorted(self.values)
            print(f'  Filled rows: {self.filled}')
            print(f'  Benchmark stats: min={int(vals[0]):,}  median={int(median(vals)):,}  max={int(vals[-1]):,}')

        if self.missing:
            print(f'  Missing benchmarks ({len(self.missing)}) -- need step 5 interpolation:')
            for name in self.missing:
                print(f'    - {name}')

        if not self.errors and not self.missing:
            print('  All rows filled, all values valid.')


def read_csv(path):
    """Tolerant CSV reader: handles utf-8 with or without BOM."""
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def validate(rows, label, value_range):
    """Validate one CSV (CPU or GPU). Returns a ValidationReport."""
    rep = ValidationReport(label)

    # 1. Header check
    if not rows:
        rep.err(f'CSV is empty')
        return rep
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in rows[0].keys()]
    if missing_cols:
        rep.err(f'Missing required columns: {missing_cols}')
        return rep

    seen_names = set()
    lo, hi = value_range

    for i, row in enumerate(rows, start=2):  # start=2 because row 1 is header
        name = (row.get('normalized_name') or '').strip()
        bench_raw = (row.get('benchmark') or '').strip()
        source = (row.get('source') or '').strip()

        # Name checks
        if not name:
            rep.err(f'Row {i}: empty normalized_name')
            continue
        if name in seen_names:
            rep.err(f'Row {i}: duplicate normalized_name {name!r}')
        seen_names.add(name)

        # Source check
        if source not in ALLOWED_SOURCES:
            rep.warn(f'Row {i} ({name}): unknown source {source!r}, expected one of {sorted(ALLOWED_SOURCES - {""})}')

        # Benchmark check
        if not bench_raw:
            rep.missing.append(name)
            continue

        # Tolerate commas just in case (e.g. "9,188" pasted from PassMark)
        bench_clean = bench_raw.replace(',', '').replace(' ', '')
        try:
            bench = float(bench_clean)
        except ValueError:
            rep.err(f'Row {i} ({name}): benchmark {bench_raw!r} is not a number')
            continue

        if bench <= 0:
            rep.err(f'Row {i} ({name}): benchmark must be positive, got {bench}')
            continue

        if not (lo <= bench <= hi):
            rep.err(f'Row {i} ({name}): benchmark {bench:,.0f} outside reasonable range {lo:,}~{hi:,}')
            continue

        # Warn if user typed comma (we accepted it but they should fix it)
        if ',' in bench_raw:
            rep.warn(f'Row {i} ({name}): benchmark contains comma, please remove')

        rep.filled += 1
        rep.values.append(bench)

    return rep


def main():
    print(f'Reading from: {OUT_DIR}')

    if not CPU_CSV.exists():
        print(f'ERROR: {CPU_CSV} not found')
        sys.exit(1)
    if not GPU_CSV.exists():
        print(f'ERROR: {GPU_CSV} not found')
        sys.exit(1)

    cpu_rows = read_csv(CPU_CSV)
    gpu_rows = read_csv(GPU_CSV)

    cpu_rep = validate(cpu_rows, f'CPU ({CPU_CSV.name})', CPU_MARK_RANGE)
    gpu_rep = validate(gpu_rows, f'GPU ({GPU_CSV.name})', G3D_MARK_RANGE)

    cpu_rep.print_report()
    gpu_rep.print_report()

    # Final summary
    print('\n=== Summary ===')
    total = len(cpu_rows) + len(gpu_rows)
    filled = cpu_rep.filled + gpu_rep.filled
    missing = len(cpu_rep.missing) + len(gpu_rep.missing)
    errors = len(cpu_rep.errors) + len(gpu_rep.errors)
    print(f'  Total rows: {total}')
    print(f'  Filled:     {filled}')
    print(f'  Missing:    {missing}  (will need interpolation in step 5)')
    print(f'  Errors:     {errors}')

    if errors > 0:
        print('\n  -> Fix errors before proceeding to step 3.')
        sys.exit(1)
    elif missing > 0:
        print('\n  -> OK to proceed to step 3, but track the missing chips for step 5.')
    else:
        print('\n  -> All clear. Ready for step 3 (build_chip_benchmark.py).')


if __name__ == '__main__':
    main()