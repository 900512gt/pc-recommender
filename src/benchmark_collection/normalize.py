"""
Chip name normalization for PassMark lookup.

The same function is reused in:
  - extract_unique_chips.py (step 1: list chips to query)
  - merge_benchmark.py      (step 4: broadcast benchmarks back to DB)

Reusing one function guarantees that what you query is what gets matched.
"""

import re


def normalize_cpu_chip(ptt_model: str) -> str:
    """
    CPU chip names in this DB (e.g. 'AMD R7 7800X3D', 'Intel i5-12400',
    'Intel Core Ultra 7 265K') are already clean enough to use as-is.
    We just trim whitespace.
    """
    return ptt_model.strip()


def normalize_gpu_chip(gpu_chip: str) -> str:
    """
    GPU chip names need cleanup:
      - Remove memory-type annotations like '(GDDR6)' or '(GDDR7)'
      - Keep capacity annotations (e.g. '8G', '16G', '-6G') as they
        represent genuinely different SKUs with different benchmarks.

    Examples:
      'AMD Radeon RX 9070 XT(GDDR6)' -> 'AMD Radeon RX 9070 XT'
      'AMD Radeon RX 9070 XT'        -> 'AMD Radeon RX 9070 XT'   (merged)
      'RTX 5060 Ti 8G(GDDR7)'        -> 'RTX 5060 Ti 8G'
      'RTX 5060 Ti 16G(GDDR7)'       -> 'RTX 5060 Ti 16G'         (not merged)
      'RTX 3050'                     -> 'RTX 3050'                (8G variant)
      'RTX 3050-6G'                  -> 'RTX 3050-6G'             (6G variant)
    """
    s = gpu_chip.strip()
    # Strip "(GDDRx)" or "(GDDRx ...)" annotations
    s = re.sub(r'\s*\([^)]*GDDR[^)]*\)', '', s, flags=re.IGNORECASE)
    return s.strip()


def passmark_query_cpu(ptt_model: str) -> str:
    """
    Suggest a search query string for PassMark CPU search.
    PassMark uses full marketing names, so we expand abbreviations:
      'AMD R7 7800X3D'  -> 'AMD Ryzen 7 7800X3D'
      'Intel i5-12400'  -> 'Intel Core i5-12400'
      'Intel Core Ultra 7 265K' -> 'Intel Core Ultra 7 265K'  (already full)
    """
    s = ptt_model.strip()
    # AMD R5/R7/R9 -> AMD Ryzen 5/7/9
    s = re.sub(r'^AMD\s+R(\d)\s+', r'AMD Ryzen \1 ', s)
    # Intel i3/i5/i7/i9 -> Intel Core i3/i5/i7/i9
    s = re.sub(r'^Intel\s+i(\d)', r'Intel Core i\1', s)
    return s


def passmark_query_gpu(gpu_chip_normalized: str) -> str:
    """
    Suggest a search query string for PassMark GPU search.
      'AMD Radeon RX 9070 XT' -> 'AMD Radeon RX 9070 XT'  (already full)
      'RTX 5080 16G'          -> 'GeForce RTX 5080'
      'RTX 5060 Ti 8G'        -> 'GeForce RTX 5060 Ti 8GB'
      'RTX 3050-6G'           -> 'GeForce RTX 3050 6GB'
      'GT 1030'               -> 'GeForce GT 1030'
      'Intel Arc B580'        -> 'Intel Arc B580'
    """
    s = gpu_chip_normalized.strip()

    # AMD / Intel: keep as-is (already in standard form)
    if s.startswith('AMD') or s.startswith('Intel'):
        return s

    # NVIDIA: prepend 'GeForce' if missing
    # Convert capacity suffix '-6G' or trailing ' 8G' to ' 6GB' / ' 8GB'
    s = re.sub(r'-(\d+)G\b', r' \1GB', s)        # 'RTX 3050-6G' -> 'RTX 3050 6GB'
    s = re.sub(r'\s(\d+)G\b', r' \1GB', s)       # 'RTX 5080 16G' -> 'RTX 5080 16GB'

    if s.startswith('RTX') or s.startswith('GTX') or s.startswith('GT'):
        s = 'GeForce ' + s
    return s


if __name__ == '__main__':
    # Quick self-test
    cpu_cases = [
        ('AMD R7 7800X3D',         'AMD R7 7800X3D',         'AMD Ryzen 7 7800X3D'),
        ('Intel i5-12400',         'Intel i5-12400',         'Intel Core i5-12400'),
        ('Intel Core Ultra 7 265K','Intel Core Ultra 7 265K','Intel Core Ultra 7 265K'),
    ]
    gpu_cases = [
        ('AMD Radeon RX 9070 XT(GDDR6)', 'AMD Radeon RX 9070 XT',       'AMD Radeon RX 9070 XT'),
        ('AMD Radeon RX 9070 XT',        'AMD Radeon RX 9070 XT',       'AMD Radeon RX 9070 XT'),
        ('RTX 5080 16G(GDDR7)',          'RTX 5080 16G',                'GeForce RTX 5080 16GB'),
        ('RTX 5060 Ti 8G(GDDR7)',        'RTX 5060 Ti 8G',              'GeForce RTX 5060 Ti 8GB'),
        ('RTX 5060 Ti 16G(GDDR7)',       'RTX 5060 Ti 16G',             'GeForce RTX 5060 Ti 16GB'),
        ('RTX 3050',                     'RTX 3050',                    'GeForce RTX 3050'),
        ('RTX 3050-6G',                  'RTX 3050-6G',                 'GeForce RTX 3050 6GB'),
        ('GT 1030',                      'GT 1030',                     'GeForce GT 1030'),
        ('Intel Arc B580',               'Intel Arc B580',              'Intel Arc B580'),
    ]
    print('CPU normalize / passmark_query:')
    for raw, exp_norm, exp_query in cpu_cases:
        n = normalize_cpu_chip(raw)
        q = passmark_query_cpu(raw)
        ok = (n == exp_norm) and (q == exp_query)
        print(f'  {"OK" if ok else "FAIL"}  {raw!r:<28} -> norm={n!r:<28} query={q!r}')

    print('\nGPU normalize / passmark_query:')
    for raw, exp_norm, exp_query in gpu_cases:
        n = normalize_gpu_chip(raw)
        q = passmark_query_gpu(n)
        ok = (n == exp_norm) and (q == exp_query)
        print(f'  {"OK" if ok else "FAIL"}  {raw!r:<35} -> norm={n!r:<28} query={q!r}')