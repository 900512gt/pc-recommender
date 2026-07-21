"""
Broadcast TGP (Total Graphics Power) from gpu_tgp.json into ga_database_v2.json.

Pipeline:
  ga_database_v2.json + gpu_tgp.json  ->  ga_database_v2.json  (in-place update)

For every GPU item in the database, this script:
  1. Reads gpu_chip (e.g. "RTX 5080 16G(GDDR7)")
  2. Looks it up directly in gpu_tgp.json["GPU"] — no normalization needed,
     keys in the JSON were built to match gpu_chip verbatim
  3. Adds two fields to the item:
       tgp_watts   : int watts (or None if not found)
       tgp_source  : 'chip_mapped' / 'missing'

CPU items and all other categories are passed through unchanged.

The script loads v2, modifies it in memory, then writes back to the same file.
If anything looks wrong, restore from git: git checkout data/ga_database_v2.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT     = Path(__file__).resolve().parent           # benchmark_collection/
DATA_DIR = ROOT.parent.parent / 'data'               # project_root/data/
DB_PATH  = DATA_DIR / 'ga_database_v2.json'
TGP_PATH = DATA_DIR / 'raw' / 'gpu_tgp.json'


def main():
    # --- Load inputs ---
    if not DB_PATH.exists():
        print(f'ERROR: {DB_PATH} not found')
        sys.exit(1)
    if not TGP_PATH.exists():
        print(f'ERROR: {TGP_PATH} not found')
        sys.exit(1)

    with open(DB_PATH, encoding='utf-8') as f:
        db = json.load(f)
    with open(TGP_PATH, encoding='utf-8') as f:
        tgp_data = json.load(f)

    tgp_lookup = tgp_data['GPU']

    total_items = sum(len(v) for v in db.values())
    print(f'Loaded ga_database_v2.json: {total_items} items across {len(db)} categories')
    print(f'Loaded gpu_tgp.json: {len(tgp_lookup)} GPU chip entries')
    print()

    # --- Annotate GPU items ---
    counter = Counter()
    missing_chips = []

    for item in db.get('GPU', []):
        chip = item['gpu_chip']
        entry = tgp_lookup.get(chip)

        if entry is not None:
            item['tgp_watts'] = entry['tgp_watts']
            item['tgp_source'] = 'chip_mapped'
            counter['chip_mapped'] += 1
        else:
            item['tgp_watts'] = None
            item['tgp_source'] = 'missing'
            counter['missing'] += 1
            missing_chips.append(chip)

    # --- Report ---
    gpu_total = len(db.get('GPU', []))
    print('=== GPU TGP annotation ===')
    print(f'  Total GPU items : {gpu_total}')
    print(f'  chip_mapped     : {counter["chip_mapped"]}')
    print(f'  missing         : {counter["missing"]}')
    if missing_chips:
        print('  Unmapped gpu_chip values (sorted):')
        for name in sorted(set(missing_chips)):
            print(f'    - {name!r}')

    # --- Write back to v2 ---
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f'\nWrote: {DB_PATH}')

    if counter['missing'] == 0:
        print('All GPU items have tgp_watts. v2 is ready.')
    else:
        print(f'{counter["missing"]} items remain without tgp_watts.')
        print('  -> Add the missing chips to data/raw/gpu_tgp.json and re-run.')


if __name__ == '__main__':
    main()
