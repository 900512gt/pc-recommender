"""
comment_filter.py
PTT 硬體評論過濾腳本

輸入：../../data/ptt_comment.xlsx（專案根目錄的 data 資料夾）
輸出：
  - filtered_comments.jsonl  → 過濾後可用於 fine-tune 的評論
  - filter_report.xlsx       → 過濾統計報告

使用方式：
  python comment_filter.py                          # 自動讀 data/ptt_comment.xlsx
  python comment_filter.py /path/ptt_comment.xlsx   # 指定其他路徑

過濾規則：
  R1 - 純 URL（無其他文字）
  R2 - 長度少於 8 字
  R3 - 句子結尾是轉折詞（截斷句）
  R4 - 只有純表情/笑聲/感嘆詞
  R5 - 純回應性用語（推、+1、感謝等）
  R6 - 只有數字、標點、符號
  R7 - 含圖片連結且無其他實質文字
"""

import re
import json
import os
import sys
from collections import defaultdict
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── 設定 ──────────────────────────────────────────────

# 預設輸入路徑：專案根目錄的 data/ptt_comment.xlsx
# 結構：
#   計算機專題/
#   ├── data/
#   │   └── ptt_comment.xlsx     ← 輸入
#   └── src/
#       └── filter/
#           ├── comment_filter.py
#           └── output/
#               ├── filtered_comments.jsonl  ← 輸出
#               └── filter_report.xlsx       ← 輸出
DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # filter/
    '..', '..', 'data', 'ptt_comment.xlsx'       # ../../data/ptt_comment.xlsx
)

# 輸出檔案（輸出到腳本所在資料夾的 output/ 子資料夾）
OUTPUT_JSONL  = 'filtered_comments.jsonl'
OUTPUT_REPORT = 'filter_report.xlsx'

# GA 資料庫中有評論的型號（只保留這些型號的評論）
GA_MODELS = {
    'CPU': [
        'AMD R7 7800X3D','Intel i5-12400','AMD R9 9950X','AMD R9 9950X3D',
        'AMD R7 9700X','AMD R9 9900X','AMD R5 7500F','Intel i7-14700',
        'Intel Core Ultra 7 265K','AMD R5 9600X','Intel Core Ultra 9 285K',
        'AMD R5 3400G','AMD R5 8600G','AMD R5 5600XT','AMD R9 9900X3D',
        'AMD R7 8700G','Intel i3-14100','AMD R5 5500X3D','AMD R7 7700',
        'Intel Core Ultra 7 265KF','AMD R5 8400F','Intel Core Ultra 5 245K',
        'AMD R5 8500G','Intel Core Ultra 5 245KF','AMD R5 9500F',
        'Intel Core Ultra 5 225','Intel Core Ultra 5 235','Intel i5-14400',
        'Intel Core Ultra 5 225F','AMD R5 5600GT','AMD R5 5500GT','Intel i5-14400F',
    ],
    'GPU': [
        'RTX5080','RX9070XT','RTX5070Ti','RTX5060Ti','RX9060XT',
        'RTX5070','RX9070','RTX5060','RTX3050','Arc B580',
        'RTX5050','RX9070GRE','GT710','GT730','RX7650GRE','GT1030',
    ],
    'RAM': [
        'DDR4-3200','DDR4-3600','DDR5-6000','DDR5-6400',
        'DDR5-4800','DDR5-5600','DDR5-8000',
    ],
    'MB': [
        'Z790','Z890','X870E','A620','B550M','B650M','B850M',
        'B760M','X870','A520M','H610M','B840','B850','B860M','H810M',
    ],
    'SSD': [
        'Acer GM7000','WD SN850X','Micron T500','Kingston KC3000',
        'Micron T710','Kingston FURY Renegade','ADATA XPG S70','Kingston NV3',
        'WD SN8100','MSI SPATIUM M580','Micron T705','Micron P310',
        'Corsair MP600','Samsung 990 PRO','Micron T700','Samsung 870 EVO',
        'ADATA LEGEND 860','ADATA LEGEND 900','ZhiTai TiPlus','Samsung 9100 PRO',
    ],
    'HDD': ['Toshiba P300 2TB'],
    'AIR_COOLER': [
        'Noctua NH-D15','利民 Peerless Assassin 120','酷碼 Hyper 212',
        'Noctua NH-D15S','Noctua NH-U12S','Scythe 虎徹3','DEEPCOOL ASSASSIN IV',
        '利民 AXP90','利民 AXP120','Scythe 無限6','Noctua NH-L9a',
    ],
    'WATER_COOLER': [
        'Montech HyperFlow','華碩 ROG RYUO IV','華碩 ROG STRIX LC III',
        '華碩 TUF GAMING LC III','華碩 ProArt LC',
    ],
    'CASE': [
        '酷碼 NR200','酷碼 NR200P','Fractal Define 7','Montech Air 100',
        '聯力 LANCOOL 216','聯力 O11 Dynamic Mini V2','聯力 O11 Dynamic EVO',
        'NZXT H5 Flow','Montech SKY TWO','華碩 AP201','Montech SKY ONE',
        '華碩 ProArt PA401','HYTE Y70','NZXT H7 Flow','Montech Air 1000',
        'Fractal Torrent Compact','Montech X1','Montech X2 PLUS','Montech KING 95',
        'Fractal Meshify 3','NZXT H6 Flow','酷碼 Q300L','Fractal North XL',
        'NZXT H9 Flow','銀欣 SUGO 16','喬思伯 D300','喬思伯 D31',
        'HYTE X50','Montech KING 65',
    ],
    'PSU': [
        '振華 LEADEX VII','海韻 FOCUS GX','全漢 金鋼彈','全漢 聖武士',
        '海韻 VERTEX GX','振華 LEADEX III','華碩 ROG LOKI','全漢 VITA GM','海韻 CORE GX',
    ],
}

# 所有合法型號（扁平化）
ALL_VALID_MODELS = {m for models in GA_MODELS.values() for m in models}

# Sheet 名稱 → 類別代碼
SHEET_TO_CAT = {
    'CPU_評論':    'CPU',
    'GPU_評論':    'GPU',
    '記憶體_評論': 'RAM',
    '主機板_評論': 'MB',
    'SSD_評論':   'SSD',
    'HDD_評論':   'HDD',
    '風冷_評論':   'AIR_COOLER',
    '水冷_評論':   'WATER_COOLER',
    '機殼_評論':   'CASE',
    '電源_評論':   'PSU',
}

CAT_DISPLAY = {
    'CPU':'CPU','GPU':'GPU','RAM':'記憶體','MB':'主機板',
    'SSD':'SSD','HDD':'HDD','AIR_COOLER':'風冷',
    'WATER_COOLER':'水冷','CASE':'機殼','PSU':'電源',
}

# ── 過濾規則 ───────────────────────────────────────────

URL_RE         = re.compile(r'^(https?://\S+|www\.\S+|i\.imgur\.\S+)$')
URL_IN_TEXT    = re.compile(r'https?://\S+|www\.\S+')
TRAILING_BREAK = re.compile(
    r'(因為|但是|但|雖然|雖|所以|然後|而且|不過|如果|就是|只是|還有|'
    r'感覺|覺得|應該|可能|或是|或者|另外|其實|畢竟|除了|加上)\s*$'
)
# 開頭是標點符號 → 上一則評論的後半段（截斷句）
LEADING_PUNCT = re.compile(r'^[，,。.？?！!\s…、；;：:]')
PURE_REACTION  = re.compile(
    r'^[哈嗯喔哦欸誒啊呀咦嗯唉噢嘿嘻lolXDxd!！？?.。\.…,，\s笑死wWwWxX死惹]+$'
)
PURE_RESPONSE  = {
    '推','+1','推推','幫推','推一個','感謝','謝謝','感謝分享',
    '謝謝分享','感謝大大','謝謝大大','好文推','優文推','好問題',
    '有推','支持','認同','同意','正解','就是這樣','就這樣',
    '噓','靠北','幫噓','離題','離','水桶','水',
}
ONLY_SYMBOLS   = re.compile(
    r'^[\d\s\.\,\!\?\。\，\！\？\-\_\+\=\*\/\\|\[\]\{\}\(\)\<\>\"\'`~@#$%^&]+$'
)
IMG_RE         = re.compile(
    r'https?://i\.imgur\.com/\S+|https?://\S+\.(jpg|jpeg|png|gif|webp)', re.I
)

def get_filter_reason(content):
    if not content:
        return 'R2-空白'
    text = str(content).strip()
    if URL_RE.match(text):
        return 'R1-純URL'
    if IMG_RE.search(text) and len(IMG_RE.sub('', text).strip()) < 5:
        return 'R7-圖片連結'
    clean = URL_IN_TEXT.sub('', text).strip()
    if len(clean) < 8:
        return 'R2-過短'
    if TRAILING_BREAK.search(text):
        return 'R3-截斷句'
    if LEADING_PUNCT.match(text):
        return 'R3-截斷句'
    if PURE_REACTION.match(text):
        return 'R4-純表情'
    if text in PURE_RESPONSE or text.replace(' ', '') in PURE_RESPONSE:
        return 'R5-純回應'
    if ONLY_SYMBOLS.match(text):
        return 'R6-純符號'
    return None

# ── 主要流程 ───────────────────────────────────────────

def run(input_path):
    print(f'讀取：{input_path}')
    wb = openpyxl.load_workbook(input_path, read_only=True, data_only=True)

    # 找所有評論 sheet
    detail_sheets = {
        name: SHEET_TO_CAT[name]
        for name in wb.sheetnames
        if name in SHEET_TO_CAT
    }

    if not detail_sheets:
        print('[錯誤] 找不到評論 Sheet，請確認檔案格式是否正確')
        print(f'       現有 Sheet：{wb.sheetnames}')
        sys.exit(1)

    print(f'找到 {len(detail_sheets)} 個評論 Sheet：{list(detail_sheets.keys())}')
    print()

    all_kept    = []
    cat_stats   = {}
    rule_totals = defaultdict(int)

    for sheet_name, cat in detail_sheets.items():
        ws      = wb[sheet_name]
        headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
        ga_models = set(GA_MODELS.get(cat, []))

        total = kept_count = 0
        filtered = defaultdict(int)

        for row in ws.iter_rows(min_row=2, values_only=True):
            if not any(v is not None for v in row):
                continue
            d = dict(zip(headers, row))

            model = d.get('對應型號', '')
            if ga_models and model not in ga_models:
                continue
            if d.get('資料層級') == '查無資料':
                continue

            total += 1
            reason = get_filter_reason(d.get('評論內容', ''))

            if reason:
                filtered[reason] += 1
                rule_totals[reason] += 1
            else:
                kept_count += 1
                all_kept.append({
                    'category':   cat,
                    'model':      model,
                    'brand':      d.get('品牌', ''),
                    'series':     d.get('系列', ''),
                    'data_level': d.get('資料層級', ''),
                    'title':      d.get('文章標題', ''),
                    'url':        d.get('文章連結', ''),
                    'tag':        d.get('推噓類型', ''),
                    'content':    d.get('評論內容', ''),
                    'date':       d.get('發佈日期', ''),
                })

        rate_str = f'{kept_count/total*100:.1f}%' if total else 'N/A'
        cat_stats[cat] = {
            'display':  CAT_DISPLAY.get(cat, cat),
            'total':    total,
            'kept':     kept_count,
            'filtered': total - kept_count,
            'kept_rate':kept_count / total if total else 0,
            'by_rule':  dict(filtered),
        }
        print(f'  {CAT_DISPLAY.get(cat, cat):6s}：{total:6,} → 保留 {kept_count:6,}（{rate_str}）')

    wb.close()
    return all_kept, cat_stats, rule_totals


def export_jsonl(kept_rows, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'✅ JSONL：{output_path}（{len(kept_rows):,} 則）')


def export_report(cat_stats, rule_totals, output_path):
    wb = Workbook()
    wb.remove(wb.active)

    def bd():
        s = Side(style='thin')
        return Border(left=s, right=s, top=s, bottom=s)

    H_FONT = Font(name='Arial', bold=True, color='FFFFFF', size=10)
    H_FILL = PatternFill('solid', fgColor='1F3864')
    NORM   = Font(name='Arial', size=10)
    BOLD   = Font(name='Arial', size=10, bold=True)
    C_ALN  = Alignment(horizontal='center', vertical='center', wrap_text=True)
    L_ALN  = Alignment(horizontal='left',   vertical='center', wrap_text=True)
    OK_F   = PatternFill('solid', fgColor='E2EFDA')
    WN_F   = PatternFill('solid', fgColor='FFF2CC')
    ER_F   = PatternFill('solid', fgColor='FCE4D6')
    GRAY_F = PatternFill('solid', fgColor='F2F2F2')

    # Sheet 1：總覽
    ws1 = wb.create_sheet('過濾總覽')
    ws1.freeze_panes = 'A2'
    headers = ['類別','過濾前','過濾後','過濾則數','保留率',
               'R1-純URL','R2-過短','R3-截斷句','R4-純表情','R5-純回應','R6-純符號','R7-圖片']
    widths  = [10,10,10,10,9,9,9,9,9,9,9,9]

    for ci,(h,w) in enumerate(zip(headers,widths),1):
        c = ws1.cell(1,ci,h)
        c.font=H_FONT; c.fill=H_FILL; c.alignment=C_ALN; c.border=bd()
        ws1.column_dimensions[get_column_letter(ci)].width = w
    ws1.row_dimensions[1].height = 22

    tb = tb_a = 0
    for ri,(cat,s) in enumerate(cat_stats.items(),2):
        rate = s['kept_rate']
        fill = OK_F if rate>=0.7 else WN_F if rate>=0.5 else (ER_F if rate>0 else GRAY_F)
        r    = s['by_rule']
        vals = [s['display'],s['total'],s['kept'],s['filtered'],
                f"{rate:.1%}" if s['total'] else 'N/A',
                r.get('R1-純URL',0),r.get('R2-過短',0)+r.get('R2-空白',0),
                r.get('R3-截斷句',0),r.get('R4-純表情',0),
                r.get('R5-純回應',0),r.get('R6-純符號',0),r.get('R7-圖片連結',0)]
        for ci,v in enumerate(vals,1):
            c = ws1.cell(ri,ci,v)
            c.font=NORM; c.fill=fill; c.alignment=C_ALN; c.border=bd()
        ws1.row_dimensions[ri].height = 20
        tb += s['total']; tb_a += s['kept']

    ri_t = len(cat_stats)+2
    r_total = tb_a/tb if tb else 0
    rt = rule_totals
    vt = ['合計',tb,tb_a,tb-tb_a,f"{r_total:.1%}",
          rt.get('R1-純URL',0),rt.get('R2-過短',0)+rt.get('R2-空白',0),
          rt.get('R3-截斷句',0),rt.get('R4-純表情',0),
          rt.get('R5-純回應',0),rt.get('R6-純符號',0),rt.get('R7-圖片連結',0)]
    for ci,v in enumerate(vt,1):
        c = ws1.cell(ri_t,ci,v)
        c.font=BOLD; c.border=bd(); c.alignment=C_ALN

    # Sheet 2：規則說明
    ws2 = wb.create_sheet('規則說明')
    ws2.column_dimensions['A'].width = 14
    ws2.column_dimensions['B'].width = 22
    ws2.column_dimensions['C'].width = 55
    for ci,h in enumerate(['規則代號','規則名稱','說明與範例'],1):
        c = ws2.cell(1,ci,h)
        c.font=H_FONT; c.fill=H_FILL; c.alignment=C_ALN; c.border=bd()
    ws2.row_dimensions[1].height = 22

    rules_desc = [
        ('R1','純 URL',     '整則評論只有一個網址，無文字。\n例：https://i.imgur.com/abc.jpg'),
        ('R2','過短',       '移除 URL 後內容少於 8 字，通常無實質資訊。\n例：「後來補的」、「對啊」'),
        ('R3','截斷句',     '句子以轉折詞結尾，評論被截斷不完整。\n例：「因為」、「但是」結尾'),
        ('R4','純表情/笑聲','只有表情詞、笑聲、感嘆詞，無實質內容。\n例：「XD」、「哈哈哈」'),
        ('R5','純回應用語', '只有制式回應，沒有評價內容。\n例：「推」、「+1」、「感謝分享」'),
        ('R6','純符號數字', '只有符號或數字，沒有文字。\n例：「123」、「...」'),
        ('R7','圖片連結',   '含圖片連結且幾乎無其他文字。\n例：imgur 連結'),
    ]
    for ri,(code,name,desc) in enumerate(rules_desc,2):
        for ci,v in enumerate([code,name,desc],1):
            c = ws2.cell(ri,ci,v)
            c.font=NORM; c.border=bd()
            c.alignment=C_ALN if ci<=2 else L_ALN
        ws2.row_dimensions[ri].height = 50

    wb.save(output_path)
    print(f'✅ 報告：{output_path}')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 決定輸入路徑
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # 預設：腳本所在資料夾的 data/ptt_comment.xlsx
        input_path = os.path.join(script_dir, DEFAULT_INPUT)

    if not os.path.exists(input_path):
        print(f'[錯誤] 找不到檔案：{input_path}')
        print(f'       請確認專案根目錄的 data/ptt_comment.xlsx 存在，或指定路徑：')
        print(f'       python comment_filter.py /path/to/ptt_comment.xlsx')
        sys.exit(1)

    # 輸出到腳本所在資料夾的 output/ 子資料夾
    output_dir  = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path  = os.path.join(output_dir, OUTPUT_JSONL)
    report_path = os.path.join(output_dir, OUTPUT_REPORT)

    print('=' * 60)
    print('  PTT 硬體評論過濾腳本')
    print('=' * 60)
    print()

    kept_rows, cat_stats, rule_totals = run(input_path)

    print()
    total_before = sum(s['total'] for s in cat_stats.values())
    total_after  = len(kept_rows)
    print(f'過濾完成：{total_before:,} → {total_after:,} 則')
    if total_before:
        print(f'整體保留率：{total_after/total_before*100:.1f}%')
    print()

    export_jsonl(kept_rows, jsonl_path)
    export_report(cat_stats, rule_totals, report_path)

    print()
    print('規則觸發次數：')
    for rule, count in sorted(rule_totals.items(), key=lambda x: -x[1]):
        print(f'  {rule}: {count:,} 則')


if __name__ == '__main__':
    main()