"""
exporter.py
Excel 輸出模組

每個類別輸出兩個 Sheet：
  {類別}_評論明細  - 每則推文一行
  {類別}_統計      - 各型號推噓數統計

另外附一個「執行摘要」Sheet 匯總所有類別。
"""

import re as _re
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from hardware_config import CATEGORY_DISPLAY

# ══════════════════════════════════════════════════════
#  工具函式
# ══════════════════════════════════════════════════════

_ILLEGAL_CHARS = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

def sanitize(val):
    """清除 Excel 不支援的控制字元"""
    if val is None:
        return ""
    if isinstance(val, float) and val != val:
        return ""
    if isinstance(val, str):
        return _ILLEGAL_CHARS.sub("", val)
    return val

# ══════════════════════════════════════════════════════
#  樣式常數
# ══════════════════════════════════════════════════════

def _thin():
    s = Side(style="thin")
    return Border(left=s, right=s, top=s, bottom=s)

H_FONT      = Font(name="Arial", bold=True, color="FFFFFF", size=10)
H_FILL      = PatternFill("solid", fgColor="1F3864")
C_ALN       = Alignment(horizontal="center", vertical="center", wrap_text=True)
L_ALN       = Alignment(horizontal="left",   vertical="center", wrap_text=True)
NORM_FONT   = Font(name="Arial", size=9)
GRAY_FONT   = Font(name="Arial", size=9, color="999999", italic=True)

PUSH_FILL   = PatternFill("solid", fgColor="E2EFDA")  # 推   → 淺綠
BOO_FILL    = PatternFill("solid", fgColor="FCE4D6")  # 噓   → 淺紅
ARROW_FILL  = PatternFill("solid", fgColor="FFFFFF")  # →   → 白
FB_FILL     = PatternFill("solid", fgColor="EBF3FB")  # fallback → 淺藍
INFO_FILL   = PatternFill("solid", fgColor="FFF8DC")  # 情報文   → 淺黃
NODATA_FILL = PatternFill("solid", fgColor="F2F2F2")  # 查無 → 灰
RED_FILL    = PatternFill("solid", fgColor="FCE4D6")
GRN_FILL    = PatternFill("solid", fgColor="E2EFDA")
YLW_FILL    = PatternFill("solid", fgColor="FFF2CC")

# ══════════════════════════════════════════════════════
#  單一類別：評論明細 Sheet
# ══════════════════════════════════════════════════════

MIN_CONTENT_LEN = 2  # 評論內容最短字數，低於此值視為無意義

def _should_keep(row):
    """
    回傳 True 表示保留這則評論，False 表示刪除
    規則一：評論內容空白 → 刪除
    規則二：評論內容少於 MIN_CONTENT_LEN 字 → 刪除
    """
    content = row.get('評論內容', '')
    content_str = str(content).strip() if content is not None else ''
    if not content_str:
        return False
    if len(content_str) < MIN_CONTENT_LEN:
        return False
    return True


def _write_detail_sheet(wb, sheet_name, all_rows):
    # 清理：過濾空白與過短評論
    all_rows = [r for r in all_rows if _should_keep(r)]

    ws = wb.create_sheet(sheet_name)
    thin = _thin()

    cols   = ["對應型號", "品牌", "系列", "資料層級",
              "文章標題", "文章連結", "推噓類型", "評論內容", "發佈日期"]
    widths = [22, 8, 16, 16, 45, 52, 10, 72, 14]
    center_cols = {1, 2, 3, 4, 7, 9}

    for ci, (h, w) in enumerate(zip(cols, widths), 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.font = H_FONT; cell.fill = H_FILL
        cell.alignment = C_ALN; cell.border = thin
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.row_dimensions[1].height = 24
    ws.freeze_panes = "A2"

    for ri, row in enumerate(all_rows, 2):
        level  = row.get("資料層級", "")
        tag    = row.get("推噓類型", "")
        nodata = (level == "查無資料")

        if nodata:
            row_fill = NODATA_FILL
        elif "情報" in level and "fallback" in level:
            row_fill = FB_FILL      # fallback+情報 → 淺藍（fallback 優先）
        elif "情報" in level:
            row_fill = INFO_FILL    # 情報文 → 淺黃
        elif "fallback" in level:
            row_fill = FB_FILL      # fallback → 淺藍
        elif tag == "推":
            row_fill = PUSH_FILL
        elif tag == "噓":
            row_fill = BOO_FILL
        else:
            row_fill = ARROW_FILL

        for ci, key in enumerate(cols, 1):
            val  = sanitize(row.get(key, ""))
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.fill      = row_fill
            cell.border    = thin
            cell.alignment = C_ALN if ci in center_cols else L_ALN
            cell.font      = GRAY_FONT if nodata else NORM_FONT
            if key == "發佈日期":
                cell.number_format = "@"
        ws.row_dimensions[ri].height = 38

# ══════════════════════════════════════════════════════
#  單一類別：統計 Sheet
# ══════════════════════════════════════════════════════

def _write_stats_sheet(wb, sheet_name, summary_stats):
    ws = wb.create_sheet(sheet_name)
    thin = _thin()

    s_cols  = ["型號", "品牌", "系列", "資料層級",
               "文章數", "推數", "噓數", "→數", "總評論數", "有無資料"]
    s_widths = [22, 8, 16, 16, 8, 8, 8, 8, 10, 10]

    for ci, (h, w) in enumerate(zip(s_cols, s_widths), 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.font = H_FONT; cell.fill = H_FILL
        cell.alignment = C_ALN; cell.border = thin
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.row_dimensions[1].height = 24
    ws.freeze_panes = "A2"

    for ri, stat in enumerate(summary_stats, 2):
        total   = stat.get("總評論數", 0)
        boo_cnt = stat.get("噓數", 0)
        row_fill = (NODATA_FILL if total == 0 else
                    RED_FILL    if boo_cnt >= 10 else
                    YLW_FILL    if "fallback" in stat.get("資料層級", "") else
                    GRN_FILL)
        for ci, key in enumerate(s_cols, 1):
            cell = ws.cell(row=ri, column=ci, value=stat.get(key, ""))
            cell.fill = row_fill; cell.border = thin
            cell.alignment = C_ALN
            cell.font = Font(name="Arial", size=10)
        ws.row_dimensions[ri].height = 22

# ══════════════════════════════════════════════════════
#  執行摘要 Sheet（所有類別匯總）
# ══════════════════════════════════════════════════════

def _write_summary_sheet(wb, results, categories_run):
    """
    results: dict { category_key: (all_rows, summary_stats) }
    """
    ws = wb.create_sheet("執行摘要")
    thin = _thin()

    h_font = Font(name="Arial", bold=True, size=11)
    v_font = Font(name="Arial", size=11)

    rows_data = []
    rows_data.append(("執行時間", datetime.now().strftime("%Y-%m-%d %H:%M")))
    rows_data.append(("搜尋看板", "PC_Shopping, Hardware"))
    rows_data.append(("最大翻頁數", 10))
    rows_data.append(("執行類別", ", ".join(
        CATEGORY_DISPLAY.get(c, c) for c in categories_run
    )))
    rows_data.append(("", ""))  # 空行

    for cat in categories_run:
        if cat not in results:
            continue
        all_rows, summary_stats = results[cat]
        display = CATEGORY_DISPLAY.get(cat, cat)
        real    = [r for r in all_rows if r.get("資料層級") != "查無資料"]

        rows_data.append((f"── {display} ──", ""))
        rows_data.append(("  搜尋型號數",   len(summary_stats)))
        rows_data.append(("  有資料型號數", sum(1 for s in summary_stats if s.get("總評論數", 0) > 0)))
        rows_data.append(("  查無資料型號", sum(1 for s in summary_stats if s.get("總評論數", 0) == 0)))
        rows_data.append(("  總評論數",     len(real)))
        rows_data.append(("  推",           sum(1 for r in real if r.get("推噓類型") == "推")))
        rows_data.append(("  噓",           sum(1 for r in real if r.get("推噓類型") == "噓")))
        rows_data.append(("  →",            sum(1 for r in real if r.get("推噓類型") == "→")))
        if summary_stats:
            top = max(summary_stats, key=lambda x: x.get("總評論數", 0))
            rows_data.append(("  評論最多型號", f"{top['型號']}（{top['總評論數']} 則）"))
        rows_data.append(("", ""))

    for ri, (k, v) in enumerate(rows_data, 1):
        cell_k = ws.cell(row=ri, column=1, value=k)
        cell_v = ws.cell(row=ri, column=2, value=v)
        cell_k.font = h_font
        cell_v.font = v_font

    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 45

# ══════════════════════════════════════════════════════
#  主要匯出函式
# ══════════════════════════════════════════════════════

def export_excel(results, categories_run):
    """
    results: dict { category_key: (all_rows, summary_stats) }
    輸出一份多 Sheet 的 Excel 檔案

    Sheet 結構：
      執行摘要
      {類別}_評論  （評論明細）
      {類別}_統計  （型號統計）
      ...
    """
    wb = Workbook()
    # 刪除預設的空白 Sheet
    wb.remove(wb.active)

    # 先寫執行摘要
    _write_summary_sheet(wb, results, categories_run)

    # 再逐類別寫入
    for cat in categories_run:
        if cat not in results:
            continue
        all_rows, summary_stats = results[cat]
        display = CATEGORY_DISPLAY.get(cat, cat)

        # Sheet 名稱限制 31 字元，取前 14 字
        detail_name = f"{display[:14]}_評論"
        stats_name  = f"{display[:14]}_統計"

        print(f"  寫入 Sheet：{detail_name}（{len(all_rows)} 行）")
        _write_detail_sheet(wb, detail_name, all_rows)

        print(f"  寫入 Sheet：{stats_name}（{len(summary_stats)} 型號）")
        _write_stats_sheet(wb, stats_name, summary_stats)

    filename = f"ptt_硬體評論_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    wb.save(filename)
    return filename