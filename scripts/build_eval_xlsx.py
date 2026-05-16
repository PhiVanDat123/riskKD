"""Build SUMMARY-mtbench-alpaca.xlsx from MT-Bench + AlpacaEval scoreboard JSONs.

Sheets:
  1. Paper Table (Qwen + Llama, lm-eval AVG + MT-Bench + AlpacaEval LC)
  2. MT-Bench Detail (all 25 models, full per-category breakdown)
  3. AlpacaEval Detail (all 25 models, all fields)
"""
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path("/workspace/riskKD/results/eval")
MTB = json.load(open(ROOT / "mtbench" / "scores.json"))
ALP = json.load(open(ROOT / "alpaca" / "scores.json"))

LABELS = {
    "vukien2301/qwen3-1.7b-deita-sft-student": "SFT student (1.7B)",
    "vukien2301/qwen3-1.7b-ultrafeedback-dpo": "DPO",
    "vukien2301/qwen3-1.7b-ultrafeedback-wpo": "WPO",
    "vukien2301/qwen3-1.7b-ultrafeedback-radpo": "Ra-DPO",
    "vukien2301/qwen3-1.7b-ultrafeedback-dckd": "DCKD",
    "vukien2301/qwen3-1.7b-ultrafeedback-adpa": "ADPA",
    "vukien2301/qwen3-1.7b-ultrafeedback-tvkd": "TVKD",
    "vukien2301/qwen3-1.7b-riskKD": "riskKD (Ours)",
    "vukien2301/qwen3-1.7b-tailriskKD": "tailriskKD (Ours)",
    "vukien2301/qwen3-1.7b-pfw-tail": "pfw-tail (Ours)",
    "vukien2301/qwen3-8b-deita-sft-teacher": "SFT teacher (8B)",
    "vukien2301/qwen3-8b-ultrafeedback-dpo-teacher": "DPO teacher (8B)",
    "vukien2301/llama-3.2-1b-deita-sft-student": "SFT student (1B)",
    "vukien2301/llama-3.2-1b-dckd-ultrafeedback": "DCKD",
    "vukien2301/llama-3.2-1b-adpa-ultrafeedback": "ADPA",
    "vukien2301/llama-3.2-1b-tvkd-ultrafeedback": "TVKD",
    "vukien2301/llama-3.2-1b-riskkd-tokenwt-epoch1": "riskKD-tokenwt (Ours)",
    "vukien2301/llama3.2-1b-tailriskKD": "tailriskKD (Ours)",
    "vukien2301/llama3.2-1b-pfw-tail": "pfw-tail (Ours)",
    "vukien2301/llama-3.2-1b-georiskKD-risk": "GeoRiskKD-risk (Ours)",
    "vukien2301/llama-3.2-1b-georiskKD-all": "GeoRiskKD-all (Ours)",
    "vukien2301/llama-3.2-1b-georiskKD-align-heavy": "GeoRiskKD-align-heavy (Ours)",
    "vukien2301/llama-3.2-1b-georiskKD-kl-conservative": "GeoRiskKD-kl-cons (Ours)",
    "vukien2301/llama-3.1-8b-deita-sft-teacher-epoch1": "SFT teacher (8B)",
    "vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1": "DPO teacher (8B)",
}

# lm-eval AVG from existing SUMMARY files
LM_EVAL = {
    # Qwen 1.7B students (from SUMMARY-qwen3-1.7b.md)
    "vukien2301/qwen3-1.7b-deita-sft-student": 60.59,
    "vukien2301/qwen3-1.7b-ultrafeedback-dpo": 61.80,
    "vukien2301/qwen3-1.7b-ultrafeedback-wpo": 62.38,
    "vukien2301/qwen3-1.7b-ultrafeedback-radpo": 62.62,
    "vukien2301/qwen3-1.7b-ultrafeedback-dckd": 61.63,
    "vukien2301/qwen3-1.7b-ultrafeedback-adpa": 59.20,
    "vukien2301/qwen3-1.7b-ultrafeedback-tvkd": 60.33,
    "vukien2301/qwen3-1.7b-riskKD": 63.01,
    "vukien2301/qwen3-1.7b-tailriskKD": 63.08,
    "vukien2301/qwen3-1.7b-pfw-tail": 63.00,
    # Qwen 8B teachers
    "vukien2301/qwen3-8b-deita-sft-teacher": 72.12,
    "vukien2301/qwen3-8b-ultrafeedback-dpo-teacher": 73.81,
    # Llama 1B students (from user's spreadsheet pasted earlier + SUMMARY-georiskKD-1ep.md)
    "vukien2301/llama-3.2-1b-deita-sft-student": 41.25,
    "vukien2301/llama-3.2-1b-dckd-ultrafeedback": 42.13,
    "vukien2301/llama-3.2-1b-adpa-ultrafeedback": 42.15,
    "vukien2301/llama-3.2-1b-tvkd-ultrafeedback": 40.97,
    "vukien2301/llama-3.2-1b-riskkd-tokenwt-epoch1": 43.15,  # the kl_inv variant; user's "Ours" in their table = 43.27
    "vukien2301/llama-3.2-1b-georiskKD-risk": 40.26,
    "vukien2301/llama-3.2-1b-georiskKD-all": 41.93,
    "vukien2301/llama-3.2-1b-georiskKD-align-heavy": 41.30,
    "vukien2301/llama-3.2-1b-georiskKD-kl-conservative": 40.97,
    # Llama 8B teachers
    "vukien2301/llama-3.1-8b-deita-sft-teacher-epoch1": 62.12,
    "vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1": 64.42,
}

CATS = ["writing", "roleplay", "extraction", "math", "coding", "reasoning", "stem", "humanities"]

HEADER_FILL = PatternFill(start_color="305496", end_color="305496", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)
OURS_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
TEACHER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
FLOOR_FILL = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")
BORDER = Border(
    left=Side(style="thin", color="999999"),
    right=Side(style="thin", color="999999"),
    top=Side(style="thin", color="999999"),
    bottom=Side(style="thin", color="999999"),
)
CENTER = Alignment(horizontal="center", vertical="center")


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
        cell.border = BORDER


def style_data(ws, row, ncols, fill=None):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.border = BORDER
        cell.alignment = Alignment(horizontal="center")
        if c == 1:
            cell.alignment = Alignment(horizontal="left")
        if fill:
            cell.fill = fill


def auto_width(ws, ncols):
    for c in range(1, ncols + 1):
        col_letter = get_column_letter(c)
        max_len = 0
        for cell in ws[col_letter]:
            v = cell.value
            if v is None:
                continue
            l = len(str(v))
            if l > max_len:
                max_len = l
        ws.column_dimensions[col_letter].width = min(max_len + 3, 40)


def row_fill(label, repo_id):
    if "(Ours)" in label:
        return OURS_FILL
    if "teacher" in label.lower():
        return TEACHER_FILL
    if "SFT student" in label or "SFT (floor)" in label:
        return FLOOR_FILL
    return None


def get_mt(repo_id, field):
    m = MTB["models"].get(repo_id)
    return m.get(field) if m else None


def get_alp(repo_id, field):
    a = ALP["models"].get(repo_id)
    return a.get(field) if a else None


wb = Workbook()

# --- SHEET 1: Paper Table ---
ws1 = wb.active
ws1.title = "Paper Table"

# Qwen first
ws1.append(["QWEN3 family (1.7B student <- 8B teacher)"])
ws1.cell(row=1, column=1).font = Font(bold=True, size=14)
ws1.merge_cells(start_row=1, start_column=1, end_row=1, end_column=6)
ws1.append([])

hdr = ["Method", "lm-eval AVG", "MT-Bench overall", "MT-Bench T1", "MT-Bench T2", "AlpacaEval LC", "AlpacaEval WR", "MT-Bench n", "Alpaca n"]
ws1.append(hdr)
style_header(ws1, 3, len(hdr))

# Order: teachers, then students sorted by AlpacaEval LC (or fallback MT-Bench)
qwen_order = [
    "vukien2301/qwen3-8b-ultrafeedback-dpo-teacher",
    "vukien2301/qwen3-8b-deita-sft-teacher",
    "vukien2301/qwen3-1.7b-ultrafeedback-adpa",
    "vukien2301/qwen3-1.7b-pfw-tail",
    "vukien2301/qwen3-1.7b-ultrafeedback-dckd",
    "vukien2301/qwen3-1.7b-riskKD",
    "vukien2301/qwen3-1.7b-tailriskKD",
    "vukien2301/qwen3-1.7b-ultrafeedback-dpo",
    "vukien2301/qwen3-1.7b-ultrafeedback-wpo",
    "vukien2301/qwen3-1.7b-ultrafeedback-radpo",
    "vukien2301/qwen3-1.7b-ultrafeedback-tvkd",
    "vukien2301/qwen3-1.7b-deita-sft-student",
]
for rid in qwen_order:
    label = LABELS.get(rid, rid)
    ws1.append([
        label,
        LM_EVAL.get(rid, ""),
        get_mt(rid, "overall"),
        get_mt(rid, "turn1"),
        get_mt(rid, "turn2"),
        get_alp(rid, "lc_win_rate"),
        get_alp(rid, "win_rate"),
        get_mt(rid, "n_judged"),
        get_alp(rid, "n_total"),
    ])
    style_data(ws1, ws1.max_row, len(hdr), row_fill(label, rid))

# Llama
ws1.append([])
ws1.append(["LLAMA family (1B student <- 8B teacher)"])
ws1.cell(row=ws1.max_row, column=1).font = Font(bold=True, size=14)
ws1.merge_cells(start_row=ws1.max_row, start_column=1, end_row=ws1.max_row, end_column=6)
ws1.append([])
ws1.append(hdr)
style_header(ws1, ws1.max_row, len(hdr))

llama_order = [
    "vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1",
    "vukien2301/llama-3.1-8b-deita-sft-teacher-epoch1",
    "vukien2301/llama3.2-1b-tailriskKD",
    "vukien2301/llama-3.2-1b-dckd-ultrafeedback",
    "vukien2301/llama3.2-1b-pfw-tail",
    "vukien2301/llama-3.2-1b-georiskKD-all",
    "vukien2301/llama-3.2-1b-tvkd-ultrafeedback",
    "vukien2301/llama-3.2-1b-riskkd-tokenwt-epoch1",
    "vukien2301/llama-3.2-1b-deita-sft-student",
    "vukien2301/llama-3.2-1b-adpa-ultrafeedback",
    "vukien2301/llama-3.2-1b-georiskKD-risk",
    "vukien2301/llama-3.2-1b-georiskKD-align-heavy",
    "vukien2301/llama-3.2-1b-georiskKD-kl-conservative",
]
for rid in llama_order:
    label = LABELS.get(rid, rid)
    ws1.append([
        label,
        LM_EVAL.get(rid, ""),
        get_mt(rid, "overall"),
        get_mt(rid, "turn1"),
        get_mt(rid, "turn2"),
        get_alp(rid, "lc_win_rate"),
        get_alp(rid, "win_rate"),
        get_mt(rid, "n_judged"),
        get_alp(rid, "n_total"),
    ])
    style_data(ws1, ws1.max_row, len(hdr), row_fill(label, rid))

auto_width(ws1, len(hdr))

# --- SHEET 2: MT-Bench Detail ---
ws2 = wb.create_sheet("MT-Bench Detail")
hdr2 = ["Family", "Method", "Overall", "Turn 1", "Turn 2"] + [c.title() for c in CATS] + ["n_judged"]
ws2.append(hdr2)
style_header(ws2, 1, len(hdr2))

for fam_key in ["qwen3-1.7b", "qwen3-8b", "llama-3.2-1b", "llama-3.1-8b"]:
    rows = [(k, v) for k, v in MTB["models"].items() if v.get("family") == fam_key]
    rows.sort(key=lambda x: -(x[1].get("overall") or 0))
    for k, v in rows:
        label = LABELS.get(k, k.split("/")[-1])
        cats = v.get("by_category", {})
        ws2.append(
            [fam_key, label, v.get("overall"), v.get("turn1"), v.get("turn2")]
            + [cats.get(c) for c in CATS]
            + [v.get("n_judged")]
        )
        style_data(ws2, ws2.max_row, len(hdr2), row_fill(label, k))

auto_width(ws2, len(hdr2))

# --- SHEET 3: AlpacaEval Detail ---
ws3 = wb.create_sheet("AlpacaEval Detail")
hdr3 = ["Family", "Method", "LC win-rate", "Win-rate", "Std Err", "Avg Length",
        "n_total", "n_wins", "n_draws", "n_losses"]
ws3.append(hdr3)
style_header(ws3, 1, len(hdr3))

for fam_key in ["qwen3-1.7b", "qwen3-8b", "llama-3.2-1b", "llama-3.1-8b"]:
    rows = [(k, v) for k, v in ALP["models"].items() if v.get("family") == fam_key]
    rows.sort(key=lambda x: -(x[1].get("lc_win_rate") or 0))
    for k, v in rows:
        label = LABELS.get(k, k.split("/")[-1])
        ws3.append([
            fam_key, label,
            v.get("lc_win_rate"), v.get("win_rate"),
            v.get("standard_error"), v.get("avg_length"),
            v.get("n_total"), v.get("n_wins"), v.get("n_draws"), v.get("n_losses"),
        ])
        style_data(ws3, ws3.max_row, len(hdr3), row_fill(label, k))

auto_width(ws3, len(hdr3))

# --- SHEET 4: Distillation Gap ---
ws4 = wb.create_sheet("Distillation Gap")

def gap_table(start_row, family, floor_rid, ceil_rid, fam_title):
    floor_mt = get_mt(floor_rid, "overall") or 0
    ceil_mt = get_mt(ceil_rid, "overall") or 0
    floor_lc = get_alp(floor_rid, "lc_win_rate") or 0
    ceil_lc = get_alp(ceil_rid, "lc_win_rate") or 0
    span_mt = ceil_mt - floor_mt
    span_lc = ceil_lc - floor_lc

    ws4.cell(row=start_row, column=1, value=fam_title).font = Font(bold=True, size=14)
    ws4.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=8)
    start_row += 1
    ws4.cell(row=start_row, column=1, value=f"Floor (SFT student): MT={floor_mt:.2f} | LC={floor_lc:.2f}").font = Font(italic=True)
    start_row += 1
    ws4.cell(row=start_row, column=1, value=f"Ceiling (8B DPO teacher): MT={ceil_mt:.2f} | LC={ceil_lc:.2f}").font = Font(italic=True)
    start_row += 1
    ws4.cell(row=start_row, column=1, value=f"Span: MT={span_mt:.2f} | LC={span_lc:.2f}").font = Font(italic=True)
    start_row += 2

    hdr = ["Method", "MT-Bench", "MT gap closed %", "AlpacaEval LC", "LC gap closed %"]
    for i, h in enumerate(hdr, 1):
        cell = ws4.cell(row=start_row, column=i, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
        cell.border = BORDER
    start_row += 1

    rows = [(k, v) for k, v in MTB["models"].items() if v.get("family") == family and k not in (floor_rid, ceil_rid)]
    rows.sort(key=lambda x: -(x[1].get("overall") or 0))
    for k, v in rows:
        label = LABELS.get(k, k.split("/")[-1])
        mt_o = get_mt(k, "overall")
        lc_o = get_alp(k, "lc_win_rate")
        mt_pct = ((mt_o - floor_mt) / span_mt * 100) if (mt_o is not None and span_mt > 0) else None
        lc_pct = ((lc_o - floor_lc) / span_lc * 100) if (lc_o is not None and span_lc > 0) else None
        ws4.cell(row=start_row, column=1, value=label)
        ws4.cell(row=start_row, column=2, value=mt_o)
        ws4.cell(row=start_row, column=3, value=f"{mt_pct:.0f}%" if mt_pct is not None else "")
        ws4.cell(row=start_row, column=4, value=lc_o)
        ws4.cell(row=start_row, column=5, value=f"{lc_pct:.0f}%" if lc_pct is not None else "")
        fill = row_fill(label, k)
        for c in range(1, 6):
            cc = ws4.cell(row=start_row, column=c)
            cc.border = BORDER
            cc.alignment = Alignment(horizontal="center" if c > 1 else "left")
            if fill:
                cc.fill = fill
        start_row += 1
    return start_row + 2


next_row = gap_table(1, "qwen3-1.7b",
                     "vukien2301/qwen3-1.7b-deita-sft-student",
                     "vukien2301/qwen3-8b-ultrafeedback-dpo-teacher",
                     "Qwen3 (1.7B student <- 8B DPO teacher)")
gap_table(next_row, "llama-3.2-1b",
          "vukien2301/llama-3.2-1b-deita-sft-student",
          "vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1",
          "Llama 3.2-1B (1B student <- 8B DPO teacher)")

auto_width(ws4, 5)

OUT = ROOT / "SUMMARY-mtbench-alpaca.xlsx"
wb.save(OUT)
print(f"Wrote {OUT}")
