# -*- coding: utf-8 -*-
"""Export 数据改变对比表 to xlsx with formatted sheets."""
import os

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
except ImportError:
    print("NO_OPENPYXL")
    raise SystemExit(0)

ROOT = os.path.dirname(os.path.abspath(__file__))

wb = Workbook()
header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill("solid", fgColor="4472C4")
good_fill = PatternFill("solid", fgColor="C6EFCE")
bad_fill = PatternFill("solid", fgColor="FFC7CE")
center = Alignment(horizontal="center")


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center


def autofit(ws):
    for col in ws.columns:
        width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(width + 4, 40)


# Sheet 1: 每个种子是否提升
ws = wb.active
ws.title = "每个种子是否提升"
ws.append(["seed", "基线 OA", "D 配置 OA", "ΔOA", "提升?"])
style_header(ws, 1, 5)
for row in [
    [42, 79.20, 86.64, +7.44, "✅"],
    [123, 80.94, 92.84, +11.89, "✅"],
    [456, 81.92, 86.83, +4.91, "✅"],
]:
    ws.append(row)
    ws.cell(row=ws.max_row, column=5).fill = good_fill
ws.append(["结论", "3 个种子全部提升，非单种子运气"])
ws.merge_cells(start_row=5, start_column=1, end_row=5, end_column=5)
autofit(ws)

# Sheet 2: 逐组逐种子
ws = wb.create_sheet("逐组逐种子(基线vs D)")
ws.append(["组", "seed", "基线 OA", "D 配置 OA", "ΔOA"])
style_header(ws, 1, 5)
rows2 = [
    ["G1", 42, 58.31, 88.21, +29.90], ["G1", 123, 64.74, 93.50, +28.76], ["G1", 456, 72.34, 77.92, +5.58],
    ["G2", 42, 88.63, 90.84, +2.21], ["G2", 123, 88.97, 93.70, +4.73], ["G2", 456, 81.79, 87.78, +5.99],
    ["G3", 42, 73.56, 77.25, +3.69], ["G3", 123, 87.87, 89.82, +1.95], ["G3", 456, 92.85, 93.50, +0.65],
    ["G4", 42, 88.12, 90.86, +2.74], ["G4", 123, 90.21, 91.25, +1.04], ["G4", 456, 86.16, 89.82, +3.66],
    ["G5", 42, 87.39, 86.05, -1.34], ["G5", 123, 72.93, 95.91, +22.98], ["G5", 456, 76.44, 85.13, +8.69],
]
for r in rows2:
    ws.append(r)
    delta_cell = ws.cell(row=ws.max_row, column=5)
    delta_cell.fill = good_fill if delta_cell.value > 0 else bad_fill
autofit(ws)

# Sheet 3: 各配置汇总
ws = wb.create_sheet("各配置汇总")
ws.append(["配置", "G1", "G2", "G3", "G4", "G5", "均值 OA"])
style_header(ws, 1, 7)
rows3 = [
    ["基线（融合开，其余全关）", 65.13, 86.46, 84.76, 88.16, 78.92, 80.69],
    ["E（只 z-score）", 63.68, 88.09, 72.01, 90.42, 79.11, 78.66],
    ["F（只平衡采样）", 82.12, 82.65, 93.71, 76.76, 90.23, 85.09],
    ["C（z-score+平衡）", 84.25, 89.40, 81.87, 90.47, 84.91, 86.18],
    ["D（关融合+z-score+平衡）", 86.54, 90.77, 86.86, 90.64, 89.03, 88.77],
    ["全改动（四样全开）", 61.51, 88.08, 87.61, 90.86, 81.70, 81.95],
    ["B（只负样本，seed42）", 48.85, 87.55, 89.92, 76.89, 84.29, 77.50],
]
for r in rows3:
    ws.append(r)
ws.cell(row=6, column=7).fill = good_fill  # D 最优
autofit(ws)

# Sheet 4: 归因对账
ws = wb.create_sheet("归因对账")
ws.append(["改动", "对比", "OA 差", "结论"])
style_header(ws, 1, 4)
for r in [
    ["类别平衡采样", "F − 基线", "+4.40", "最大功臣"],
    ["关掉动态融合", "D − C", "+2.59", "动态调节净负"],
    ["z-score×平衡 协同", "C − (E+F+基线)", "+3.12", "一起开才有加成"],
    ["z-score 单独", "E − 基线", "−2.03", "单独用反而小负"],
    ["unseen 负样本", "全改动 − C / B − 基线", "−4.23 / −3.38", "负贡献，排除"],
]:
    ws.append(r)
ws.append(["对账", "+4.40 − 2.03 + 3.12 + 2.59 = +8.08 = D(88.77) − 基线(80.69) ✅"])
ws.merge_cells(start_row=7, start_column=1, end_row=7, end_column=4)
autofit(ws)

# Sheet 5: LongKou
ws = wb.create_sheet("LongKou")
ws.append(["配置", "OA", "AA", "Kappa", "说明"])
style_header(ws, 1, 5)
for r in [
    ["旧基线（seed42）", 69.54, 55.89, 0.4923, "dynamic_seed42_20ep 存档"],
    ["C（开 z-score）", 36.30, 46.68, 0.0815, "z-score 破坏 float32(0-28) 分布"],
    ["关 z-score+关融合+平衡", 54.35, 57.02, 0.3279, "3-seed；G1 大胜，G3 仍难"],
]:
    ws.append(r)
autofit(ws)

out = os.path.join(ROOT, "数据改变对比表.xlsx")
wb.save(out)
print("written:", out)
