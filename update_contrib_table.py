# -*- coding: utf-8 -*-
"""Update 改动贡献一览.xlsx with final LongKou conclusions and delivery sheet."""
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

XLSX = Path(r"C:\Users\39448\Desktop\SpectralZero\result_IP\改动贡献一览.xlsx")

wb = openpyxl.load_workbook(XLSX)
ws = wb["改动贡献一览"]

# rows: header=1, data=2..10 (current 9 items)
# Row 8 (idx 8) = LongKou 关 z-score (未定论) -> final conclusion
# Row 9 (idx 9) = 多模板文本增强 (未验证) -> measured negative, rolled back

ws.cell(row=8, column=6).value = "已完整验证：G1 修复(88.75→92.14)但G2种子波动大、3-seed总OA 54.35<69.54，LongKou不列为创新点"
ws.cell(row=8, column=5).value = "G1 +3.39 | 总体 -15.19"

ws.cell(row=9, column=6).value = "已实测：LongKou压缩属性 OA 46.54 vs 54.35 = -7.81，负贡献，属性已回滚；Indian未单测"
ws.cell(row=9, column=5).value = "OA -7.81"
ws.cell(row=9, column=4).value = "负贡献"

# append final LongKou compact-text row
next_row = ws.max_row + 1
ws.cell(row=next_row, column=1, value=10)
ws.cell(row=next_row, column=2, value="压缩文本属性（LongKou）")
ws.cell(row=next_row, column=3, value="负贡献")
ws.cell(row=next_row, column=4, value="OA -7.81")
ws.cell(row=next_row, column=5, value="AA -3.80")
ws.cell(row=next_row, column=6, value="离线判别性提升≠端到端提升，已回滚属性，保留诚实记录")

# delivery sheet
if "最终交付" in wb.sheetnames:
    del wb["最终交付"]
dw = wb.create_sheet("最终交付")
dw.append(["创新点交付：Indian Pines 数据预处理与采样优化（类别平衡采样 + z-score 协同，关闭动态融合）"])
dw.append([])
dw.append(["口径", "基线 OA", "创新配置 D OA", "ΔOA", "基线 AA", "创新 AA", "基线 Kappa", "创新 Kappa"])
rows = [
    ["seed42 单seed（与你基线同口径）", 80.88, 86.64, "+5.76", 70.67, 71.18, 0.5417, 0.6695],
    ["3-seed 均值（42/123/456）", 80.69, 88.77, "+8.08", 68.65, 75.01, 0.5936, 0.7253],
]
for r in rows:
    dw.append(r)
dw.append([])
dw.append(["基线来源：result_IP/dynamic_seed42_20ep/Indian/Indian_Summary.txt（seed42，固定最后cp）"])
dw.append(["评测协议：固定最后 checkpoint，多种子均值，不挑最优；G6 单类组不计"])
dw.append(["亮点：最差组 G3 从 65.55(seed42基线) → 86.86(3-seed均值)；归因已对账（平衡+4.40/关融合+2.59/z-score协同+3.12/负样本-4.23）"])
dw.append(["LongKou/WHHL：z-score、压缩文本属性均为负贡献已回滚登记，不列为交付创新点"])

for c in dw[3]:
    c.font = Font(bold=True)
dw.cell(row=1, column=1).font = Font(bold=True, size=12)
dw.cell(row=1, column=1).alignment = Alignment(horizontal="left")
for row in dw.iter_rows(min_row=4, max_row=5):
    for c in row:
        c.alignment = Alignment(horizontal="center")

wb.save(XLSX)
print("updated:", XLSX)
