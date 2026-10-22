#!/bin/bash
# 后续队列 —— 等 Phase 3（.done_night）结束后自动接上
#   1) LongKou 温和平衡采样 sampler_power=0.5，seed 42（3 折，约 2.75 h）
#   2) Indian 类平衡采样 sampler_power=1.0，seed 42（6 折，约 0.6 h）
#
# 为什么要等：当前 Phase 3 正在用旧 main.py 跑，必须等它结束再切换代码，
# 否则同一批实验用的代码不一致。
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
mkdir -p runs
status () { echo "$(date '+%F %T') | FOLLOWUP | $*" >> runs/STATUS.txt; }

if [ -f runs/.done_followup ]; then
  status "已跑过（.done_followup 存在），退出"; exit 0
fi

status "等待 Phase 3 完成（runs/.done_night）"
while [ ! -f runs/.done_night ]; do sleep 60; done
status "Phase 3 已完成"

# 切换代码：默认行为与旧版完全一致（power=1.0），仅新增 --sampler_power
if [ -f main_power.py ]; then
  cp main.py main_pre_power.bak
  cp main_power.py main.py
  { echo "switched=$(date '+%F %T')"; md5sum main.py; } >> runs/STATUS.txt
  status "main.py 已切换为 sampler_power 版"
else
  status "!! 未找到 main_power.py，中止"; exit 2
fi

bash run_lk_mild.sh 42 >> runs/_followup_lkmild.log 2>&1
status "LongKou mild 结束 exit=$?"

bash run_ip_bal.sh 42 >> runs/_followup_ipbal.log 2>&1
status "Indian bal 结束 exit=$?"

echo "$(date '+%F %T')" > runs/.done_followup
status "后续队列全部结束"
