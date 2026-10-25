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

# main.py 已内置 --sampler_power（6f69fed），不再切换源码：
# 原实现依赖未入库的 main_power.py（干净 checkout 上直接中止），且覆盖 main.py 后无法回滚。
if ! grep -q -- "--sampler_power" main.py; then
  status "!! main.py 不支持 --sampler_power，中止（需 6f69fed 或更新）"; exit 2
fi
status "main.py 已支持 --sampler_power，无需切换"

bash run_lk_mild.sh 42 >> runs/_followup_lkmild.log 2>&1
status "LongKou mild 结束 exit=$?"

bash run_ip_bal.sh 42 >> runs/_followup_ipbal.log 2>&1
status "Indian bal 结束 exit=$?"

echo "$(date '+%F %T')" > runs/.done_followup
status "后续队列全部结束"
