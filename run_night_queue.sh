#!/bin/bash
# 夜间总队列 —— 无人值守，按顺序自动接续，单步失败不中断
#
# 执行顺序：
#   PHASE 1  LongKou A/B，seed 42      （核心结果，约 5.5 h）
#   PHASE 2  IP λ 裁决 0.4 / 0.6，seed 42（约 1.2 h）
#   PHASE 3  LongKou A/B，seed 123     （稳健性，约 5.5 h）
#
# 每步内部均已实现「折级断点续跑」；整个队列也可被中断后重跑本脚本续上。
# 状态：runs/STATUS.txt 追加；完成标记：runs/.done_night
set -u
cd /root/autodl-tmp/SpectralZero || exit 1
mkdir -p runs
status () { echo "$(date '+%F %T') | QUEUE | $*" >> runs/STATUS.txt; }

status "===== 夜间队列启动 ====="

status "PHASE 1 开始：LongKou A/B seed 42"
bash run_lk_ab.sh 42 >> runs/_phase1.log 2>&1
status "PHASE 1 结束 exit=$?"

status "PHASE 2 开始：IP λ 裁决 seed 42"
bash run_ip_lambda.sh 42 >> runs/_phase2.log 2>&1
status "PHASE 2 结束 exit=$?"

status "PHASE 3 开始：LongKou A/B seed 123"
bash run_lk_ab.sh 123 >> runs/_phase3.log 2>&1
status "PHASE 3 结束 exit=$?"

echo "$(date '+%F %T')" > runs/.done_night
status "===== 夜间队列全部结束 ====="
