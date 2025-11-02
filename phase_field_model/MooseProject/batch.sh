#!/bin/bash
# ==========================================================
# Author: Hao Tang (Nov 2025)
# Purpose: Batch run multiple MOOSE input files efficiently
# ==========================================================

# ===== Config =====
INPUT_DIR="generated_inputs"
LOG_DIR="logs"
MOOSE_EXEC="/home/xtanghao/MooseProject/newt"
NPROC_PER_JOB=4
MAX_PARALLEL=6

# =========================

mkdir -p "$LOG_DIR"

# 统计可用输入文件
FILES=(${INPUT_DIR}/case_*.i)
TOTAL=${#FILES[@]}

echo "---------------------------------------------"
echo " 🧩 Starting batch run of ${TOTAL} MOOSE cases"
echo "    Using ${NPROC_PER_JOB} cores per job"
echo "    Up to ${MAX_PARALLEL} parallel jobs"
echo "---------------------------------------------"

# 当前运行的任务数
running_jobs=0

for file in "${FILES[@]}"; do
    base=$(basename "$file" .i)
    logfile="${LOG_DIR}/${base}.out"

    echo "🚀 Launching ${base} ..."
    mpiexec -n ${NPROC_PER_JOB} ${MOOSE_EXEC} -i "$file" > "$logfile" 2>&1 &

    ((running_jobs++))

    # 如果已达到最大并行数，则等待有任务完成
    if (( running_jobs >= MAX_PARALLEL )); then
        wait -n    # 等待任意一个后台任务完成
        ((running_jobs--))
    fi
done

# 等待剩余任务完成
wait

echo "✅ All MOOSE simulations finished."
echo "Logs saved in: ${LOG_DIR}/"
