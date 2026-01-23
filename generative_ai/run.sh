#!/bin/bash

# =====================================
# 只 sweep 物理相场相关参数
# phy_weight × phy_alpha × phy_beta
# =====================================

phy_weight=("0.001" "0.01" "0.05")
phy_alpha=("0.5" "1.0" "5.0" "10.0")
phy_beta=("0.5" "1.0" "5.0" "10.0")

# 最大并行任务数
MAX_JOBS=1

echo "开始 sweep 相场物理参数（并行=${MAX_JOBS}）"
echo "Sweep: phy_weight × phy_alpha × phy_beta"

for W in "${phy_weight[@]}"; do
    for A in "${phy_alpha[@]}"; do
        for B in "${phy_beta[@]}"; do

            echo "启动任务: phy_weight=$W phy_alpha=$A phy_beta=$B"

            python step2_train_vaev12.py \
                --phy_weight "$W" \
                --phy_alpha "$A" \
                --phy_beta "$B" &

            # 控制并行数
            while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
                wait -n
            done

        done
    done
done

# 等待所有后台任务完成
wait
echo "全部相场参数 sweep 完成 ✅"
