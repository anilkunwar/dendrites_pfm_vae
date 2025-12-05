#!/bin/bash

# =====================================
# 参数空间
# =====================================

latent_size=("4" "8" "16" "32" "128")
component_num=("16" "32" "48" "64")
scale_weight=("1" "0.5" "1.5" "0.25")

# 最大并行数
MAX_JOBS=2

echo "开始批量实验（并行=${MAX_JOBS}）..."

#for T in "${grad_weights[@]}"; do
    for N in "${latent_size[@]}"; do
        for G in "${component_num[@]}"; do
            for W in "${scale_weight[@]}"; do

                echo "启动任务: latent_size=$N component_num=$G scale_weight=$W"

                # ------------------------------ #
                # 后台执行任务（并行）
                # ------------------------------ #
                python step2_train_vaev2.py \
                    --latent_size "$N" \
                    --n_components "$G" \
                    --scale_weight "$W" &

                # 控制并行度为 2
                while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
                    wait -n   # 等任意一个后台任务结束
                done

            done
        done
    done
#done

# 等所有剩余任务完成
wait

echo "全部任务完成！"
