#!/bin/bash

# =====================================
# 参数空间（当前模型使用）
# =====================================

beta=("0.1" "1.0" "2.0")
beta_warmup_ratio=("0.2" "0.3" "0.4")
ctr_weight=("0.5" "1.0" "2.0")
smooth_weight=("0.05" "1.0" "2.0")

# 最大并行数
MAX_JOBS=2

echo "开始批量实验（并行=${MAX_JOBS}）..."
echo "Sweep parameters: beta × beta_warmup_ratio × ctr_weight × smooth_weight"

for B in "${beta[@]}"; do
    for W in "${beta_warmup_ratio[@]}"; do
        for C in "${ctr_weight[@]}"; do
            for S in "${smooth_weight[@]}"; do

                echo "启动任务: beta=$B warmup_ratio=$W ctr_weight=$C smooth_weight=$S"

                # ------------------------------ #
                # 后台执行任务（并行）
                # ------------------------------ #
                python step2_train_vaev9.py \
                    --beta "$B" \
                    --beta_warmup_ratio "$W" \
                    --ctr_weight "$C" \
                    --smooth_weight "$S" &

                # 控制并行度
                while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
                    wait -n   # 等任意一个后台任务结束
                done

            done
        done
    done
done

# 等所有剩余任务完成
wait

echo "全部任务完成！"
