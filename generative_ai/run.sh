#!/bin/bash

# =====================================
# 参数空间（当前模型使用）
# 只搜索：beta / beta_warmup_ratio / ctr_weight / smooth_weight / var_scale
# 其它任何参数不动
# =====================================

beta=("0.1" "1.0" "5.0")
beta_warmup_ratio=("0.25" "0.5" "0.75")
ctr_weight=("0.5" "1.0" "5.0")
smooth_weight=("0.05" "1.0" "5.0")
var_scale=("0.5" "1.0" "5.0")   # ✅ 新增：置信度尺度（只影响你输出置信度的映射）

# 最大并行数
MAX_JOBS=2

echo "开始批量实验（并行=${MAX_JOBS}）..."
echo "Sweep parameters: beta × beta_warmup_ratio × ctr_weight × smooth_weight × var_scale"

for B in "${beta[@]}"; do
    for W in "${beta_warmup_ratio[@]}"; do
        for C in "${ctr_weight[@]}"; do
            for S in "${smooth_weight[@]}"; do
                for V in "${var_scale[@]}"; do

                    echo "启动任务: beta=$B warmup_ratio=$W ctr_weight=$C smooth_weight=$S var_scale=$V"

                    # ------------------------------ #
                    # 后台执行任务（并行）
                    # ------------------------------ #
                    python step2_train_vaev11.py \
                        --beta "$B" \
                        --beta_warmup_ratio "$W" \
                        --ctr_weight "$C" \
                        --smooth_weight "$S" \
                        --var_scale "$V" &

                    # 控制并行度
                    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
                        wait -n   # 等任意一个后台任务结束
                    done

                done
            done
        done
    done
done

# 等所有剩余任务完成
wait

echo "全部任务完成！"
