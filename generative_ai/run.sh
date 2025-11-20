#!/bin/bash

# =====================================
# 批量实验参数范围
# =====================================

# 隐空间大小
latent_size_list=("4" "8" "16" "32")

# 图像噪声概率
noise_list=("0.1" "0.3" "0.7" "0.9")

# kl 权重
kl_weights=("0.01" "0.1" "0.5")

# ========================================================
# 其他变量（你需要根据 .bat 里的内容补齐）
# 例如：
# grad_weights=("0.1" "1.0" "2.0")
# OUTROOT="outputs"
# ========================================================

# TODO：根据你的完整 .bat 内容补齐这里的变量
# grad_weights=( ... )
# OUTROOT="..."

echo "开始批量实验..."

# 三层嵌套循环
for T in "${latent_size_list[@]}"; do
    for N in "${noise_list[@]}"; do
        for G in "${kl_weights[@]}"; do

            echo "-----------------------------------------"
            echo "训练 step2_train_vae.py"
            echo "noise_prob=$N   latent_size=$T   w_grad=$G"
            echo "输出目录: $OUTROOT"
            echo "-----------------------------------------"

            python step2_train_vae.py \
                --noise_prob "$N" \
                --latent_size "$T" \
                --w_grad "$G" \
                --fig_root "$OUTROOT"

        done
    done
done

echo "全部任务完成！"
