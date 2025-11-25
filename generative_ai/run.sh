#!/bin/bash

# =====================================
# 批量实验参数范围
# =====================================

# 隐空间大小
latent_size_list=("2" "4" "8" "16" "32" "64" "128")

# 图像噪声概率
noise_list=("0.1" "0.3" "0.7" "0.9")

# kl 权重
kl_weights=("0.01" "0.1" "0.5" "1.0" "2.0")

grad_weights=("0.01" "0.1" "0.5" "1.0")
#con_weights=("0.01" "0.1" "0.5" "1.0")

echo "开始批量实验..."

# 三层嵌套循环
for T in "${latent_size_list[@]}"; do
    for N in "${noise_list[@]}"; do
        for G in "${kl_weights[@]}"; do
            for W in "${grad_weights[@]}"; do
#              for K in "${con_weights[@]}"; do
                echo "-----------------------------------------"
                echo "训练 step2_train_vae.py"
                echo "noise_prob=$N latent_size=$T w_kl=$G  w_grad=$W"
                echo "-----------------------------------------"

                python step2_train_vae.py \
                    --noise_prob "$N" \
                    --latent_size "$T" \
                    --w_kl "$G" \
                    --w_grad "$W" \
#                    --w_con "$K"
#              done
            done
        done
    done
done

echo "全部任务完成！"
