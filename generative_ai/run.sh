#!/bin/bash



# =====================================
# 批量实验参数范围
# =====================================

## 隐空间大小
#latent_size_list=("256")
#
## 图像噪声概率
#noise_list=("0.8")

# kl 权重
beta_start=("0.01" "0.1" "0.5" "1.0")

grad_weights=("0.01" "0.1" "0.5" "1.0")

component_num=("3" "15" "32" "48")

scale_weight=("1" "0.5" "1.5" "0.25" "0.1")

echo "开始批量实验..."

# 三层嵌套循环
for T in "${beta_start[@]}"; do
    for N in "${grad_weights[@]}"; do
        for G in "${component_num[@]}"; do
            for W in "${scale_weight[@]}"; do
                echo "-----------------------------------------"
                echo "训练 step2_train_vaev2.py"
                echo "grad_weights=$N beta_start=$T component_num=$G  scale_weight=$W"
                echo "-----------------------------------------"

                python step2_train_vaev2.py \
                    --w_phy "$N" \
                    --beta_start "$T" \
                    --beta_end 1 \
                    --n_components "$G" \
                    --scale_weight "$W"
            done
        done
    done
done

echo "全部任务完成！"
