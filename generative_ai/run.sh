#!/bin/bash
set -euo pipefail

# ============================================================
# 随机（但在合理范围内）生成一组超参数并调用训练脚本
# - 训练脚本会自动保存 run_args.json 到实验目录
# ============================================================

PY_SCRIPT="step2_train_vae.py"

# 运行次数 & 最大并行数（可按需修改）
N_RUNS=50
MAX_JOBS=1

# ---------- 可调的“合理范围” ----------
epochs=(2000)
batch_size=(512)
#latent_size=(8 16 32)
#mdn_components=(8 16 32)
#mdn_hidden=(32 64 128)

beta=(0.001 0.003 0.005 0.01 0.02 0.03 0.1)
gamma=(0.001 0.003 0.005 0.01 0.02 0.03 0.1)
beta_warm=(0.05 0.1 0.15 0.2 0.25 0.3 0.5)
gamma_warm=(0.05 0.1 0.15 0.2 0.25 0.3 0.5)

# 物理相场正则：权重可包含 0（关闭）
phy_weight=(0 0.0001 0.0005 0.001 0.003 0.005 0.01)
phy_alpha=(0.5 1.0 2.0 5.0 10.0)
phy_beta=(0.5 1.0 2.0 5.0 10.0)

scale_weight=(0.1 0.2 0.3 0.5 0.8 1.0 2.0)
var_scale=(0.0001 0.0003 0.001 0.003 0.01 0.1 1.0)
lr=(0.00005 0.00008 0.0001 0.00015 0.0002 0.0003 0.0005)

patience=(50 80 100 150)
seed_max=100000

# ---------- 工具函数 ----------
pick() {
  local -n arr=$1
  echo "${arr[$RANDOM % ${#arr[@]}]}"
}

rand_seed() {
  echo $((RANDOM % seed_max))
}

echo "将运行 $N_RUNS 次随机超参数实验（并行=$MAX_JOBS）"
echo "脚本: $PY_SCRIPT"
echo

for ((i=1; i<=N_RUNS; i++)); do
  E=$(pick epochs)
  BS=$(pick batch_size)
  LR=$(pick lr)
#  LAT=$(pick latent_size)
#  K=$(pick mdn_components)
#  MH=$(pick mdn_hidden)

  BETA=$(pick beta)
  BW=$(pick beta_warm)
  GAMMA=$(pick gamma)
  GW=$(pick gamma_warm)

  PW=$(pick phy_weight)
  PA=$(pick phy_alpha)
  PB=$(pick phy_beta)

  SW=$(pick scale_weight)
  VS=$(pick var_scale)

  PAT=$(pick patience)
  SEED=$(rand_seed)

#  echo "[$i/$N_RUNS] epochs=$E bs=$BS lr=$LR lat=$LAT K=$K mdn_hidden=$MH beta=$BETA bw=$BW gamma=$GAMMA gw=$GW phy_w=$PW a=$PA b=$PB sw=$SW var=$VS pat=$PAT seed=$SEED"

  python "$PY_SCRIPT" \
    --epochs "$E" \
    --batch_size "$BS" \
    --lr "$LR" \
#    --latent_size "$LAT" \
#    --mdn_components "$K" \
#    --mdn_hidden "$MH" \
    --beta "$BETA" \
    --beta_warmup_ratio "$BW" \
    --gamma "$GAMMA" \
    --gamma_warmup_ratio "$GW" \
    --phy_weight "$PW" \
    --phy_alpha "$PA" \
    --phy_beta "$PB" \
    --scale_weight "$SW" \
    --var_scale "$VS" \
    --patience "$PAT" \
    --seed "$SEED" &

  # 控制并行数
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done
done

wait
echo
echo "全部随机实验运行完成 ✅"
