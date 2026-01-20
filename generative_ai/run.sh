#!/usr/bin/env bash
set -euo pipefail

# =========================
# User config (edit here)
# =========================
PYTHON_BIN="${PYTHON_BIN:-python}"                 # 或写成你的conda python绝对路径
TRAIN_PY="${TRAIN_PY:-train_vae_mdn.py}"           # 训练脚本路径（相对或绝对）

# 固定参数（按需改）
EPOCHS=200
BATCH_SIZE=128
LR=1e-4
SEED=0
IMAGE_SIZE="(3,64,64)"
LATENT_SIZE=32
HIDDEN_DIM=128
NUM_PARAMS=15
MDN_COMPONENTS=16
MDN_HIDDEN=256
PATIENCE=100
SAVE_ROOT="results"

# GPU（可选）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 日志目录
RUNS_DIR="results"
LOG_DIR="${RUNS_DIR}/logs"
mkdir -p "${LOG_DIR}"

# =========================
# Sweep params (focus on these)
# =========================
# baseline: beta=2.0, warm=0.3, ctr=0.8, smooth=2.0, var_scale=1.0  :contentReference[oaicite:2]{index=2}
BETAS=(1.0 2.0 3.0)
BETA_WARMUP_RATIOS=(0.2 0.3 0.5)
CTR_WEIGHTS=(0.4 0.8 1.2)
SMOOTH_WEIGHTS=(0.5 1.0 2.0)
VAR_SCALES=(0.5 1.0 2.0)

# 如果你想先小规模试跑，把数组改小（例如每个只放2个值）

timestamp="$(date +%Y%m%d_%H%M%S)"

echo "=== Sweep start: ${timestamp} ==="
echo "Python: ${PYTHON_BIN}"
echo "Script: ${TRAIN_PY}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo

# =========================
# Run loop
# =========================
run_one () {
  local beta="$1"
  local warm="$2"
  local ctr="$3"
  local smooth="$4"
  local vscale="$5"

  local tag="beta${beta}_warm${warm}_ctr${ctr}_sm${smooth}_vs${vscale}"
  local logfile="${LOG_DIR}/${timestamp}_${tag}.log"

  echo ">>> RUN ${tag}"
  echo "    log: ${logfile}"

  # -u: unbuffered 输出，方便实时看日志
  ${PYTHON_BIN} -u "${TRAIN_PY}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --image_size "${IMAGE_SIZE}" \
    --latent_size "${LATENT_SIZE}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --num_params "${NUM_PARAMS}" \
    --mdn_components "${MDN_COMPONENTS}" \
    --mdn_hidden "${MDN_HIDDEN}" \
    --beta "${beta}" \
    --beta_warmup_ratio "${warm}" \
    --ctr_weight "${ctr}" \
    --smooth_weight "${smooth}" \
    --var_scale "${vscale}" \
    --patience "${PATIENCE}" \
    --save_root "${SAVE_ROOT}" \
    2>&1 | tee "${logfile}"

  echo "<<< DONE ${tag}"
  echo
}

# 建议：先跑 baseline，确保路径/数据集没问题
echo "=== Baseline sanity run ==="
run_one 2.0 0.3 0.8 2.0 1.0

# 网格扫描
echo "=== Grid sweep ==="
for beta in "${BETAS[@]}"; do
  for warm in "${BETA_WARMUP_RATIOS[@]}"; do
    for ctr in "${CTR_WEIGHTS[@]}"; do
      for smooth in "${SMOOTH_WEIGHTS[@]}"; do
        for vscale in "${VAR_SCALES[@]}"; do
          run_one "${beta}" "${warm}" "${ctr}" "${smooth}" "${vscale}"
        done
      done
    done
  done
done

echo "=== Sweep finished: ${timestamp} ==="
