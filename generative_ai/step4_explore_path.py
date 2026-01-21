import json
import os
import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataloader import smooth_scale
from src.modelv11 import mdn_point_and_confidence

def get_init_tensor(image_size=(64, 64)):

    arr = np.load("data/case_000/npy_files/0.000000.npy")  # shape (H, W, 3)

    # assert arr.max() <= 5 and arr.min() >= -5, "Inappropriate values occured"
    arr = cv2.resize(arr, image_size)
    tensor_t = torch.from_numpy(arr).float().permute(2, 0, 1)
    tensor_t = smooth_scale(tensor_t)

    # build control variable
    c = [0.0]

    # find meta
    meta_path = "data/case_000/case_000.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    c += meta

    return tensor_t


# ====== CONFIG ======
MODEL_ROOT = "/home/xtanghao/THPycharm/dendrites_pfm_vae/tmp/results_expV9/V9_lat=8_beta=0.1_warm=0.3_ctr=2.0_smooth=0.05_time=20260105_071004/"
CKPT_PATH  = os.path.join(MODEL_ROOT, "ckpt", "best.pt")
OUT_DIR    = os.path.join(MODEL_ROOT, "heuristic_search_continuous")

IMAGE_SIZE = 64
VAR_SCALE = 1
TOPK_MODES = 3

STEPS = 200
SEED = 0

# --- naive random walk params ---
RW_SIGMA = 0.25          # 扰动幅度（大一点就更容易崩）
NUM_CAND = 32            # 每步试多少个候选

# --- safe guards ---
PRIOR_L2_W = 0.03        # 惩罚 ||z||^2
MAX_NORM   = 3.5         # 限制 z 的范数（粗暴但有效）
PIX_CLIP_MIN = -0.2      # 用于检测异常 decode
PIX_CLIP_MAX =  1.2

def save_step(out_dir, step, img, z, t, fd, score):
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="coolwarm")
    plt.colorbar(fraction=0.046)
    plt.title(f"step={step}  score={score:.3f}\nt={t:.3f}, FD={fd:.3f}, ||z||={np.linalg.norm(z):.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"step_{step:03d}.png"), dpi=200)
    plt.close()


def main():

    run_dir = os.path.join(OUT_DIR, str(time.time()))
    os.makedirs(run_dir, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.eval()

    # === 初始化：直接从 prior 采样 ===
    # 这一步本身就可能生成很差的图（取决于你的 VAE prior match 是否好）
    with torch.no_grad():
        recon, mu_q, logvar_q, (pi_s, mu_s, log_sigma_s), z = model(get_init_tensor().unsqueeze(0).to(device))

    for step in range(0, STEPS + 1):

        # ---- stochastic prediction + confidence (from sampled z) ----
        theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
            pi_s, mu_s, log_sigma_s, var_scale=VAR_SCALE, topk=TOPK_MODES
        )
        y_pred_s = theta_hat_s.detach().cpu().numpy().tolist()
        conf_s = conf_param_s.detach().cpu().numpy().tolist()
        conf_global_s = conf_global_s.detach().cpu().numpy().tolist()

        best = None
        best_score = -1e18
        best_img = None
        best_t = None
        best_fd = None

        # 生成候选
        for _ in range(NUM_CAND):

            # 记录


            dz = np.random.randn(*z.shape).astype(np.float32) * RW_SIGMA
            z_cand = z + dz

            # 避免探索太远
            nrm = np.linalg.norm(z_cand)
            if nrm > MAX_NORM:
                z_cand = z_cand / (nrm + 1e-12) * MAX_NORM

            img_cand = decode_single_channel(model, device, z_cand)

            t_cand = predict_time(model, device, z_cand)
            fd_cand = fractal_dimension(img_cand)

            s = heuristic_score(img_cand)

            if s > best_score:
                best_score = s
                best = z_cand
                best_img = img_cand
                best_t = t_cand
                best_fd = fd_cand

        if best is None:
            print("[Stop] no valid candidate (all rejected).")
            break

        z = best

        # # safe: 定期投影回 dataset latent 的最近邻附近（非常有效）
        # if MODE == "safe" and USE_PROJ and (step % PROJ_EVERY == 0):
        #     _, idx = knn.kneighbors(z.reshape(1, -1))
        #     z_nn = Z_data[idx[0, 0]]
        #     z = PROJ_ALPHA * z + (1 - PROJ_ALPHA) * z_nn

        save_step(run_dir, step, best_img, z, best_t, best_fd, best_score)

    print("Done. Saved to:", run_dir)


if __name__ == "__main__":
    main()
