import json
import os
import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.evaluate_metrics import generate_analysis_figure
from src.dataloader import smooth_scale
from src.modelv11 import mdn_point_and_confidence

def get_init_tensor(image_size: tuple):

    arr = np.load("data/case_000/npy_files/0.000000.npy")  # shape (H, W, 3)

    arr = cv2.resize(arr, image_size)
    tensor_t = torch.from_numpy(arr).float().permute(2, 0, 1)
    tensor_t = smooth_scale(tensor_t)

    return tensor_t


# ====== CONFIG ======
MODEL_ROOT = "results/VAEv12_MDN_lat=16_var_scale=0.1K=16_beta=0.01_warm=0.1_gamma=0.001_warm=0.1_phy_weight=0.0_phy_alpha=1_phy_beta=1_scale_weight=0.1_time=20260124_055835/"
CKPT_PATH  = os.path.join(MODEL_ROOT, "ckpt", "best.pt")
OUT_DIR    = os.path.join(MODEL_ROOT, "heuristic_search")

IMAGE_SIZE = (48, 48)
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

def save_step(out_dir, step, img, z, params, coverage, score):
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="coolwarm")
    plt.colorbar(fraction=0.046)
    plt.title(f"step={step} score={score:.3f}\n t={params[0]:.3f}, Coverage={coverage:.3f}, ||z||={np.linalg.norm(z):.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"step_{step:03d}.png"), dpi=200)
    plt.close()
    print(f"step={step} score={score:.3f}\n t={params[0]:.3f}, Coverage={coverage:.3f}, ||z||={np.linalg.norm(z):.2f}")

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
        recon, mu_q, logvar_q, (pi_s, mu_s, log_sigma_s), z = model(get_init_tensor(IMAGE_SIZE).unsqueeze(0).to(device))
        # ---- stochastic prediction + confidence (from sampled z) ----
        theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
            pi_s, mu_s, log_sigma_s, var_scale=VAR_SCALE, topk=TOPK_MODES
        )

    recon = recon.cpu().detach().numpy()[0, 0]
    z = z.cpu().detach().numpy()[0]
    y_pred_s = theta_hat_s.detach().cpu().numpy()[0]
    conf_s = conf_param_s.detach().cpu().numpy()[0]
    conf_global_s = conf_global_s.detach().cpu().numpy()[0]

    _, metrics, scores = generate_analysis_figure(recon)
    s = scores["empirical_score"]
    c = metrics["dendrite_coverage"]
    save_step(run_dir, 0, recon, z, y_pred_s, c, s)

    for step in range(1, STEPS + 1):
        # 生成候选
        best_z = None
        best_H_score = -1e18
        best_score = -1e18
        best_img = None
        best_params = None
        best_coverage = None
        for _ in range(NUM_CAND):

            dz = np.random.randn(*z.shape).astype(np.float32) * RW_SIGMA
            z_cand = z + dz
            z_cand_tensor = torch.from_numpy(z_cand).unsqueeze(0).to(device)
            with torch.no_grad():
                recon_cand, (theta_hat_s_cand, conf_param_s_cand, conf_global_s_cand, modes_s_cand) = model.inference(z_cand_tensor)

            recon_cand = recon_cand.cpu().detach().numpy()[0, 0]
            y_pred_s_cand = theta_hat_s_cand.detach().cpu().numpy()[0]
            conf_s_cand = conf_param_s_cand.detach().cpu().numpy()[0]
            conf_global_s_cand = conf_global_s_cand.detach().cpu().numpy()[0]

            _, metrics_cand, scores_cand = generate_analysis_figure(recon_cand)
            s_cand = scores_cand["empirical_score"]
            c_cand = metrics_cand["dendrite_coverage"]

            # 总结全局匹配度
            H = - np.linalg.norm(y_pred_s_cand - y_pred_s) - (s_cand - s)

            if H > best_H_score and c_cand > c:
                best_H_score = H
                best_score = s
                best_z = z_cand
                best_img = recon
                best_params = y_pred_s
                best_coverage = c

        if best_z is None:
            print("[Stop] no valid candidate (all rejected).")
            break
        else:
            print(f"[Next] find best candidate with H score={best_H_score:.2f}")

        z = best_z
        save_step(run_dir, 0, best_img, best_z, best_params, best_coverage, best_score)

    print("Done. Saved to:", run_dir)


if __name__ == "__main__":
    main()
