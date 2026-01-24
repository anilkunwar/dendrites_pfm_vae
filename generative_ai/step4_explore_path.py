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

def plot_latent_path(run_dir, z_path, scores=None, coverages=None):
    """
    z_path: (T, D)
    画：PCA投影到2D的隐空间轨迹 + 箭头 + 步号
    """
    Z = np.stack(z_path, axis=0)  # (T, D)
    Zc = Z - Z.mean(axis=0, keepdims=True)

    # --- PCA to 2D (no sklearn needed) ---
    # SVD: Zc = U S V^T, principal axes in V
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    Z2 = Zc @ Vt[:2].T  # (T, 2)

    plt.figure(figsize=(7, 6))
    plt.plot(Z2[:, 0], Z2[:, 1], marker="o", linewidth=1)

    # arrows
    for i in range(len(Z2) - 1):
        plt.annotate(
            "",
            xy=(Z2[i + 1, 0], Z2[i + 1, 1]),
            xytext=(Z2[i, 0], Z2[i, 1]),
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    # step labels (太密就只标一部分)
    T = len(Z2)
    stride = max(1, T // 30)
    for i in range(0, T, stride):
        plt.text(Z2[i, 0], Z2[i, 1], str(i), fontsize=8)

    plt.title("Latent exploration path (PCA 2D projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "latent_path_pca2d.png"), dpi=200)
    plt.close()

    # 可选：再画 norm / score / coverage 随 step
    steps = np.arange(T)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, np.linalg.norm(Z, axis=1))
    plt.title("||z|| over steps")
    plt.xlabel("step")
    plt.ylabel("||z||")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "latent_norm_over_steps.png"), dpi=200)
    plt.close()

    if scores is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(steps, scores)
        plt.title("empirical_score over steps")
        plt.xlabel("step")
        plt.ylabel("score")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "score_over_steps.png"), dpi=200)
        plt.close()

    if coverages is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(steps, coverages)
        plt.title("dendrite_coverage over steps")
        plt.xlabel("step")
        plt.ylabel("coverage")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "coverage_over_steps.png"), dpi=200)
        plt.close()

# ====== CONFIG ======
MODEL_ROOT = "results/VAEv12_MDN_lat=16_var_scale=0.1K=16_beta=0.01_warm=0.1_gamma=0.001_warm=0.1_phy_weight=0.0_phy_alpha=1_phy_beta=1_scale_weight=0.1/"
CKPT_PATH  = os.path.join(MODEL_ROOT, "ckpt", "best.pt")
OUT_DIR    = os.path.join(MODEL_ROOT, "heuristic_search")

IMAGE_SIZE = (48, 48)
# SEED = 0

VAR_SCALE = 1
TOPK_MODES = 3
STEPS = 200

# --- naive random walk params ---
RW_SIGMA = 0.25          # 扰动幅度
NUM_CAND = 32            # 每步试多少个候选

def save_step(out_dir, step, img, z, params, coverage, score):
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="coolwarm")
    plt.colorbar(fraction=0.046)
    plt.title(f"step={step} score={score:.3f} t={params[0]:.3f}, Coverage={coverage:.3f}, ||z||={np.linalg.norm(z):.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"step_{step:03d}.png"), dpi=200)
    plt.close()
    print(f"step={step} score={score:.3f} t={params[0]:.3f}, Coverage={coverage:.3f}, ||z||={np.linalg.norm(z):.2f}")

def main():

    run_dir = os.path.join(OUT_DIR, str(time.time()))
    os.makedirs(run_dir, exist_ok=True)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)

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

    z_path = [z.copy()]
    score_path = [float(s)]
    coverage_path = [float(c)]
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

            if c_cand < c:
                print(f"    [Reject]c_cand={c_cand:.3f}<c={c:.3f}")
                continue

            # 总结全局匹配度
            H = - np.linalg.norm(y_pred_s_cand - y_pred_s) - (s_cand - s)

            if H > best_H_score:
                best_H_score = H
                best_score = s_cand
                best_z = z_cand
                best_img = recon_cand
                best_params = y_pred_s_cand
                best_coverage = c_cand

        if best_z is None:
            print("[Stop] no valid candidate (all rejected).")
            break
        else:
            print(f"[Next] find best candidate with H score={best_H_score:.2f}")

        z = best_z
        s = best_score
        c = best_coverage
        y_pred_s = best_params
        save_step(run_dir, step, best_img, best_z, best_params, best_coverage, best_score)

        z_path.append(z.copy())
        score_path.append(float(s))
        coverage_path.append(float(c))

    plot_latent_path(run_dir, z_path, scores=score_path, coverages=coverage_path)
    print("Done. Saved to:", run_dir)


if __name__ == "__main__":
    main()
