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

def plot_latent_exploration(
    run_dir,
    z_path,
    scores=None,
    coverages=None,
    cand_clouds=None,
    cand_values=None,
    value_name="H",
    colorize_candidates=False,
    show_step_labels=True,
    max_step_labels=30,
):
    """
    路径 +（可选）候选云 + 自动高亮 best candidate（来自 z_path）

    约定：
    - z_path[t+1] == cand_clouds[t] 中被选中的 best
    """

    os.makedirs(run_dir, exist_ok=True)

    Zpath = np.asarray(z_path)          # (T+1, D)
    T_plus_1, D = Zpath.shape
    T = T_plus_1 - 1

    if cand_clouds is not None:
        if len(cand_clouds) != T:
            raise ValueError(f"cand_clouds length must be T={T}, got {len(cand_clouds)}")

    if colorize_candidates:
        if cand_values is None or len(cand_values) != T:
            raise ValueError("colorize_candidates=True requires cand_values with length T")

    # ---------- PCA basis (path + all clouds) ----------
    if cand_clouds is None:
        Z_all = Zpath
    else:
        Z_all = [Zpath]
        for C in cand_clouds:
            Z_all.append(np.asarray(C))
        Z_all = np.concatenate(Z_all, axis=0)

    mean = Z_all.mean(axis=0)
    Zc = Z_all - mean
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    W = Vt[:2].T

    Zp2 = (Zpath - mean) @ W   # (T+1, 2)

    # ---------- main figure ----------
    plt.figure(figsize=(7.5, 6.5))
    mappable = None

    # ----- candidate clouds -----
    if cand_clouds is not None:
        if colorize_candidates:
            vmin = min(float(np.min(v)) for v in cand_values)
            vmax = max(float(np.max(v)) for v in cand_values)

        for t, C in enumerate(cand_clouds):
            C = np.asarray(C)
            C2 = (C - mean) @ W

            if colorize_candidates:
                vals = np.asarray(cand_values[t])
                sc = plt.scatter(
                    C2[:, 0], C2[:, 1],
                    c=vals, vmin=vmin, vmax=vmax,
                    s=10, alpha=0.25, linewidths=0
                )
                mappable = sc
            else:
                plt.scatter(
                    C2[:, 0], C2[:, 1],
                    s=10, alpha=0.18, color="gray", linewidths=0
                )

            # ---- 自动高亮 best：z_path[t+1] ----
            z_best = Zpath[t + 1]              # (D,)
            z_best2 = (z_best - mean) @ W      # (2,)
            plt.scatter(
                z_best2[0], z_best2[1],
                s=90, marker="*", color="gold",
                edgecolors="black", linewidths=0.7, zorder=5
            )

    # ----- accepted path -----
    plt.plot(
        Zp2[:, 0], Zp2[:, 1],
        "-o", linewidth=1.6, markersize=4,
        label="accepted path", zorder=4
    )

    for i in range(len(Zp2) - 1):
        plt.annotate(
            "",
            xy=(Zp2[i + 1, 0], Zp2[i + 1, 1]),
            xytext=(Zp2[i, 0], Zp2[i, 1]),
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    if show_step_labels:
        stride = max(1, T_plus_1 // max_step_labels)
        for i in range(0, T_plus_1, stride):
            plt.text(Zp2[i, 0], Zp2[i, 1], str(i), fontsize=8)

    plt.title("Latent exploration (PCA 2D) with candidate clouds")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if mappable is not None:
        cb = plt.colorbar(mappable, fraction=0.046, pad=0.04)
        cb.set_label(value_name)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "latent_exploration_pca2d.png"), dpi=220)
    plt.close()

    # ---------- auxiliary plots ----------
    steps = np.arange(T_plus_1)

    plt.figure(figsize=(7, 4))
    plt.plot(steps, np.linalg.norm(Zpath, axis=1))
    plt.title("||z|| over accepted steps")
    plt.xlabel("step")
    plt.ylabel("||z||")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "latent_norm_over_steps.png"), dpi=220)
    plt.close()

    if scores is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(steps, scores)
        plt.title("score over accepted steps")
        plt.xlabel("step")
        plt.ylabel("score")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "score_over_steps.png"), dpi=220)
        plt.close()

    if coverages is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(steps, coverages)
        plt.title("coverage over accepted steps")
        plt.xlabel("step")
        plt.ylabel("coverage")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "coverage_over_steps.png"), dpi=220)
        plt.close()

# ====== CONFIG ======
MODEL_ROOT = "results/VAEv12_MDN_lat=16_var_scale=0.1K=16_beta=0.01_warm=0.1_gamma=0.001_warm=0.1_phy_weight=0.0_phy_alpha=1_phy_beta=1_scale_weight=0.1/"
CKPT_PATH  = os.path.join(MODEL_ROOT, "ckpt", "best.pt")
OUT_DIR    = os.path.join(MODEL_ROOT, "heuristic_search")

IMAGE_SIZE = (48, 48)
# SEED = 0

VAR_SCALE = 0.5
STEPS = 100

# --- naive random walk params ---
RW_SIGMA = 0.25          # 扰动幅度
NUM_CAND = 48            # 每步试多少个候选

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
            pi_s, mu_s, log_sigma_s, var_scale=VAR_SCALE
        )

    recon = recon.cpu().detach().numpy()[0, 0]
    z = z.cpu().detach().numpy()[0]
    y_pred_s = theta_hat_s.detach().cpu().numpy()[0]
    conf_s = conf_param_s.detach().cpu().numpy()[0]
    conf_global_s = conf_global_s.detach().cpu().numpy()[0]

    _, metrics, scores = generate_analysis_figure(recon)
    s = scores["empirical_score"]
    c = metrics["dendrite_coverage"]
    t = y_pred_s[0]
    save_step(run_dir, 0, recon, z, y_pred_s, c, s)

    z_path = [z.copy()]
    cand_clouds = []
    cand_H = []
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
        z_cands = []
        H_list = []
        for _ in range(NUM_CAND):

            dz = np.random.randn(*z.shape).astype(np.float32) * RW_SIGMA
            z_cand = z + dz
            z_cand_tensor = torch.from_numpy(z_cand).unsqueeze(0).to(device)
            with torch.no_grad():
                recon_cand, (theta_hat_s_cand, conf_param_s_cand, conf_global_s_cand, modes_s_cand) = \
                    model.inference(z_cand_tensor, var_scale=VAR_SCALE)

            recon_cand = recon_cand.cpu().detach().numpy()[0, 0]
            y_pred_s_cand = theta_hat_s_cand.detach().cpu().numpy()[0]
            conf_s_cand = conf_param_s_cand.detach().cpu().numpy()[0]
            conf_global_s_cand = conf_global_s_cand.detach().cpu().numpy()[0]

            _, metrics_cand, scores_cand = generate_analysis_figure(recon_cand)
            t_cand = y_pred_s_cand[0]
            s_cand = scores_cand["empirical_score"]
            c_cand = metrics_cand["dendrite_coverage"]

            # 总结全局匹配度
            H = - np.linalg.norm(y_pred_s_cand - y_pred_s) - (s_cand - s)

            # save cands
            z_cands.append(z_cand.copy())
            H_list.append(float(H))

            if c_cand < c or t_cand < t:
                print(f"    [Reject]c_cand={c_cand:.3f}<c={c:.3f} or t_cand={t_cand:.3f}<t={t:.3f}")
                continue

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
        t = best_params[0]
        y_pred_s = best_params

        save_step(run_dir, step, best_img, best_z, best_params, best_coverage, best_score)

        z_path.append(z.copy())
        score_path.append(float(s))
        coverage_path.append(float(c))

        cand_clouds.append(np.stack(z_cands, axis=0))  # (NUM_CAND, D)
        cand_H.append(np.array(H_list, dtype=float))  # (NUM_CAND,)

    plot_latent_exploration(
        run_dir,
        z_path,
        scores=score_path,
        coverages=coverage_path,
        cand_clouds=cand_clouds,
        cand_values=cand_H,
        colorize_candidates=True
    )
    print("Done. Saved to:", run_dir)


if __name__ == "__main__":
    main()
