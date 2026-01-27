import csv
import json
import os
import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.modelv11 import postprocess_image, mdn_point_and_confidence
from src.evaluate_metrics import generate_analysis_figure
from src.dataloader import smooth_scale, inv_smooth_scale
from src.visualizer import plot_line_evolution


def get_init_tensor(image_size: tuple):
    """
    生成一个 50x50 的初始图像：
    - 左侧 5 列为 1
    - 其余为 0
    - 扩展为 3 通道
    - resize 到 image_size
    - 做 smooth_scale
    """

    # ---- step 1: generate 50x50 base image ----
    base_eta = np.zeros((50, 50), dtype=np.float32)
    base_eta[:, :5] = 1.0   # 左侧 5 列为 1

    base_c = np.zeros((50, 50), dtype=np.float32)
    base_c[:, :5] = 0.2
    base_c[:, 5:] = 0.8

    base_p = np.zeros((50, 50), dtype=np.float32)

    # ---- step 2: expand to 3 channels ----
    arr = np.stack([base_eta, base_c, base_p], axis=-1)  # (50, 50, 3)

    # ---- step 3: resize ----
    arr = cv2.resize(arr, image_size, interpolation=cv2.INTER_LINEAR)

    # ---- step 4: to tensor + smooth_scale ----
    tensor_t = torch.from_numpy(arr).float().permute(2, 0, 1)  # (3, H, W)
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

    # plt.title("Latent exploration (PCA 2D) with candidate clouds")
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

    plot_line_evolution(steps, np.linalg.norm(Zpath, axis=1), xlabel="step", ylabel="||z||", save_path=os.path.join(run_dir, "latent_norm_over_steps.png"))

    if scores is not None:
        plot_line_evolution(steps, scores, xlabel="step", ylabel="score",
                            save_path=os.path.join(run_dir, "score_over_steps.png"))

    if coverages is not None:
        plot_line_evolution(steps, coverages, xlabel="step", ylabel="dendrite coverage",
                            save_path=os.path.join(run_dir, "coverage_over_steps.png"))

# ====== CONFIG ======
MODEL_ROOT = "results/final_model"
CKPT_PATH  = os.path.join(MODEL_ROOT, "ckpt", "best.pt")
OUT_DIR    = os.path.join(MODEL_ROOT, "heuristic_search")

IMAGE_SIZE = (48, 48)
STRICT = False

VAR_SCALE = 1
STEPS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_step(out_dir, step, img, z, params, coverage, score, hopping_strength):
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="coolwarm")
    # plt.colorbar(fraction=0.046)
    # plt.title(f"step={step} score={score:.3f} t={params[0]:.3f}, Coverage={coverage:.3f}, ||z||={np.linalg.norm(z):.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"step_{step:03d}.png"), dpi=200)
    plt.close()

    # ---- CSV recording ----
    csv_path = os.path.join(out_dir, "params_recording.csv")
    header = [
        "step", "score", "coverage", "hopping_strength",
        "t", "POT_LEFT", "fo", "Al", "Bl", "Cl", "As", "Bs", "Cs",
        "cleq", "cseq", "L1o", "L2o", "ko", "Noise"
    ]

    params = np.asarray(params).reshape(-1)
    if params.size != 15:
        raise ValueError(f"Expected params length=15 (t..Noise), got {params.size}: {params}")

    row = [
        int(step),
        float(score),
        float(coverage),
        float(hopping_strength),
        *[float(x) for x in params.tolist()],
    ]

    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"step={step} score={score:.3f} t={params[0]:.3f}, Coverage={coverage:.3f}, ||z||={np.linalg.norm(z):.2f}")

def explore_once(model, RW_SIGMA, NUM_CAND):

    run_dir = os.path.join(OUT_DIR, str(time.time()))
    os.makedirs(run_dir, exist_ok=True)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)

    # === 初始化：直接从 prior 采样 ===
    # 这一步本身就可能生成很差的图（取决于你的 VAE prior match 是否好）
    with torch.no_grad():
        recon, mu_q, logvar_q, (pi_s, mu_s, log_sigma_s), z = model(get_init_tensor(IMAGE_SIZE).unsqueeze(0).to(device))
        # ---- stochastic prediction + confidence (from sampled z) ----
        theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
            pi_s, mu_s, log_sigma_s, var_scale=VAR_SCALE
        )
    recon = inv_smooth_scale(recon)
    recon = recon.cpu().detach().numpy()[0, 0]
    recon = postprocess_image(recon)
    z = z.cpu().detach().numpy()[0]
    y_pred_s = theta_hat_s.detach().cpu().numpy()[0]
    conf_s = conf_param_s.detach().cpu().numpy()[0]
    conf_global_s = conf_global_s.detach().cpu().numpy()[0]

    _, metrics, scores = generate_analysis_figure(np.clip(recon, 0, 1))
    s = scores["empirical_score"]
    c = metrics["dendrite_coverage"]
    t = y_pred_s[0]
    save_step(run_dir, 0, recon, z, y_pred_s, c, s, hopping_strength=RW_SIGMA)

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
            recon_cand = inv_smooth_scale(recon_cand)
            recon_cand = recon_cand.cpu().detach().numpy()[0, 0]
            recon_cand = postprocess_image(recon_cand)
            y_pred_s_cand = theta_hat_s_cand.detach().cpu().numpy()[0]
            conf_s_cand = conf_param_s_cand.detach().cpu().numpy()[0]
            conf_global_s_cand = conf_global_s_cand.detach().cpu().numpy()[0]

            _, metrics_cand, scores_cand = generate_analysis_figure(np.clip(recon_cand, 0, 1))
            t_cand = y_pred_s_cand[0]
            s_cand = scores_cand["empirical_score"]
            cnn_cand = metrics_cand["connected_components"]
            c_cand = metrics_cand["dendrite_coverage"]

            # 总结全局匹配度
            H = - np.linalg.norm(y_pred_s_cand - y_pred_s) - (s_cand - s)

            # save cands
            z_cands.append(z_cand.copy())
            H_list.append(float(H))

            if c_cand < c or t_cand < t or (cnn_cand >= 3 and STRICT):
                print(
                    f"    [Reject]c_cand={c_cand:.3f}<c={c:.3f} or t_cand={t_cand:.3f}<t={t:.3f} connected_components={cnn_cand}")
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

        save_step(run_dir, step, best_img, best_z, best_params, best_coverage, best_score, hopping_strength=RW_SIGMA)

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

    return score_path, coverage_path

def main():

    model = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.eval()

    # with open("results.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #
    #     writer.writerow([
    #         "sigma", "cand_num", "run_id",
    #         "ss_len", "ss_mean", "ss_var", "ss_min", "ss_max",
    #         "cs_len", "cs_mean", "cs_var", "cs_min", "cs_max"
    #     ])
    #
    #     for sigma in [0.01, 0.1, 0.25, 0.5]:
    #         for cand_num in [16, 32, 64, 128]:
    #             for run_id in range(10):
    #                 ss, cs = explore_once(model, sigma, cand_num)
    #
    #                 ss = np.array(ss)
    #                 cs = np.array(cs)
    #
    #                 writer.writerow([
    #                     sigma, cand_num, run_id,
    #                     len(ss), ss.mean(), ss.var(), ss.min(), ss.max(),
    #                     len(cs), cs.mean(), cs.var(), cs.min(), cs.max()
    #                 ])
    explore_once(model, 0.2, 32)

if __name__ == "__main__":
    main()
