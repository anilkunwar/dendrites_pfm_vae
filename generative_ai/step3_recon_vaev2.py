# eval_vae_mdn_with_confidence.py
import os
import json
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataloader import DendritePFMDataset


# ==========================================================
# 直接在这里改配置（不使用命令行参数）
# ==========================================================
IMAGE_SIZE = (3, 64, 64)
MODEL_ROOT = "results/VAEv11_MDN_lat=16_K=8_beta=5.0_warm=0.3_ctr=1.0_smooth=2.0_time=20260120_221422"  # 改成你的路径
BATCH_SIZE = 32

SAVE_EXAMPLES = True
NUM_EXAMPLES = 8

# 可选：参数名 json（list[str]，长度= num_params），没有就留空
PARAM_NAMES_JSON = ""  # e.g. "data/param_names.json"

# 置信度映射尺度：conf_i = exp(-Var_i / VAR_SCALE)
# 若你训练日志里 var 很小/很大，可调这个值让 conf 分布更“有区分度”
VAR_SCALE = 1.0

# top-k 模式输出（用于保存分析）
TOPK_MODES = 3

# ==========================================================
# 辅助函数（保留你原结构，并增加置信度保存/画图）
# ==========================================================
def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12):
    """
    y_true/y_pred: [N, P]
    returns dict with per-dim and overall metrics
    """
    assert y_true.shape == y_pred.shape, (y_true.shape, y_pred.shape)
    err = y_pred - y_true  # [N,P]

    mae = np.mean(np.abs(err), axis=0)
    mse = np.mean(err ** 2, axis=0)
    rmse = np.sqrt(mse)

    # R^2 per dim
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0) + eps
    r2 = 1.0 - ss_res / ss_tot

    # Pearson correlation per dim
    yt = y_true - np.mean(y_true, axis=0, keepdims=True)
    yp = y_pred - np.mean(y_pred, axis=0, keepdims=True)
    cov = np.sum(yt * yp, axis=0)
    std = np.sqrt(np.sum(yt ** 2, axis=0) * np.sum(yp ** 2, axis=0)) + eps
    corr = cov / std

    overall = {
        "MAE_mean": float(np.mean(mae)),
        "RMSE_mean": float(np.mean(rmse)),
        "R2_mean": float(np.mean(r2)),
        "Corr_mean": float(np.mean(corr)),
    }

    per_dim = {
        "MAE": mae.tolist(),
        "RMSE": rmse.tolist(),
        "R2": r2.tolist(),
        "Corr": corr.tolist(),
    }
    return {"overall": overall, "per_dim": per_dim}


def plot_regression_summary(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str, prefix: str, param_names=None):
    """
    Produces:
      1) MAE bar chart per parameter
      2) R2 bar chart per parameter
      3) Overall scatter (flattened)
      4) Residual histogram (flattened)
    """
    os.makedirs(save_dir, exist_ok=True)
    N, P = y_true.shape
    if param_names is None or len(param_names) != P:
        param_names = [f"p{i}" for i in range(P)]

    m = regression_metrics(y_true, y_pred)
    mae = np.array(m["per_dim"]["MAE"])
    r2 = np.array(m["per_dim"]["R2"])

    # 1) MAE bar
    plt.figure(figsize=(max(10, P * 0.5), 4))
    x = np.arange(P)
    plt.bar(x, mae)
    plt.xticks(x, param_names, rotation=60, ha="right")
    plt.ylabel("MAE")
    plt.title("Control parameter regression: MAE per parameter")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_mae_per_param.png"), dpi=300)
    plt.close()

    # 2) R2 bar
    plt.figure(figsize=(max(10, P * 0.5), 4))
    plt.bar(x, r2)
    plt.xticks(x, param_names, rotation=60, ha="right")
    plt.ylabel("R²")
    plt.title("Control parameter regression: R² per parameter")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_r2_per_param.png"), dpi=300)
    plt.close()

    # 3) Overall scatter (flatten)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    plt.figure(figsize=(5, 5))
    plt.scatter(yt, yp, s=6, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title("Overall true vs pred (all params flattened)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_overall_scatter.png"), dpi=300)
    plt.close()

    # 4) Residual histogram
    res = (y_pred - y_true).reshape(-1)
    plt.figure(figsize=(6, 4))
    plt.hist(res, bins=60)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title("Residual distribution (all params flattened)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_residual_hist.png"), dpi=300)
    plt.close()

    return m


def plot_confidence_summary(conf_param: np.ndarray, conf_global: np.ndarray, save_dir: str, prefix: str, param_names=None):
    """
    conf_param: [N, P] in (0, 1]
    conf_global: [N]
    """
    os.makedirs(save_dir, exist_ok=True)
    N, P = conf_param.shape
    if param_names is None or len(param_names) != P:
        param_names = [f"p{i}" for i in range(P)]

    # per-param mean confidence bar
    mean_c = conf_param.mean(axis=0)
    plt.figure(figsize=(max(10, P * 0.5), 4))
    x = np.arange(P)
    plt.bar(x, mean_c)
    plt.xticks(x, param_names, rotation=60, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean confidence")
    plt.title("Mean confidence per parameter")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_conf_param_mean.png"), dpi=300)
    plt.close()

    # global confidence hist
    plt.figure(figsize=(6, 4))
    plt.hist(conf_global, bins=60)
    plt.xlabel("Global confidence")
    plt.ylabel("Count")
    plt.title("Global confidence distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_conf_global_hist.png"), dpi=300)
    plt.close()

    # flattened param confidence hist
    plt.figure(figsize=(6, 4))
    plt.hist(conf_param.reshape(-1), bins=60)
    plt.xlabel("Param confidence (flattened)")
    plt.ylabel("Count")
    plt.title("Param confidence distribution (all params flattened)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_conf_param_hist.png"), dpi=300)
    plt.close()


# ==========================================================
# MDN 置信度计算（脚本内自包含，避免依赖别的文件）
# ==========================================================
@torch.no_grad()
def mdn_point_and_confidence(pi, mu, log_sigma, var_scale: float = 1.0, topk: int = 3):
    """
    pi: [B, K]
    mu/log_sigma: [B, K, P]
    returns:
      theta_hat: [B, P]  (混合均值)
      conf_param: [B, P] (逐参数置信度)
      conf_global: [B]   (全局置信度)
      modes: dict: top-k (pi_k, mu_k, idx_k)
    """
    eps = 1e-12
    B, K, P = mu.shape
    pi_ = pi / (pi.sum(dim=-1, keepdim=True) + eps)

    # mean
    theta_hat = torch.sum(pi_.unsqueeze(-1) * mu, dim=1)  # [B,P]

    # variance: Var = E[sigma^2 + mu^2] - (E[mu])^2
    sigma2 = torch.exp(2.0 * log_sigma)
    e_mu2 = torch.sum(pi_.unsqueeze(-1) * (sigma2 + mu ** 2), dim=1)  # [B,P]
    var = torch.clamp(e_mu2 - theta_hat ** 2, min=0.0)

    # confidence
    conf_param = torch.exp(-var / max(var_scale, eps))  # [B,P]

    # global confidence from mixture entropy
    entropy = -torch.sum(pi_ * torch.log(pi_ + eps), dim=-1)  # [B]
    conf_global = torch.exp(-entropy)

    # top-k modes
    k = min(topk, K)
    topv, topi = torch.topk(pi_, k=k, dim=-1)  # [B,k]
    top_mu = torch.gather(mu, 1, topi.unsqueeze(-1).expand(B, k, P))  # [B,k,P]
    modes = {"pi_topk": topv, "mu_topk": top_mu, "idx_topk": topi}
    return theta_hat, conf_param, conf_global, modes


# ==========================================================
# 主流程
# ==========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = DendritePFMDataset(IMAGE_SIZE, os.path.join("data", "dataset_split.json"), split="test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # load model
    ckpt_path = os.path.join(MODEL_ROOT, "ckpt", "best.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    vae = torch.load(ckpt_path, weights_only=False).to(device)
    vae.eval()

    save_fig_path = os.path.join(MODEL_ROOT, "figures_mdn")
    os.makedirs(save_fig_path, exist_ok=True)

    # param names
    param_names = None
    if PARAM_NAMES_JSON and os.path.exists(PARAM_NAMES_JSON):
        with open(PARAM_NAMES_JSON, "r", encoding="utf-8") as f:
            param_names = json.load(f)

    all_y = []

    # deterministic (use mu_q as z)
    all_pred_det = []
    all_conf_param_det = []
    all_conf_global_det = []
    all_modes_det = []  # optional: store topk modes

    # stochastic (use sampled z from forward)
    all_pred_sto = []
    all_conf_param_sto = []
    all_conf_global_sto = []
    all_modes_sto = []

    # evaluate
    with torch.no_grad():
        for iteration, (x, y, did, _) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)

            # forward: recon and latent (sampled z)
            recon_x, mu_q, logvar_q, mdn_out_sto, z = vae(x)
            pi_s, mu_s, log_sigma_s = mdn_out_sto

            # ---- stochastic prediction + confidence (from sampled z) ----
            theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
                pi_s, mu_s, log_sigma_s, var_scale=VAR_SCALE, topk=TOPK_MODES
            )

            # ---- deterministic prediction + confidence (use mu_q as z, no sampling) ----
            # 直接过 mdn_head(mu_q)
            pi_d, mu_d, log_sigma_d = vae.mdn_head(mu_q)
            theta_hat_d, conf_param_d, conf_global_d, modes_d = mdn_point_and_confidence(
                pi_d, mu_d, log_sigma_d, var_scale=VAR_SCALE, topk=TOPK_MODES
            )

            all_y.append(_to_numpy(y))

            all_pred_det.append(_to_numpy(theta_hat_d))
            all_conf_param_det.append(_to_numpy(conf_param_d))
            all_conf_global_det.append(_to_numpy(conf_global_d))
            # modes: 保存 topk 的 (pi, mu) 便于分析（体积会大一点，按需）
            all_modes_det.append({
                "pi_topk": _to_numpy(modes_d["pi_topk"]),
                "mu_topk": _to_numpy(modes_d["mu_topk"]),
                "idx_topk": _to_numpy(modes_d["idx_topk"]),
            })

            all_pred_sto.append(_to_numpy(theta_hat_s))
            all_conf_param_sto.append(_to_numpy(conf_param_s))
            all_conf_global_sto.append(_to_numpy(conf_global_s))
            all_modes_sto.append({
                "pi_topk": _to_numpy(modes_s["pi_topk"]),
                "mu_topk": _to_numpy(modes_s["mu_topk"]),
                "idx_topk": _to_numpy(modes_s["idx_topk"]),
            })

            # save a few recon examples
            if SAVE_EXAMPLES and iteration < NUM_EXAMPLES:
                bi = 0
                title = f"idx={iteration:04d}_did={did[bi]}"

                # 同时把该样本的 det/sto 置信度写进标题，方便你目视对比
                cg_d = float(conf_global_d[bi].item())
                cg_s = float(conf_global_s[bi].item())

                plt.figure(figsize=(6, 6))
                plt.suptitle(f"{title}\nconf_global_det={cg_d:.3f} | conf_global_sto={cg_s:.3f}")

                plt.subplot(3, 2, 1); plt.axis("off"); plt.title("x ch0");     plt.imshow(_to_numpy(x[bi])[0])
                plt.subplot(3, 2, 2); plt.axis("off"); plt.title("recon ch0"); plt.imshow(_to_numpy(recon_x[bi])[0])
                plt.subplot(3, 2, 3); plt.axis("off"); plt.title("x ch1");     plt.imshow(_to_numpy(x[bi])[1])
                plt.subplot(3, 2, 4); plt.axis("off"); plt.title("recon ch1"); plt.imshow(_to_numpy(recon_x[bi])[1])
                plt.subplot(3, 2, 5); plt.axis("off"); plt.title("x ch2");     plt.imshow(_to_numpy(x[bi])[2])
                plt.subplot(3, 2, 6); plt.axis("off"); plt.title("recon ch2"); plt.imshow(_to_numpy(recon_x[bi])[2])

                plt.tight_layout()
                plt.savefig(os.path.join(save_fig_path, f"{title}.png"), dpi=300)
                plt.close()

    # concat arrays
    y_true = np.concatenate(all_y, axis=0)  # [N,P]

    y_pred_det = np.concatenate(all_pred_det, axis=0)
    conf_param_det = np.concatenate(all_conf_param_det, axis=0)
    conf_global_det = np.concatenate(all_conf_global_det, axis=0)

    y_pred_sto = np.concatenate(all_pred_sto, axis=0)
    conf_param_sto = np.concatenate(all_conf_param_sto, axis=0)
    conf_global_sto = np.concatenate(all_conf_global_sto, axis=0)

    # metrics + plots
    metrics_det = plot_regression_summary(y_true, y_pred_det, save_fig_path, prefix="ctr_det", param_names=param_names)
    metrics_sto = plot_regression_summary(y_true, y_pred_sto, save_fig_path, prefix="ctr_stoch", param_names=param_names)

    plot_confidence_summary(conf_param_det, conf_global_det, save_fig_path, prefix="det", param_names=param_names)
    plot_confidence_summary(conf_param_sto, conf_global_sto, save_fig_path, prefix="stoch", param_names=param_names)

    # save metrics
    with open(os.path.join(save_fig_path, "ctr_reg_metrics_det.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_det, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_fig_path, "ctr_reg_metrics_stoch.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_sto, f, ensure_ascii=False, indent=2)

    # save predictions + confidence for downstream analysis
    np.save(os.path.join(save_fig_path, "y_true.npy"), y_true)

    np.save(os.path.join(save_fig_path, "y_pred_det.npy"), y_pred_det)
    np.save(os.path.join(save_fig_path, "conf_param_det.npy"), conf_param_det)
    np.save(os.path.join(save_fig_path, "conf_global_det.npy"), conf_global_det)

    np.save(os.path.join(save_fig_path, "y_pred_stoch.npy"), y_pred_sto)
    np.save(os.path.join(save_fig_path, "conf_param_stoch.npy"), conf_param_sto)
    np.save(os.path.join(save_fig_path, "conf_global_stoch.npy"), conf_global_sto)

    # 可选：保存 top-k 模式（体积会大一些）
    # 如果你觉得太占空间，把下面这段注释掉即可
    with open(os.path.join(save_fig_path, "mdn_modes_det.json"), "w", encoding="utf-8") as f:
        json.dump(all_modes_det, f, ensure_ascii=False)
    with open(os.path.join(save_fig_path, "mdn_modes_stoch.json"), "w", encoding="utf-8") as f:
        json.dump(all_modes_sto, f, ensure_ascii=False)

    # Print summary
    print("\n===== Control regression metrics (deterministic, using mu_q as z) =====")
    print(metrics_det["overall"])
    print("Per-dim MAE (first 5):", metrics_det["per_dim"]["MAE"][:5])
    print("Per-dim R2  (first 5):", metrics_det["per_dim"]["R2"][:5])
    print("Global confidence mean:", float(conf_global_det.mean()))

    print("\n===== Control regression metrics (stochastic, using sampled z) =====")
    print(metrics_sto["overall"])
    print("Global confidence mean:", float(conf_global_sto.mean()))

    print("\nSaved to:", save_fig_path)


if __name__ == "__main__":
    main()
