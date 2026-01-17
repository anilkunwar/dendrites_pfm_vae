import os
import json
import math
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataloader import DendritePFMDataset


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


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = DendritePFMDataset(args.image_size, os.path.join("data", "dataset_split.json"), split="test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    vae = torch.load(os.path.join(args.model_root, "ckpt", "best.pt"), weights_only=False).to(device)
    vae.eval()

    save_fig_path = os.path.join(args.model_root, "figures")
    os.makedirs(save_fig_path, exist_ok=True)

    all_y = []
    all_pred = []
    all_pred_stochastic = []

    # evaluate
    with torch.no_grad():
        for iteration, (x, y, did, _) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)

            # recon and latent
            recon_x, mu_q, logvar_q, ctr_pred_stoch, z = vae(x)

            # 建议用 mu_q 做“确定性回归评估”，避免采样噪声拉低精度
            ctr_pred = vae.ctr_head(mu_q)

            all_y.append(_to_numpy(y))
            all_pred.append(_to_numpy(ctr_pred))
            all_pred_stochastic.append(_to_numpy(ctr_pred_stoch))

            if args.save_examples and iteration < args.num_examples:
                # show first item in batch
                bi = 0
                title = f"idx={iteration:04d}_did={did[bi]}"
                plt.figure(figsize=(6, 6))
                plt.suptitle(title)

                plt.subplot(3, 2, 1); plt.axis('off'); plt.title("x ch0");      plt.imshow(_to_numpy(x[bi])[0])
                plt.subplot(3, 2, 2); plt.axis('off'); plt.title("recon ch0");  plt.imshow(_to_numpy(recon_x[bi])[0])
                plt.subplot(3, 2, 3); plt.axis('off'); plt.title("x ch1");      plt.imshow(_to_numpy(x[bi])[1])
                plt.subplot(3, 2, 4); plt.axis('off'); plt.title("recon ch1");  plt.imshow(_to_numpy(recon_x[bi])[1])
                plt.subplot(3, 2, 5); plt.axis('off'); plt.title("x ch2");      plt.imshow(_to_numpy(x[bi])[2])
                plt.subplot(3, 2, 6); plt.axis('off'); plt.title("recon ch2");  plt.imshow(_to_numpy(recon_x[bi])[2])

                plt.tight_layout()
                plt.savefig(os.path.join(save_fig_path, f"{title}.png"), dpi=300)
                plt.close()

    y_true = np.concatenate(all_y, axis=0)              # [N,P]
    y_pred = np.concatenate(all_pred, axis=0)           # [N,P] (deterministic from mu)
    y_pred_stoch = np.concatenate(all_pred_stochastic, axis=0)  # [N,P] (from sampled z)

    # Metrics + plots
    param_names = None
    if args.param_names_json and os.path.exists(args.param_names_json):
        with open(args.param_names_json, "r", encoding="utf-8") as f:
            param_names = json.load(f)

    metrics_det = plot_regression_summary(y_true, y_pred, save_fig_path, prefix="ctr_det", param_names=param_names)
    metrics_sto = plot_regression_summary(y_true, y_pred_stoch, save_fig_path, prefix="ctr_stoch", param_names=param_names)

    # Save metrics to json
    with open(os.path.join(save_fig_path, "ctr_reg_metrics_det.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_det, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_fig_path, "ctr_reg_metrics_stoch.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_sto, f, ensure_ascii=False, indent=2)

    # Print a short summary
    print("\n===== Control regression metrics (deterministic, using mu_q) =====")
    print(metrics_det["overall"])
    print("Per-dim MAE (first 5):", metrics_det["per_dim"]["MAE"][:5])
    print("Per-dim R2  (first 5):", metrics_det["per_dim"]["R2"][:5])

    print("\n===== Control regression metrics (stochastic, using sampled z) =====")
    print(metrics_sto["overall"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=tuple, default=(3, 64, 64))
    parser.add_argument("--model_root", type=str, default='results/V9_lat=8_beta=0.1_warm=0.3_ctr=1.0_smooth=0.05_time=20260105_012629')

    # regression eval options
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_examples", action="store_true", help="Save a few recon examples.")
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--param_names_json", type=str, default="", help="Optional json list of param names, length=num_params.")

    args = parser.parse_args()
    main(args)
