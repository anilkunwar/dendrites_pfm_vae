import os
import numpy as np
import matplotlib.pyplot as plt

from src.visualizer import plot_line_evolution, plot_scatter_evolution, plot_histogram


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

def plot_regression_summary(y_true: np.ndarray, y_pred: np.ndarray, prefix: str, save_dir: str=None, param_names=None):
    """
    Produces:
      1) MAE bar chart per parameter
      2) R2 bar chart per parameter
      3) Overall scatter (flattened)
      4) Residual histogram (flattened)
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    N, P = y_true.shape
    if param_names is None or len(param_names) != P:
        param_names = [f"p{i}" for i in range(P)]

    m = regression_metrics(y_true, y_pred)
    mae = np.array(m["per_dim"]["MAE"])
    r2 = np.array(m["per_dim"]["R2"])
    x = np.arange(P)

    # 1) MAE bar
    plot_line_evolution(x, mae, ylabel="MAE", title="Control parameter regression: MAE per parameter")

    # 2) R2 bar
    plot_line_evolution(x, r2, ylabel="R²", title="Control parameter regression: R² per parameter")

    # 3) Overall scatter (flatten)
    plot_scatter_evolution(y_true.reshape(-1), y_pred.reshape(-1), xlabel="True", ylabel="Pred", title="Overall true vs pred (all params flattened)")

    # 4) Residual histogram
    plot_histogram((y_pred - y_true).reshape(-1), xlabel="Residual (pred - true)", ylabel="Count", title="Residual distribution (all params flattened)")

    return m

def plot_confidence_summary(conf_param: np.ndarray, conf_global: np.ndarray, prefix: str, save_dir: str=None, param_names=None):
    """
    conf_param: [N, P] in (0, 1]
    conf_global: [N]
    """
    if save_dir is not None:
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
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_conf_param_mean.png"), dpi=300)
    else:
        plt.show()
    plt.close()

    # global confidence hist
    plt.figure(figsize=(6, 4))
    plt.hist(conf_global, bins=60)
    plt.xlabel("Global confidence")
    plt.ylabel("Count")
    plt.title("Global confidence distribution")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_conf_global_hist.png"), dpi=300)
    else:
        plt.show()
    plt.close()

    # flattened param confidence hist
    plt.figure(figsize=(6, 4))
    plt.hist(conf_param.reshape(-1), bins=60)
    plt.xlabel("Param confidence (flattened)")
    plt.ylabel("Count")
    plt.title("Param confidence distribution (all params flattened)")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_conf_param_hist.png"), dpi=300)
    else:
        plt.show()
    plt.close()