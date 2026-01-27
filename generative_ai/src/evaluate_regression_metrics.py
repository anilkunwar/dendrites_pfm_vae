import os
import numpy as np
from scipy import stats
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

def plot_qq(residuals: np.ndarray, title: str = "Q–Q plot of residuals", save_path: str = None):
    """Normal Q–Q plot for residuals."""
    residuals = np.asarray(residuals).reshape(-1)
    residuals = residuals[np.isfinite(residuals)]  # 防止 NaN/inf

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_regression_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
    save_dir: str = None,
    param_names=None
):
    """
    Produces:
      1) MAE bar chart per parameter
      2) R2 bar chart per parameter
      3) Overall scatter (flattened)
      4) Residual histogram (flattened)
      5) Residual Q–Q plot (flattened)
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
    plot_scatter_evolution(
        y_true.reshape(-1),
        y_pred.reshape(-1),
        xlabel="True",
        ylabel="Pred",
        title="Overall true vs pred (all params flattened)"
    )

    # 4) Residual histogram
    residuals = (y_pred - y_true).reshape(-1)
    plot_histogram(residuals, xlabel="Residual", ylabel="Count")

    # 5) Residual Q–Q plot
    qq_save = None if save_dir is None else os.path.join(save_dir, f"{prefix}_residuals_qq.png")
    plot_qq(residuals, save_path=qq_save)

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
    plot_histogram(conf_param.mean(axis=0), ylabel="Mean confidence", title="Mean confidence per parameter")

    # global confidence hist
    plot_histogram(conf_global, xlabel="Global confidence", ylabel="Count")

    # flattened param confidence hist
    plot_histogram(conf_param.reshape(-1), xlabel="Param confidence", ylabel="Count")