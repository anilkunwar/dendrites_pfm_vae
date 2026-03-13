import os
import numpy as np

from src.visualizer import plot_line_evolution, plot_scatter_evolution, plot_histogram, plot_qq_evolution


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
    plot_qq_evolution(residuals, title="Residual Q–Q plot", save_path=qq_save)

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

def plot_figure6_layoutB_bigfont(
    y_true,
    y_pred,
    conf_param,
    param_names,
    save_path=None,
    rotate_xticks=0,
    topk_annot=0,
):
    """
    Figure 6 (white background, 2+1 layout):
      (a) residual histogram (+ mean line)
      (b) Q–Q plot
      (c) x-axis = parameters, bars = MAE, line (right y-axis) = Confidence

    Requirements:
    - (a) mean line meaning annotated + legend
    - Panel labels (a)(b)(c) OUTSIDE top-left of each panel (user-tuned positions)
    - Labels must not go out of figure bounds (reserve margins + clamp)
    - (c) NO red mean dashed line
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from matplotlib.lines import Line2D

    # helpers (generic, reusable)
    from src.visualizer import _apply_white_background, draw_histogram, draw_line

    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 18,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 2.2,
        "lines.linewidth": 3.0,
        "legend.fontsize": 16,
    })

    # ---------- data ----------
    resid = (y_pred - y_true).flatten()
    resid = resid[np.isfinite(resid)]
    mean_r = float(np.mean(resid)) if resid.size else 0.0

    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    mae = np.asarray(mae).flatten()

    conf_mean = np.asarray(conf_param).mean(axis=0).flatten()

    # ---------- palette ----------
    col_hist = "#4B5563"
    col_red  = "#B91C1C"   # mean line + panel labels
    col_bpts = "#4F46E5"
    col_bln  = "#111827"
    col_mae  = "#0F766E"
    col_conf = "#6D28D9"

    # ---------- layout: top row taller ----------
    fig = plt.figure(figsize=(14.8, 8.6))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.35, 1.15],
        hspace=0.40,
        wspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    _apply_white_background(fig, [ax_a, ax_b, ax_c])

    # ================= (a) residual histogram =================
    # IMPORTANT: show_mean=False to avoid duplicated mean line;
    # we'll draw the mean ourselves (dashed) and add legend.
    draw_histogram(
        ax_a, resid,
        bins=35,
        color=col_hist,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.85,
        show_mean=False,
    )

    # dashed mean line + legend
    ax_a.axvline(mean_r, color=col_red, linestyle="--", linewidth=3.2, zorder=6)
    mean_handle = Line2D([0], [0], color=col_red, linestyle="--", linewidth=3.2,
                         label=f"Mean = {mean_r:.3f}")
    ax_a.legend(handles=[mean_handle], loc="upper right", frameon=False)

    ax_a.set_xlabel("Residual")
    ax_a.set_ylabel("Frequency")

    # # keep your mean text annotation if you still want it (optional)
    # ax_a.text(
    #     0.55, 0.92,
    #     r"Mean:" + f" = {mean_r:.3f}",
    #     transform=ax_a.transAxes,
    #     fontsize=18,
    #     va="top", ha="left",
    #     bbox=dict(facecolor="white", edgecolor="none", pad=2.0),
    #     zorder=10,
    # )

    # ================= (b) Q-Q plot =================
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
    ax_b.scatter(osm, osr, s=26, color=col_bpts, zorder=3)
    draw_line(ax_b, osm, slope * osm + intercept, color=col_bln, marker=None, linewidth=3.2)

    ax_b.set_xlabel("Theoretical Quantiles")
    ax_b.set_ylabel("Sample Quantiles", labelpad=18)

    ax_b.text(
        0.26, 0.90,
        r"$R^2$" + f" = {r**2:.4f}\n" + f"slope = {slope:.3f}",
        transform=ax_b.transAxes,
        fontsize=18,
        va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", pad=2.0),
        zorder=10,
    )

    # ================= (c) MAE bars + confidence line =================
    n = len(param_names)
    x = np.arange(n)

    if rotate_xticks is None or rotate_xticks == 0:
        rot = 35 if n >= 10 else 0
    else:
        rot = rotate_xticks

    xtick_fs = 16 if n >= 14 else (17 if n >= 10 else 18)

    ax_c.bar(x, mae, color=col_mae, edgecolor="black", linewidth=1.0, zorder=2)
    ax_c.set_ylabel("MAE", color=col_mae)
    ax_c.tick_params(axis="y", colors=col_mae)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        param_names,
        fontsize=xtick_fs,
        rotation=rot,
        ha="right" if rot else "center",
    )
    ax_c.set_xlim(-0.7, n - 0.3)

    ax_c2 = ax_c.twinx()
    draw_line(ax_c2, x, conf_mean, color=col_conf, marker="o", markersize=7, linewidth=2.8)
    ax_c2.set_ylabel("Confidence", color=col_conf)
    ax_c2.tick_params(axis="y", colors=col_conf)

    # IMPORTANT: no red dashed mean line in (c)

    # ---------- keep your existing margins ----------
    fig.subplots_adjust(
        left=0.13,
        right=0.92,
        top=0.955,
        bottom=0.24,
        wspace=0.30,
        hspace=0.40,
    )

    def add_panel_label_outside(ax, s):
        pos = ax.get_position()
        x0 = pos.x0 - 0.070
        y1 = pos.y1 + 0.030
        x0 = max(0.005, x0)
        y1 = min(0.995, y1)
        fig.text(
            x0, y1, s,
            ha="left", va="top",
            fontsize=30, fontweight="bold",
            color=col_red,
        )

    add_panel_label_outside(ax_a, "(a)")
    add_panel_label_outside(ax_b, "(b)")
    add_panel_label_outside(ax_c, "(c)")

    # optional top-k annotations
    if topk_annot and topk_annot > 0:
        top_idx = np.argsort(mae)[-topk_annot:]
        ymax = float(np.max(mae)) if mae.size else 1.0
        for i in top_idx:
            ax_c.text(
                i,
                mae[i] + 0.02 * ymax,
                f"{mae[i]:.3f}",
                ha="center", va="bottom",
                fontsize=14,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
                zorder=10,
            )

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.show()

    return {
        "per_dim": {"MAE": mae},
        "overall": {
            "MAE_mean": float(np.mean(mae)) if mae.size else 0.0,
            "residual_mean": mean_r,
        },
    }

def plot_figure7_confidence_distributions(
    conf_param,
    conf_global,
    save_path=None,
):
    """
    Figure 7 (1x2 horizontal layout)
    Color palette consistent with Figure 6
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from src.visualizer import _apply_white_background

    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.linewidth": 2.2,
        "lines.linewidth": 2.5,
    })

    conf_global = np.asarray(conf_global).flatten()
    conf_param  = np.asarray(conf_param).flatten()

    # ===== unified color system (same as Fig6) =====
    col_global = "#0F766E"   # teal (same as MAE)
    col_param  = "#6D28D9"   # purple (same as confidence)
    col_mean   = "#B91C1C"   # deep red (same as mean lines)
    col_label  = "#B91C1C"   # panel labels

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14.5, 5.5),
        gridspec_kw=dict(wspace=0.30)
    )

    ax_a, ax_b = axes
    _apply_white_background(fig, axes)

    # ================= (a) Global =================
    mean_global = np.mean(conf_global)

    ax_a.hist(
        conf_global,
        bins=10,
        color=col_global,
        edgecolor="black",
        alpha=0.85
    )

    ax_a.axvline(
        mean_global,
        color=col_mean,
        linestyle="--",
        linewidth=3,
        label=f"Mean = {mean_global:.3f}"
    )

    ax_a.set_xlabel("Global Confidence", fontweight="bold")
    ax_a.set_ylabel("Frequency", fontweight="bold")
    ax_a.legend(frameon=False)

    # ================= (b) Parameter-wise =================
    mean_param = np.mean(conf_param)

    ax_b.hist(
        conf_param,
        bins=10,
        color=col_param,
        edgecolor="black",
        alpha=0.85
    )

    ax_b.axvline(
        mean_param,
        color=col_mean,
        linestyle="--",
        linewidth=3,
        label=f"Mean = {mean_param:.3f}"
    )

    ax_b.set_xlabel("Parameter-wise Confidence", fontweight="bold")
    ax_b.set_ylabel("Frequency", fontweight="bold")
    ax_b.legend(frameon=False)

    # ===== layout first =====
    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.92,
        bottom=0.15,
        wspace=0.30,
    )

    # ===== panel labels (balanced position) =====
    def add_panel_label(ax, s):
        pos = ax.get_position()
        x = pos.x0 - 0.075
        y = pos.y1 + 0.050

        x = max(0.02, x)
        y = min(0.97, y)

        fig.text(
            x, y, s,
            ha="left",
            va="top",
            fontsize=32,
            fontweight="bold",
            color=col_label,
        )

    add_panel_label(ax_a, "(a)")
    add_panel_label(ax_b, "(b)")

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight", facecolor="white")

    plt.show()
