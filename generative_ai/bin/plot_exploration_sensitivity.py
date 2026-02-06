import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric_pretty(
    df: pd.DataFrame,
    metric: str,
    x: str = "sigma",
    hue: str = "cand_num",
    y_label = None,
    title: str = None,
    alpha_band: float = 0.22,
    lw: float = 2.0,
    ms: float = 6.5,
    figsize=(7.5, 5)
):
    """
    Pretty scientific plot:
      - dashed line for trend
      - solid dots for means
      - shaded band for ± std
      - one metric per figure
    """
    stats = (
        df.groupby([hue, x])[metric]
        .agg(['mean', 'std'])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    for h in sorted(stats[hue].unique()):
        sub = stats[stats[hue] == h].sort_values(x)

        xs = sub[x].to_numpy()
        mean = sub['mean'].to_numpy()
        std = sub['std'].to_numpy()

        # 先画虚线，拿到这条线的颜色
        (line,) = ax.plot(
            xs, mean,
            linestyle='--',
            linewidth=lw,
            alpha=0.9
        )
        color = line.get_color()  # 🔥 关键：统一颜色源

        # 实心点（同色）
        ax.scatter(
            xs, mean,
            s=ms ** 2,
            color=color,
            zorder=3,
            label=f"{hue}={h}"
        )

        # 误差带（同色，半透明）
        ax.fill_between(
            xs,
            mean - std,
            mean + std,
            color=color,
            alpha=alpha_band
        )

    # 样式收尾
    ax.set_xlabel(x)
    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(metric)
    if title is not None:
        ax.set_title(title)

    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

# 用法示例：
df = pd.read_csv("../results.csv")
plot_metric_pretty(df, metric="ss_len", x="sigma", hue="cand_num", y_label="Total Steps")
plot_metric_pretty(df, metric="ss_mean", x="sigma", hue="cand_num", y_label="Mean Dendrite Intensity Score")
plot_metric_pretty(df, metric="cs_max", x="sigma", hue="cand_num", y_label="Max Dendrite Coverage")

