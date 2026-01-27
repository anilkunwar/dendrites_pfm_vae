import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, Literal
from matplotlib.collections import LineCollection


def plot_line_evolution(
        x: np.ndarray,
        y: np.ndarray,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 5),
        show_mean: bool = False,
        show_fill: bool = True,
        show_markers: bool = True,
        show_colorbar: bool = False,
        color: str = '#3498DB',
        cmap: str = 'viridis',
        style: str = 'darkgrid',
        dpi: int = 300
):
    """
    创建美观的线条演化图

    Parameters:
    -----------
    x : np.ndarray
        X轴数据
    y : np.ndarray
        Y轴数据
    xlabel : str, optional
        X轴标签
    ylabel : str, optional
        Y轴标签
    title : str, optional
        图表标题
    save_path : str, optional
        保存路径（含文件名），不填则显示图表
    figsize : tuple, default=(8, 5)
        图表尺寸
    show_mean : bool, default=False
        是否显示均值线
    show_fill : bool, default=True
        是否显示填充区域
    show_markers : bool, default=True
        是否显示数据点标记
    show_colorbar : bool, default=False
        是否显示颜色条（根据x值渐变）
    color : str, default='#3498DB'
        主色调（当show_colorbar=False时使用）
    cmap : str, default='viridis'
        色图名称（当show_colorbar=True时使用）
        可选: 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlBu', 'Spectral'
    style : str, default='darkgrid'
        seaborn样式: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
    dpi : int, default=300
        保存图片的分辨率

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects

    Example:
    --------
    >>> x = np.arange(100)
    >>> y = np.random.randn(100).cumsum()
    >>> plot_line_evolution(x, y, xlabel="Step", ylabel="Value",
    ...                      title="My Plot", show_colorbar=True)
    """
    # 设置样式
    sns.set_theme(style=style, palette="muted")

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    if show_colorbar:
        # 使用渐变色线条
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(x.min(), x.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5, alpha=0.9)
        lc.set_array(x)
        line = ax.add_collection(lc)

        # 添加散点标记
        if show_markers:
            scatter = ax.scatter(x, y, c=x, cmap=cmap, s=40,
                                 edgecolors='white', linewidths=1.2,
                                 zorder=5, alpha=0.8, norm=norm)

        # 设置坐标轴范围
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min() - 0.05 * (y.max() - y.min()),
                    y.max() + 0.05 * (y.max() - y.min()))

        # 添加颜色条
        cbar = fig.colorbar(line, ax=ax, pad=0.02, aspect=30)
        if xlabel:
            cbar.set_label(xlabel, rotation=270, labelpad=20,
                           fontsize=11, fontweight='semibold')
        else:
            cbar.set_label('Value', rotation=270, labelpad=20,
                           fontsize=11, fontweight='semibold')
        cbar.outline.set_edgecolor('#95A5A6')
        cbar.outline.set_linewidth(1.2)

        # 填充区域（使用渐变色）
        if show_fill:
            ax.fill_between(x, np.min(y), y, alpha=0.15, color='steelblue')

    else:
        # 使用单一颜色
        marker_style = 'o' if show_markers else None
        markersize = 5 if show_markers else 0

        sns.lineplot(
            x=x, y=y, ax=ax,
            linewidth=2.5,
            marker=marker_style,
            markersize=markersize,
            color=color,
            markeredgecolor='white',
            markeredgewidth=1.2,
            alpha=0.85
        )

        # 添加填充区域
        if show_fill:
            ax.fill_between(x, np.min(y), y, alpha=0.2, color=color)

    # 添加均值参考线
    if show_mean:
        mean_val = np.mean(y)
        ax.axhline(
            mean_val,
            color='#E74C3C',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label=f'Mean: {mean_val:.3f}'
        )
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)

    # 设置标签和标题
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='semibold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold')
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # 优化网格和边框
    if show_colorbar:
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#95A5A6')
            spine.set_linewidth(1.2)
    else:
        sns.despine(left=False, bottom=False)

    # 调整布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    return fig, ax


def plot_scatter_evolution(
        x: np.ndarray,
        y: np.ndarray,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6),
        show_regression: bool = False,
        show_density: bool = False,
        show_colorbar: bool = False,
        color: str = '#3498DB',
        cmap: str = 'viridis',
        color_by: Optional[np.ndarray] = None,
        size: Union[int, np.ndarray] = 60,
        alpha: float = 0.7,
        style: str = 'darkgrid',
        marker: str = 'o',
        edgecolor: str = 'white',
        linewidth: float = 1.0,
        dpi: int = 300
):
    """
    创建美观的散点图

    Parameters:
    -----------
    x : np.ndarray
        X轴数据
    y : np.ndarray
        Y轴数据
    xlabel : str, optional
        X轴标签
    ylabel : str, optional
        Y轴标签
    title : str, optional
        图表标题
    save_path : str, optional
        保存路径（含文件名），不填则显示图表
    figsize : tuple, default=(8, 6)
        图表尺寸
    show_regression : bool, default=False
        是否显示回归线
    show_density : bool, default=False
        是否显示密度等高线
    show_colorbar : bool, default=False
        是否显示颜色条
    color : str, default='#3498DB'
        点的颜色（当show_colorbar=False且color_by=None时使用）
    cmap : str, default='viridis'
        色图名称（当show_colorbar=True或color_by不为None时使用）
        可选: 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlBu', 'Spectral'
    color_by : np.ndarray, optional
        用于着色的第三维数据（如时间、索引、其他变量）
    size : int or np.ndarray, default=60
        点的大小，可以是固定值或数组（用于气泡图）
    alpha : float, default=0.7
        透明度 (0-1)
    style : str, default='darkgrid'
        seaborn样式
    marker : str, default='o'
        标记样式: 'o', 's', '^', 'D', '*', 'p', 'h'
    edgecolor : str, default='white'
        边缘颜色
    linewidth : float, default=1.0
        边缘宽度
    dpi : int, default=300
        保存图片的分辨率

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects

    Example:
    --------
    >>> x = np.random.randn(100)
    >>> y = 2 * x + np.random.randn(100) * 0.5
    >>> plot_scatter_evolution(x, y, xlabel="X", ylabel="Y",
    ...                         show_regression=True, show_colorbar=True)
    """
    # 设置样式
    sns.set_theme(style=style, palette="muted")

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 确定颜色设置
    if color_by is not None or show_colorbar:
        use_colorbar = True
        c_values = color_by if color_by is not None else np.arange(len(x))
        norm = plt.Normalize(c_values.min(), c_values.max())
    else:
        use_colorbar = False
        c_values = color
        norm = None

    # 绘制散点
    scatter = ax.scatter(
        x, y,
        c=c_values,
        s=size,
        alpha=alpha,
        cmap=cmap if use_colorbar else None,
        norm=norm,
        marker=marker,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=5
    )

    # 添加颜色条
    if use_colorbar:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
        if color_by is not None and hasattr(color_by, 'name'):
            cbar.set_label(color_by.name, rotation=270, labelpad=20,
                           fontsize=11, fontweight='semibold')
        else:
            cbar.set_label('Value', rotation=270, labelpad=20,
                           fontsize=11, fontweight='semibold')
        cbar.outline.set_edgecolor('#95A5A6')
        cbar.outline.set_linewidth(1.2)

    # 添加回归线
    if show_regression:
        # 计算线性回归
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = p(x_line)

        ax.plot(x_line, y_line,
                color='#E74C3C',
                linestyle='--',
                linewidth=2.5,
                alpha=0.8,
                label=f'y = {z[0]:.2f}x + {z[1]:.2f}',
                zorder=3)

        # 计算R²
        y_pred = p(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # 添加文本框显示统计信息
        textstr = f'$R^2$ = {r2:.3f}\n$\\rho$ = {np.corrcoef(x, y)[0, 1]:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=props)

        ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)

    # 添加密度等高线
    if show_density:
        from scipy.stats import gaussian_kde
        try:
            # 计算密度
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)

            # 绘制等高线
            contours = ax.tricontour(x, y, z,
                                     levels=5,
                                     colors='gray',
                                     alpha=0.3,
                                     linewidths=1.0,
                                     zorder=1)
        except:
            pass  # 如果数据太少或其他原因无法计算密度，跳过

    # 设置标签和标题
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='semibold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold')
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # 优化网格和边框
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    if use_colorbar:
        for spine in ax.spines.values():
            spine.set_edgecolor('#95A5A6')
            spine.set_linewidth(1.2)
    else:
        sns.despine(left=False, bottom=False)

    # 调整布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    return fig, ax

# 使用示例
if __name__ == "__main__":
    # 示例1: 带颜色条（根据步数渐变）
    steps = np.arange(100)
    z_norms = np.linalg.norm(np.random.randn(100, 10), axis=1)

    plot_line_evolution(
        x=steps,
        y=z_norms,
        xlabel="Step",
        ylabel="||z||",
        title="Latent Norm Evolution with Colorbar",
        save_path="example_colorbar.png",
        show_colorbar=True,
        show_mean=True,
        cmap='viridis'  # 可选: 'plasma', 'coolwarm', 'RdYlBu' 等
    )

    # 示例2: 不带颜色条（单色）
    plot_line_evolution(
        x=steps,
        y=z_norms,
        xlabel="Step",
        ylabel="||z||",
        title="Latent Norm Evolution",
        save_path="example_no_colorbar.png",
        show_colorbar=False,
        show_mean=True,
        color='#9B59B6'
    )

    # 示例3: 不同的色图
    plot_line_evolution(
        x=steps,
        y=z_norms,
        xlabel="Iteration",
        ylabel="Loss",
        title="Training with Plasma Colormap",
        save_path="example_plasma.png",
        show_colorbar=True,
        cmap='plasma',
        show_markers=True
    )


def plot_histogram(
        data: np.ndarray,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6),
        bins: Union[int, str] = 'auto',
        show_kde: bool = True,
        show_mean: bool = True,
        show_median: bool = False,
        show_std: bool = False,
        color: str = '#3498DB',
        kde_color: str = '#E74C3C',
        alpha: float = 0.7,
        style: str = 'darkgrid',
        stat: Literal['count', 'frequency', 'density', 'probability'] = 'count',
        edgecolor: str = 'white',
        linewidth: float = 1.2,
        vertical: bool = False,
        cumulative: bool = False,
        dpi: int = 300
):
    """
    创建美观的直方图

    Parameters:
    -----------
    data : np.ndarray
        数据数组
    xlabel : str, optional
        X轴标签
    ylabel : str, optional
        Y轴标签
    title : str, optional
        图表标题
    save_path : str, optional
        保存路径（含文件名），不填则显示图表
    figsize : tuple, default=(8, 6)
        图表尺寸
    bins : int or str, default='auto'
        直方图分箱数量，'auto' 自动选择
    show_kde : bool, default=True
        是否显示核密度估计曲线
    show_mean : bool, default=True
        是否显示均值线
    show_median : bool, default=False
        是否显示中位数线
    show_std : bool, default=False
        是否显示±1标准差区域
    color : str, default='#3498DB'
        直方图颜色
    kde_color : str, default='#E74C3C'
        KDE曲线颜色
    alpha : float, default=0.7
        透明度 (0-1)
    style : str, default='darkgrid'
        seaborn样式
    stat : str, default='count'
        统计类型: 'count', 'frequency', 'density', 'probability'
    edgecolor : str, default='white'
        柱子边缘颜色
    linewidth : float, default=1.2
        柱子边缘宽度
    vertical : bool, default=False
        是否垂直显示（横向直方图）
    cumulative : bool, default=False
        是否显示累积分布
    dpi : int, default=300
        保存图片的分辨率

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects

    Example:
    --------
    >>> data = np.random.randn(1000)
    >>> plot_histogram(data, xlabel="Value", title="Distribution",
    ...                show_kde=True, show_mean=True)
    """
    # 设置样式
    sns.set_theme(style=style, palette="muted")

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 计算统计量
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)

    if vertical:
        # 横向直方图
        if show_kde:
            sns.histplot(
                y=data,
                bins=bins,
                kde=True,
                stat=stat,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
                kde_kws={'cut': 0},
                line_kws={'color': kde_color, 'linewidth': 2.5, 'alpha': 0.8},
                ax=ax,
                cumulative=cumulative
            )
        else:
            sns.histplot(
                y=data,
                bins=bins,
                kde=False,
                stat=stat,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
                ax=ax,
                cumulative=cumulative
            )

        # 添加均值线
        if show_mean:
            ax.axhline(mean_val, color='#2ECC71', linestyle='--',
                       linewidth=2.5, alpha=0.8, label=f'Mean: {mean_val:.3f}', zorder=10)

        # 添加中位数线
        if show_median:
            ax.axhline(median_val, color='#F39C12', linestyle='--',
                       linewidth=2.5, alpha=0.8, label=f'Median: {median_val:.3f}', zorder=10)

        # 添加标准差区域
        if show_std:
            ax.axhspan(mean_val - std_val, mean_val + std_val,
                       alpha=0.15, color='#9B59B6', label=f'±1 SD: {std_val:.3f}', zorder=1)

    else:
        # 纵向直方图（默认）
        if show_kde:
            sns.histplot(
                x=data,
                bins=bins,
                kde=True,
                stat=stat,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
                kde_kws={'cut': 0},
                line_kws={'color': kde_color, 'linewidth': 2.5, 'alpha': 0.8},
                ax=ax,
                cumulative=cumulative
            )
        else:
            sns.histplot(
                x=data,
                bins=bins,
                kde=False,
                stat=stat,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
                ax=ax,
                cumulative=cumulative
            )

        # 添加均值线
        if show_mean:
            ax.axvline(mean_val, color='#2ECC71', linestyle='--',
                       linewidth=2.5, alpha=0.8, label=f'Mean: {mean_val:.3f}', zorder=10)

        # 添加中位数线
        if show_median:
            ax.axvline(median_val, color='#F39C12', linestyle='--',
                       linewidth=2.5, alpha=0.8, label=f'Median: {median_val:.3f}', zorder=10)

        # 添加标准差区域
        if show_std:
            ax.axvspan(mean_val - std_val, mean_val + std_val,
                       alpha=0.15, color='#9B59B6', label=f'±1 SD: {std_val:.3f}', zorder=1)

    # # 添加统计信息文本框
    # if show_mean or show_median or show_std:
    #     stats_text = []
    #     stats_text.append(f'n = {len(data)}')
    #     if show_mean:
    #         stats_text.append(f'μ = {mean_val:.3f}')
    #     if show_median:
    #         stats_text.append(f'Median = {median_val:.3f}')
    #     if show_std:
    #         stats_text.append(f'σ = {std_val:.3f}')
    #
    #     textstr = '\n'.join(stats_text)
    #     props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    #     ax.text(0.75, 0.95, textstr, transform=ax.transAxes,
    #             fontsize=10, verticalalignment='top', bbox=props)

    # 设置标签和标题
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='semibold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold')
    else:
        if not vertical:
            ax.set_ylabel(stat.capitalize(), fontsize=12, fontweight='semibold')
        else:
            ax.set_xlabel(stat.capitalize(), fontsize=12, fontweight='semibold')

    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # 添加图例
    if show_mean or show_median or show_std:
        ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)

    # 优化网格和边框
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, axis='y' if not vertical else 'x')
    ax.set_axisbelow(True)
    sns.despine(left=False, bottom=False)

    # 调整布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    return fig, ax


def plot_histogram_with_line(
        x_hist: np.ndarray,
        x_line: np.ndarray,
        y_line: np.ndarray,
        hist_label: Optional[str] = None,
        line_label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel_hist: Optional[str] = None,
        ylabel_line: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6),
        bins: Union[int, str] = 'auto',
        hist_color: str = '#3498DB',
        line_color: str = '#E74C3C',
        hist_alpha: float = 0.6,
        line_alpha: float = 0.9,
        show_markers: bool = True,
        marker_size: int = 6,
        line_width: float = 2.5,
        style: str = 'darkgrid',
        stat: Literal['count', 'frequency', 'density', 'probability'] = 'count',
        show_grid: bool = True,
        dpi: int = 300
):
    """
    创建直方图+折线图的组合图（共享X轴，双Y轴）

    Parameters:
    -----------
    x_hist : np.ndarray
        直方图数据（一维数组）
    x_line : np.ndarray
        折线图的X轴数据
    y_line : np.ndarray
        折线图的Y轴数据
    hist_label : str, optional
        直方图图例标签
    line_label : str, optional
        折线图图例标签
    xlabel : str, optional
        X轴标签
    ylabel_hist : str, optional
        左Y轴标签（直方图）
    ylabel_line : str, optional
        右Y轴标签（折线图）
    title : str, optional
        图表标题
    save_path : str, optional
        保存路径
    figsize : tuple, default=(10, 6)
        图表尺寸
    bins : int or str, default='auto'
        直方图分箱数量
    hist_color : str, default='#3498DB'
        直方图颜色
    line_color : str, default='#E74C3C'
        折线颜色
    hist_alpha : float, default=0.6
        直方图透明度
    line_alpha : float, default=0.9
        折线透明度
    show_markers : bool, default=True
        是否显示折线的标记点
    marker_size : int, default=6
        标记点大小
    line_width : float, default=2.5
        折线宽度
    style : str, default='darkgrid'
        seaborn样式
    stat : str, default='count'
        直方图统计类型
    show_grid : bool, default=True
        是否显示网格
    dpi : int, default=300
        保存分辨率

    Returns:
    --------
    fig, ax1, ax2 : matplotlib figure and axes objects

    Example:
    --------
    >>> hist_data = np.random.randn(1000)
    >>> x = np.arange(10)
    >>> y = np.random.rand(10) * 100
    >>> plot_histogram_with_line(hist_data, x, y,
    ...                           hist_label="Distribution",
    ...                           line_label="Metric")
    """
    # 设置样式
    sns.set_theme(style=style, palette="muted")

    # 创建图表和主坐标轴
    fig, ax1 = plt.subplots(figsize=figsize)

    # === 绘制直方图（左Y轴）===
    sns.histplot(
        x=x_hist,
        bins=bins,
        stat=stat,
        color=hist_color,
        alpha=hist_alpha,
        edgecolor='white',
        linewidth=1.2,
        ax=ax1,
        label=hist_label if hist_label else 'Histogram'
    )

    # 设置左Y轴
    ax1.set_xlabel(xlabel if xlabel else 'Value', fontsize=12, fontweight='semibold')
    ax1.set_ylabel(ylabel_hist if ylabel_hist else stat.capitalize(),
                   fontsize=12, fontweight='semibold', color=hist_color)
    ax1.tick_params(axis='y', labelcolor=hist_color, labelsize=10)

    # === 创建右Y轴并绘制折线图 ===
    ax2 = ax1.twinx()

    marker_style = 'o' if show_markers else None
    ax2.plot(
        x_line, y_line,
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        marker=marker_style,
        markersize=marker_size,
        markeredgecolor='white',
        markeredgewidth=1.5,
        label=line_label if line_label else 'Line',
        zorder=10
    )

    # 设置右Y轴
    ax2.set_ylabel(ylabel_line if ylabel_line else 'Value',
                   fontsize=12, fontweight='semibold', color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color, labelsize=10)

    # === 智能Y轴缩放 ===
    # 为直方图留出顶部空间（避免与折线重叠）
    y1_min, y1_max = ax1.get_ylim()
    ax1.set_ylim(y1_min, y1_max * 1.15)

    # 为折线图设置合理范围（带padding）
    y2_min, y2_max = np.min(y_line), np.max(y_line)
    y2_range = y2_max - y2_min
    padding = y2_range * 0.1 if y2_range > 0 else 1
    ax2.set_ylim(y2_min - padding, y2_max + padding)

    # === 设置标题 ===
    if title:
        ax1.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # === 组合图例 ===
    # 获取两个轴的图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # 在顶部中央显示组合图例
    ax1.legend(h1 + h2, l1 + l2,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.08),
               ncol=2,
               frameon=True,
               shadow=True,
               fontsize=11,
               fancybox=True)

    # === 网格设置 ===
    if show_grid:
        ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, axis='y')
        ax1.set_axisbelow(True)

    # === 边框美化 ===
    ax1.spines['left'].set_color(hist_color)
    ax1.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_color(line_color)
    ax2.spines['right'].set_linewidth(2)
    ax1.spines['top'].set_visible(False)

    # 调整布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    return fig, ax1, ax2


def plot_dual_histogram_lines(
        x_shared: np.ndarray,
        y_hist: np.ndarray,
        y_line: np.ndarray,
        hist_label: Optional[str] = None,
        line_label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel_hist: Optional[str] = None,
        ylabel_line: Optional[str] = None,
        bin_labels: Optional[list[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6),
        bins: Union[int, np.ndarray] = 20,
        hist_color: str = '#3498DB',
        line_color: str = '#E74C3C',
        hist_alpha: float = 0.6,
        line_alpha: float = 0.9,
        show_markers: bool = True,
        show_legend: bool = False,
        marker_size: int = 6,
        line_width: float = 2.5,
        style: str = 'darkgrid',
        show_grid: bool = True,
        dpi: int = 300
):
    """
    创建共享X轴的直方图+折线图组合（X轴完全对齐）
    适合：横轴是分类或bin区间，有对应的频数（直方图）和指标值（折线）

    Parameters:
    -----------
    x_shared : np.ndarray
        共享的X轴数据（例如：bin中心、类别）
    y_hist : np.ndarray
        直方图的高度（频数、计数等）
    y_line : np.ndarray
        折线图的Y值
    其他参数同 plot_histogram_with_line

    Example:
    --------
    >>> bins = np.arange(0, 100, 10)
    >>> counts = np.random.randint(10, 100, len(bins))
    >>> metric = np.random.rand(len(bins)) * 50
    >>> plot_dual_histogram_lines(bins, counts, metric)
    """
    # 设置样式
    sns.set_theme(style=style, palette="muted")

    # 创建图表
    fig, ax1 = plt.subplots(figsize=figsize, dpi=300)

    # === 绘制条形图（直方图风格）===
    width = (x_shared[1] - x_shared[0]) * 0.8 if len(x_shared) > 1 else 0.8
    ax1.bar(
        x_shared, y_hist,
        width=width,
        color=hist_color,
        alpha=hist_alpha,
        edgecolor='white',
        linewidth=1.2,
        label=hist_label if hist_label else 'Histogram'
    )

    # === X轴 bin 名称 ===
    if bin_labels is not None:
        assert len(bin_labels) == len(x_shared), "bin_labels should has the same dim with x_shared"
        ax1.set_xticks(x_shared)
        ax1.set_xticklabels(bin_labels, rotation=90, ha='right', fontsize=10)

    # 设置左Y轴
    if xlabel is not None:
        ax1.set_xlabel(xlabel, fontsize=12, fontweight='semibold')
    if ylabel_hist is not None:
        ax1.set_ylabel(ylabel_hist if ylabel_hist else 'Count',
                   fontsize=12, fontweight='semibold', color=hist_color)
    ax1.tick_params(axis='y', labelcolor=hist_color, labelsize=10)

    # === 创建右Y轴并绘制折线 ===
    ax2 = ax1.twinx()

    marker_style = 'o' if show_markers else None
    ax2.plot(
        x_shared, y_line,
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        marker=marker_style,
        markersize=marker_size,
        markeredgecolor='white',
        markeredgewidth=1.5,
        label=line_label if line_label else 'Metric',
        zorder=10
    )

    # 设置右Y轴
    if ylabel_line is not None:
        ax2.set_ylabel(ylabel_line if ylabel_line else 'Metric Value',
                   fontsize=12, fontweight='semibold', color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color, labelsize=10)

    # === 智能Y轴缩放 ===
    y1_min, y1_max = 0, np.max(y_hist)
    ax1.set_ylim(0, y1_max * 1.15)

    y2_min, y2_max = np.min(y_line), np.max(y_line)
    y2_range = y2_max - y2_min
    padding = y2_range * 0.1 if y2_range > 0 else 1
    ax2.set_ylim(y2_min - padding, y2_max + padding)

    # === 标题 ===
    if title:
        ax1.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # === 组合图例 ===
    if show_legend:
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.08),
               ncol=2,
               frameon=True,
               shadow=True,
               fontsize=11,
               fancybox=True)

    # === 网格和边框 ===
    if show_grid:
        ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, axis='y')
        ax1.set_axisbelow(True)

    ax1.spines['left'].set_color(hist_color)
    ax1.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_color(line_color)
    ax2.spines['right'].set_linewidth(2)
    ax1.spines['top'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    return fig, ax1, ax2