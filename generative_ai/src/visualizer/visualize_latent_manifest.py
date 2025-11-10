import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':

    model = torch.load()

    """
    Visualize latent manifolds and feature maps from the encoder.
    Fixed version by ChatGPT.
    """
    z_mean, z_sd = model.encode(x)
    z_mean = z_mean.detach().cpu()

    # 假设数据能重构成正方形
    d1 = int(np.sqrt(x.shape[0]))
    d2 = d1

    print("Latent manifolds and images at current epoch")

    # (2) 创建绘图
    fig = plt.figure(constrained_layout=True, figsize=(25, 12))
    gs = fig.add_gridspec(2, 3, wspace=0.8)

    # --- ① 潜在空间散点与密度 ---
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(z_mean[:, -2], z_mean[:, -1], s=10, alpha=.3)
    sns.kdeplot(
        x=z_mean[:, -2],
        y=z_mean[:, -1],
        cmap="Reds",
        fill=True,
        alpha=0.4,
        ax=ax1
    )
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_xlabel('$z_1$')
    ax1.set_ylabel('$z_2$')

    # --- ② 绘制潜在特征图像 ---
    # 假设前四个维度可重构为 2×2 网格
    for i, title in enumerate(['$s_1$', '$s_2$', '$z_1$', '$z_2$']):
        if i >= z_mean.shape[1]:
            break
        ax = fig.add_subplot(gs[i // 2, 1 + (i % 2)])
        z_img = z_mean[:, i].reshape(d1, d2)
        im = ax.imshow(z_img, origin='lower', cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        ax.axis('off')
        ax.set_title(title)

    plt.show()
