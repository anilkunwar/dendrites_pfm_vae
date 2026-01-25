#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3x3 图像网格 + 每张图两个分数的可视化示例
- 每个子图：显示图片 + 半透明信息框（score_a / score_b）
- 额外：在每张图左下角画一个小条形“微型可视化”对比两分数（可选但很直观）
- 全局：标题 + 统一色条（按 score_a 映射边框颜色，强化对比）
"""

from __future__ import annotations

from typing import Union, Sequence, List, Tuple, Optional
import os
import glob
import random
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["image.cmap"] = "coolwarm"

from src.evaluate_metrics import generate_analysis_figure, get_severity_level


# ---------- 1) 占位符函数：传入图片，返回两个浮点分数 ----------
def placeholder_score_fn(img: np.ndarray) -> Tuple[str, str]:

    _, metrics, s = generate_analysis_figure(np.clip(img, 0, 1))

    return f"{metrics['fractal_dimension']:.2f}", f"{s['empirical_score']:.2f}({get_severity_level(s['empirical_score'])})"


def load_images_from_glob(
    data_dir_or_pattern: Union[str, Sequence[str]],
    *,
    k: int = 9,
    recursive: bool = True,
    seed: Optional[int] = None,
) -> List[Tuple[str, np.ndarray]]:
    """
    Randomly load k images from npy files resolved by directory / glob / list.

    Parameters
    ----------
    data_dir_or_pattern : str | Sequence[str]
        - directory → search for **/*.npy
        - glob pattern → glob directly
        - list / tuple → treated as file paths
    k : int
        Number of images to load
    recursive : bool
        Whether to use recursive glob (**)
    seed : int | None
        Random seed

    Returns
    -------
    list of (path, image_array)
        image_array is uint8 RGB, ready for matplotlib.imshow
    """
    if seed is not None:
        random.seed(seed)

    # -------- resolve candidate paths --------
    if isinstance(data_dir_or_pattern, (list, tuple)):
        candidates = list(data_dir_or_pattern)
    else:
        s = str(data_dir_or_pattern)
        if os.path.isdir(s):
            pattern = os.path.join(s, "**", "**", "*.npy") if recursive else os.path.join(s, "*.npy")
            candidates = glob.glob(pattern, recursive=recursive)
        else:
            candidates = glob.glob(s, recursive=recursive)

    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError(f"No .npy files found from: {data_dir_or_pattern}")

    # -------- sample --------
    k = min(int(k), len(candidates))
    chosen = random.sample(candidates, k=k)

    images: List[Tuple[str, np.ndarray]] = []

    for p in chosen:
        arr = np.load(p)

        images.append((p, arr[..., 0]))

    return images

# ---------- 3) 绘图主函数 ----------
def plot_grid_with_scores(
    images: List[np.ndarray],
    score_fn,
    title: str = "Two Scores Comparison per Image",
    score_names: Tuple[str, str] = ("Fractal Dimension", "Dendrite Intensity Score"),
):
    assert len(images) == 9, "需要正好 9 张图片来画 3×3。"

    # 计算分数
    scores = [score_fn(img) for img in images]

    # 布局：更紧凑但不挤
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    for i, ax in enumerate(axes.flat):
        img = images[i]
        a, b = scores[i]

        ax.imshow(img)
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])

        # 半透明信息框（右上角）
        text = f"{score_names[0]}: {a}\n{score_names[1]}: {b}"
        ax.text(
            0.98,
            0.02,
            text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc="black", ec="none", alpha=0.55),
        )

        # 每个子图的小标题：#idx
        ax.set_title(f"#{i+1}", fontsize=11, pad=6)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


# ---------- 4) 入口 ----------
def main():
    items = load_images_from_glob("data")

    paths, images = zip(*items)

    plot_grid_with_scores(
        images=list(images),
        score_fn=placeholder_score_fn,
    )


if __name__ == "__main__":
    main()
