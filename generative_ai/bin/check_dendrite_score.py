#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-row interface images + metric trend plots (clean, publication-style)
"""

from __future__ import annotations
from typing import Union, Sequence, List, Tuple, Optional

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from src.evaluate_metrics import generate_analysis_figure


# ============================================================
# 1. Metric computation (renamed, no placeholder)
# ============================================================
def compute_interface_metrics(img: np.ndarray) -> Tuple[float, float]:
    """
    Parameters
    ----------
    img : np.ndarray
        2D image array

    Returns
    -------
    fractal_dimension : float
    dendrite_intensity_score : float
    """
    _, metrics, score = generate_analysis_figure(np.clip(img, 0, 1))
    return float(metrics["fractal_dimension"]), float(score["empirical_score"])


# ============================================================
# 2. Load images
# ============================================================
def load_images_from_glob(
    data_dir_or_pattern: Union[str, Sequence[str]],
    *,
    k: int = 9,
    recursive: bool = True,
    seed: Optional[int] = None,
) -> List[np.ndarray]:

    if seed is not None:
        random.seed(seed)

    if isinstance(data_dir_or_pattern, (list, tuple)):
        candidates = list(data_dir_or_pattern)
    else:
        s = str(data_dir_or_pattern)
        if os.path.isdir(s):
            pattern = os.path.join(s, "**", "*.npy") if recursive else os.path.join(s, "*.npy")
            candidates = glob.glob(pattern, recursive=recursive)
        else:
            candidates = glob.glob(s, recursive=recursive)

    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError("No .npy files found")

    k = min(k, len(candidates))
    chosen = random.sample(candidates, k)

    images = []
    for p in chosen:
        arr = np.load(p)
        images.append(arr[..., 0])

    return images


# ============================================================
# 3. Visualization
# ============================================================
def plot_row_images_with_metric_trends(
    images: List[np.ndarray],
    metric_fn,
    *,
    dpi: int = 150,
):
    """
    Top: 1 row of images (no labels, no titles)
    Bottom: two metric trends (solid dots + dashed line, no titles)
    """

    # ---- compute metrics ----
    metrics = [metric_fn(img) for img in images]
    fractal_dims = np.array([m[0] for m in metrics])
    dendrite_scores = np.array([m[1] for m in metrics])

    n = len(images)
    x = np.arange(1, n + 1)

    fig_w = max(10, 1.35 * n)
    fig_h = 6.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[2.2, 1.8, 1.8],
        hspace=0.15,
    )

    # =======================
    # Image row
    # =======================
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_axis_off()

    gap = 0.01
    cell_w = (1.0 - gap * (n - 1)) / n

    for i, img in enumerate(images):
        x0 = i * (cell_w + gap)
        iax = ax_img.inset_axes([x0, 0, cell_w, 1.0])
        iax.imshow(img)
        iax.set_xticks([])
        iax.set_yticks([])
        for s in iax.spines.values():
            s.set_visible(False)

    # =======================
    # Fractal dimension trend
    # =======================
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(
        x,
        fractal_dims,
        linestyle="--",
        marker="o",
        markersize=5,
        linewidth=1.5,
    )
    ax1.set_ylabel("Fractal dimension")
    ax1.set_xlim(0.5, n + 0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.grid(True, alpha=0.15)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # =======================
    # Dendrite intensity trend
    # =======================
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax2.plot(
        x,
        dendrite_scores,
        linestyle="--",
        marker="o",
        markersize=5,
        linewidth=1.5,
    )
    ax2.set_ylabel("Dendrite intensity score")
    ax2.set_xlabel("Figure ID")
    ax2.set_xlim(0.5, n + 0.5)
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.08)
    plt.show()


# ============================================================
# 4. Entry point
# ============================================================
def main():
    images = load_images_from_glob("data", k=9, seed=0)

    plot_row_images_with_metric_trends(
        images=images,
        metric_fn=compute_interface_metrics,
    )


if __name__ == "__main__":
    main()
