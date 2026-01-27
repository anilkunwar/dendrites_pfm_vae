#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Union, Sequence, List, Tuple

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe

from src.evaluate_metrics import generate_analysis_figure


# ============================================================
# Reproducibility & style
# ============================================================
SEED = 160
random.seed(SEED)
np.random.seed(SEED)

sns.set_theme(
    context="paper",
    style="white",
    font_scale=1.05,
)


# ============================================================
# Utilities
# ============================================================
def normalize_to_01(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    img = img.astype(float)
    mn, mx = img.min(), img.max()
    if mx - mn < eps:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def minmax_01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn) + 1


def compute_interface_metrics(img_01: np.ndarray) -> Tuple[float, float]:
    _, metrics, score = generate_analysis_figure(np.clip(img_01, 0, 1))
    return float(metrics["fractal_dimension"]), float(score["empirical_score"])


def load_images_from_glob(
    data_dir_or_pattern: Union[str, Sequence[str]],
    *,
    k: int = 9,
    recursive: bool = True,
) -> List[np.ndarray]:

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
# Plot
# ============================================================
def plot_images_with_seaborn_bars(
    images: List[np.ndarray],
    *,
    cmap: str = "coolwarm",
    dpi: int = 160,
):
    n = len(images)
    ids = np.arange(1, n + 1)

    # normalize images + compute metrics
    images_01 = [normalize_to_01(im) for im in images]
    metrics = [compute_interface_metrics(im) for im in images_01]
    fractal_raw = np.array([m[0] for m in metrics])
    dendrite_raw = np.array([m[1] for m in metrics])

    # 🔽 归一化后的柱高
    fractal_norm = minmax_01(fractal_raw)
    dendrite_norm = minmax_01(dendrite_raw)

    fig_w = max(10, 1.25 * n)
    fig = plt.figure(figsize=(fig_w, 6.2), dpi=dpi)
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.45, 2.2, 1.45],
        hspace=0.08,
    )

    # -------------------------------
    # Top: Fractal dimension (normalized bars)
    # -------------------------------
    ax_top = fig.add_subplot(gs[0, 0])
    sns.barplot(
        x=ids,
        y=fractal_norm,
        ax=ax_top,
        color=sns.color_palette("deep")[0],
    )

    for i, (v_raw, v_norm) in enumerate(zip(fractal_raw, fractal_norm)):
        ax_top.text(
            i, v_norm,
            f"{v_raw:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax_top.set_ylabel("Fractal dimension")
    ax_top.set_xlabel("Figure ID")
    ax_top.set_yticks([])
    ax_top.spines[["left", "right", "top"]].set_visible(False)

    # -------------------------------
    # Middle: morphology images
    # -------------------------------
    ax_mid = fig.add_subplot(gs[1, 0])
    ax_mid.set_axis_off()

    gap = 0.01
    cell_w = (1.0 - gap * (n - 1)) / n

    for i, im in enumerate(images_01):
        x0 = i * (cell_w + gap)
        iax = ax_mid.inset_axes([x0, 0, cell_w, 1])
        iax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        iax.set_xticks([])
        iax.set_yticks([])
        for s in iax.spines.values():
            s.set_visible(False)

        txt = iax.text(
            0.97, 0.95, f"{i+1}",
            transform=iax.transAxes,
            ha="right", va="top",
            fontsize=16,
            color="white",
            fontfamily="cursive",
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=2.5, foreground="black"),
            pe.Normal(),
        ])

    # -------------------------------
    # Bottom: Dendrite intensity (normalized bars)
    # -------------------------------
    ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)
    sns.barplot(
        x=ids,
        y=dendrite_norm,
        ax=ax_bot,
        color=sns.color_palette("deep")[1],
    )

    for i, (v_raw, v_norm) in enumerate(zip(dendrite_raw, dendrite_norm)):
        ax_bot.text(
            i, v_norm,
            f"{v_raw:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax_bot.set_ylabel("Dendrite intensity")
    ax_bot.set_xlabel("Figure ID")
    ax_bot.set_yticks([])
    ax_bot.spines[["left", "right", "top"]].set_visible(False)

    fig.subplots_adjust(left=0.05, right=0.995, top=0.97, bottom=0.10)
    plt.show()


# ============================================================
# Entry
# ============================================================
def main():
    images = load_images_from_glob("../data", k=9)
    plot_images_with_seaborn_bars(images)


if __name__ == "__main__":
    main()
