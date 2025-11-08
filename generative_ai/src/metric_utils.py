"""metrics_utils.py — Utility functions for quantitative evaluation of microstructure learning models.

This module implements standard image quality metrics (MSE, RMSE, PSNR, SSIM) and
material-science-specific metrics (μSIM, fractal dimension, blob count, shape index)
for analyzing dendrite evolution and structural features in ConvLSTM or VAE models.
"""

import numpy as np
from skimage import metrics, measure, morphology
from scipy.ndimage import label

# === Basic metrics ===
def mse(gt, pred):
    """Mean Squared Error."""
    return metrics.mean_squared_error(gt, pred)

def rmse(gt, pred):
    """Normalized Root Mean Squared Error."""
    return metrics.normalized_root_mse(gt, pred)

def psnr(gt, pred):
    """Peak Signal-to-Noise Ratio."""
    return metrics.peak_signal_noise_ratio(gt, pred)

def ssim(gt, pred, win_size=7):
    """Structural Similarity Index."""
    return metrics.structural_similarity(gt, pred, win_size=win_size)

# === μSIM: Micro-Structural Similarity Index ===
def micro_ssim(gt, pred, mask, win_size=7):
    """Compute SSIM focusing only on microstructure interfaces (mask==1)."""
    gt_local = gt * mask
    pred_local = pred * mask
    return metrics.structural_similarity(gt_local, pred_local, win_size=win_size)

# === Fractal Dimension ===
def fractal_dimension(Z, threshold=0.5):
    """Estimate fractal dimension via box-counting method."""
    Z = Z < threshold
    p = min(Z.shape)
    sizes = 2 ** np.arange(1, int(np.log2(p)), 1)
    counts = []
    for s in sizes:
        reduced = measure.block_reduce(Z, (s, s), np.max)
        counts.append(np.sum(reduced))
    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    return -coeffs[0]

# === Blob Count ===
def blob_count(binary_img):
    """Count the number of connected blobs in a binary image."""
    labeled, num = label(binary_img > 0.5)
    return num

# === Shape Index ===
def shape_index(mean_curvature, gauss_curvature):
    """Compute shape index based on curvature fields."""
    numerator = (mean_curvature**2 - gauss_curvature)
    denominator = (mean_curvature**2 + gauss_curvature + 1e-8)
    return (2/np.pi) * np.arctan(numerator / denominator)