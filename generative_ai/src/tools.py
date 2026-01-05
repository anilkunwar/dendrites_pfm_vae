import numpy as np
from skimage.filters import threshold_otsu

def fractal_dimension_boxcount(img, threshold=None, n_scales=8, min_box_size=2):
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(float)
    if threshold is None:
        threshold = threshold_otsu(img)
    Z = img > threshold
    p = min(Z.shape)
    max_box_size = max(min_box_size + 1, p // 2)
    sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size),
                        num=n_scales, dtype=int)
    sizes = np.unique(sizes)
    counts = []
    for size in sizes:
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], size), axis=0),
            np.arange(0, Z.shape[1], size), axis=1)
        counts.append(np.count_nonzero(S))
    counts = np.array(counts)
    valid = counts > 0
    sizes = sizes[valid]
    counts = counts[valid]
    if len(sizes) < 2:
        return np.nan, sizes, counts
    log_inv_eps = np.log(1.0 / sizes.astype(float))
    log_N = np.log(counts.astype(float))
    coeffs = np.polyfit(log_inv_eps, log_N, 1)
    return coeffs[0], sizes, counts