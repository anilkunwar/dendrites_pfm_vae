"""
Comprehensive dendrite analysis
"""
import os
import glob
import random
from typing import Dict, Optional, Sequence, Tuple, Union, List, Any

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import pdist
from skimage import measure, morphology
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

class ComprehensiveDendriteAnalyzer:

    def __init__(self, image, pixel_size_um=1.0):
        if len(image.shape) == 3:
            self.image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            self.image_rgb = image.copy()
        else:
            self.image_gray = image.copy()
            self.image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        self.pixel_size = float(pixel_size_um)

        # Binarization (assumes foreground darker than background)
        thresh = threshold_otsu(self.image_gray)
        self.binary = self.image_gray > thresh

        self.skeleton = morphology.skeletonize(self.binary)
        self.dist_transform = ndimage.distance_transform_edt(self.binary)

        M = cv2.moments(self.binary.astype(np.uint8))
        if M["m00"] != 0:
            self.centroid_x = int(M["m10"] / M["m00"])
            self.centroid_y = int(M["m01"] / M["m00"])
        else:
            self.centroid_y, self.centroid_x = np.array(self.binary.shape) // 2

    # ----------------------------
    # Original 10 metrics
    # ----------------------------
    def branching_density(self) -> float:
        skeleton = self.skeleton
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode="constant")
        branch_points = np.sum((neighbor_count >= 13) & skeleton)
        total_area = np.sum(self.binary)
        if total_area == 0:
            return 0.0
        return branch_points / total_area * 10000

    def tip_density(self) -> float:
        skeleton = self.skeleton
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode="constant")
        tip_points = np.sum((neighbor_count == 11) & skeleton)
        total_area = np.sum(self.binary)
        if total_area == 0:
            return 0.0
        return tip_points / total_area * 10000

    def growth_asymmetry(self) -> float:
        cx, cy = self.centroid_x, self.centroid_y
        h, w = self.binary.shape
        q1 = np.sum(self.binary[:cy, cx:])
        q2 = np.sum(self.binary[:cy, :cx])
        q3 = np.sum(self.binary[cy:, :cx])
        q4 = np.sum(self.binary[cy:, cx:])
        quadrants = np.array([q1, q2, q3, q4])
        total = np.sum(quadrants)
        if total == 0:
            return 0.0
        asymmetry = np.std(quadrants) / (np.mean(quadrants) + 1e-6)
        return min(asymmetry, 1.0)

    def perimeter_complexity(self) -> float:
        contours, _ = cv2.findContours(self.binary.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return 0.0
        max_perimeter = max([cv2.arcLength(c, True) for c in contours])
        area = np.sum(self.binary)
        if area == 0:
            return 0.0
        return max_perimeter / np.sqrt(area)

    def growth_compactness(self) -> float:
        area = np.sum(self.binary)
        if area == 0:
            return 1.0
        contours, _ = cv2.findContours(self.binary.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return 1.0
        max_perimeter = max([cv2.arcLength(c, True) for c in contours])
        compactness = 4 * np.pi * area / (max_perimeter ** 2 + 1e-6)
        return min(compactness, 1.0)

    def radial_growth_variance(self) -> float:
        cx, cy = self.centroid_x, self.centroid_y
        contours, _ = cv2.findContours(self.binary.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return 0.0
        distances = []
        for contour in contours:
            for point in contour:
                x, y = point[0]
                distances.append(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))
        if not distances:
            return 0.0
        return float(np.var(distances))

    def skeleton_tortuosity(self) -> float:
        skeleton = self.skeleton
        skeleton_length = np.sum(skeleton)
        if skeleton_length == 0:
            return 0.0
        coords = np.argwhere(skeleton)
        if len(coords) < 2:
            return 0.0
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        diagonal = np.linalg.norm(max_coords - min_coords)
        if diagonal == 0:
            return 0.0
        return float(skeleton_length / diagonal)

    def multiscale_entropy(self, scales=(1, 2, 4, 8)) -> float:
        entropies = []
        for scale in scales:
            h, w = self.binary.shape
            new_h, new_w = h // scale, w // scale
            if new_h < 2 or new_w < 2:
                continue
            scaled = cv2.resize(self.binary.astype(np.uint8),
                                (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST)
            hist, _ = np.histogram(scaled, bins=2, range=(0, 1))
            hist = hist / (hist.sum() + 1e-6)
            entropy = -np.sum(hist * np.log2(hist + 1e-6))
            entropies.append(float(entropy))
        return float(np.mean(entropies)) if entropies else 0.0

    def interface_roughness(self) -> float:
        # NOTE: if image_gray is already [0..255], multiplying by 255 may overflow. Consider normalizing upstream if needed.
        edges = cv2.Canny((self.image_gray * 255).astype(np.uint8), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return 0.0

        roughness_values = []
        for contour in contours:
            contour = contour.reshape(-1, 2)
            if len(contour) < 3:
                continue
            vectors = np.diff(contour, axis=0)
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            angle_diff = np.abs(np.diff(angles))
            angle_diff[angle_diff > np.pi] = 2 * np.pi - angle_diff[angle_diff > np.pi]
            roughness_values.extend(angle_diff.tolist())

        return float(np.std(roughness_values)) if roughness_values else 0.0

    def shape_index(self):
        gy, gx = np.gradient(self.image_gray)
        gyy, gyx = np.gradient(gy)
        gxy, gxx = np.gradient(gx)
        H = (gxx + gyy) / 2.0
        K = gxx * gyy - gxy * gyx
        eta = np.arctan2(H, np.sqrt(np.abs(H ** 2 - K))) / np.pi + 0.5
        return eta

    # ----------------------------
    # Literature-ish metrics
    # ----------------------------
    def secondary_dendrite_arm_spacing_linear(self) -> float:
        skeleton = self.skeleton
        h, w = skeleton.shape
        spacings = []

        for y in range(0, h, 10):
            row = skeleton[y, :]
            transitions = np.diff(row.astype(int))
            arm_positions = np.where(np.abs(transitions) > 0)[0]
            if len(arm_positions) > 1:
                spacings.extend(np.diff(arm_positions).tolist())

        for x in range(0, w, 10):
            col = skeleton[:, x]
            transitions = np.diff(col.astype(int))
            arm_positions = np.where(np.abs(transitions) > 0)[0]
            if len(arm_positions) > 1:
                spacings.extend(np.diff(arm_positions).tolist())

        if not spacings:
            return 0.0
        return float(np.median(spacings) * self.pixel_size)

    def secondary_dendrite_arm_spacing_transform(self) -> float:
        local_max = morphology.local_maxima(self.dist_transform)
        coords = np.argwhere(local_max)
        if len(coords) < 2:
            return 0.0
        distances = pdist(coords)
        return float(np.median(distances) * self.pixel_size) if len(distances) > 0 else 0.0

    def primary_dendrite_arm_spacing(self) -> float:
        labeled = measure.label(self.skeleton)
        regions = measure.regionprops(labeled)
        if len(regions) == 0:
            return 0.0
        centroids = [r.centroid for r in regions if r.area > 50]
        if len(centroids) < 2:
            return 0.0
        centroids = np.array(centroids)
        distances = pdist(centroids)
        return float(np.median(distances) * self.pixel_size) if len(distances) > 0 else 0.0

    def dendrite_tip_velocity_indicator(self) -> float:
        skeleton = self.skeleton
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode="constant")
        tips = (neighbor_count == 11) & skeleton
        tip_coords = np.argwhere(tips)
        if len(tip_coords) == 0:
            return 0.0

        sharpness_values = []
        for y, x in tip_coords:
            y1, y2 = max(0, y - 5), min(skeleton.shape[0], y + 6)
            x1, x2 = max(0, x - 5), min(skeleton.shape[1], x + 6)
            local_region = skeleton[y1:y2, x1:x2]
            density = np.sum(local_region) / local_region.size
            sharpness_values.append(1 - density)
        return float(np.mean(sharpness_values)) if sharpness_values else 0.0

    def dendrite_tip_radius(self) -> float:
        skeleton = self.skeleton
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode="constant")
        tips = (neighbor_count == 11) & skeleton
        tip_radii = self.dist_transform[tips]
        if len(tip_radii) == 0:
            return 0.0
        return float(np.mean(tip_radii) * self.pixel_size)

    def sholl_analysis(self, center=None, step_size=10) -> Dict[str, object]:
        skeleton = self.skeleton
        if center is None:
            cx, cy = self.centroid_x, self.centroid_y
        else:
            cx, cy = center

        h, w = skeleton.shape
        max_radius = int(np.sqrt((h - cy) ** 2 + (w - cx) ** 2))

        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

        radii = np.arange(step_size, max_radius, step_size)
        intersections = []
        for r in radii:
            ring_mask = (distances >= r - step_size / 2) & (distances < r + step_size / 2)
            intersections.append(int(np.sum(skeleton & ring_mask)))
        intersections = np.array(intersections)

        if len(intersections) > 0:
            max_intersections = int(np.max(intersections))
            critical_radius = float(radii[np.argmax(intersections)] if max_intersections > 0 else 0)
            primary_dendrites = self._count_primary_dendrites(cy, cx)
            ramification_index = float(max_intersections / (primary_dendrites + 1))

            if len(intersections) > 1:
                valid_idx = intersections > 0
                if np.sum(valid_idx) > 1:
                    log_density = np.log10(intersections[valid_idx] / (np.pi * radii[valid_idx] ** 2) + 1e-6)
                    regression_coef = float(np.polyfit(radii[valid_idx], log_density, 1)[0])
                else:
                    regression_coef = 0.0
            else:
                regression_coef = 0.0
        else:
            max_intersections = 0
            critical_radius = 0.0
            ramification_index = 0.0
            regression_coef = 0.0

        return {
            "radii": radii * self.pixel_size,
            "intersections": intersections,
            "max_intersections": max_intersections,
            "critical_radius": critical_radius * self.pixel_size,
            "ramification_index": ramification_index,
            "regression_coefficient": regression_coef,
            "total_intersections": int(np.sum(intersections)),
        }

    def _count_primary_dendrites(self, cy, cx, radius=20) -> int:
        h, w = self.skeleton.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        ring_mask = (distances >= radius - 2) & (distances < radius + 2)
        ring_skeleton = self.skeleton & ring_mask
        labeled = measure.label(ring_skeleton)
        return int(len(np.unique(labeled)) - 1)

    def dendrite_density(self) -> float:
        total_pixels = self.binary.size
        dendrite_pixels = np.sum(self.binary)
        return float(dendrite_pixels / total_pixels)

    def principal_curvatures(self, num_samples=100) -> Dict[str, float]:
        contours, _ = cv2.findContours(
            self.binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            return {"mean_curvature": 0.0, "curvature_variance": 0.0, "max_curvature": 0.0}

        main_contour = max(contours, key=cv2.contourArea)
        if len(main_contour) < 3:
            return {"mean_curvature": 0.0, "curvature_variance": 0.0, "max_curvature": 0.0}

        curvatures = []
        contour = main_contour.reshape(-1, 2)
        step = max(1, len(contour) // num_samples)

        for i in range(0, len(contour) - 2, step):
            p1 = contour[i]
            p2 = contour[i + 1]
            p3 = contour[min(i + 2, len(contour) - 1)]

            area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) -
                (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
            d1 = np.linalg.norm(p2 - p1)
            d2 = np.linalg.norm(p3 - p2)
            d3 = np.linalg.norm(p3 - p1)

            if d1 * d2 * d3 > 0:
                curvature = 4 * area / (d1 * d2 * d3)
                curvatures.append(float(curvature))

        if not curvatures:
            return {"mean_curvature": 0.0, "curvature_variance": 0.0, "max_curvature": 0.0}

        return {
            "mean_curvature": float(np.mean(curvatures)),
            "curvature_variance": float(np.var(curvatures)),
            "max_curvature": float(np.max(curvatures)),
        }

    def persistent_homology_features(self) -> Dict[str, float]:
        labeled = measure.label(self.binary)
        num_components = int(len(np.unique(labeled)) - 1)

        filled = ndimage.binary_fill_holes(self.binary)
        holes = filled & ~self.binary
        labeled_holes = measure.label(holes)
        num_holes = int(len(np.unique(labeled_holes)) - 1)

        euler_characteristic = int(num_components - num_holes)

        return {
            "num_components": num_components,
            "num_holes": num_holes,
            "euler_characteristic": euler_characteristic,
            "betti_0": num_components,
            "betti_1": num_holes,
        }

    def fractal_dimension_boxcount(self, max_box_size=None, min_box_size=2) -> float:
        binary = self.binary
        if max_box_size is None:
            max_box_size = min(binary.shape) // 4

        sizes, counts = [], []
        for box_size in range(max_box_size, min_box_size - 1, -2):
            count = 0
            for i in range(0, binary.shape[0], box_size):
                for j in range(0, binary.shape[1], box_size):
                    box = binary[i:i + box_size, j:j + box_size]
                    if np.any(box):
                        count += 1
            if count > 0:
                sizes.append(box_size)
                counts.append(count)

        if len(sizes) < 2:
            return 0.0

        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return float(-coeffs[0])

    # ----------------------------
    # Metrics bundle + scoring
    # ----------------------------
    def compute_all_metrics(self) -> Dict[str, float]:
        metrics = {}

        # Original 9
        metrics["branching_density"] = self.branching_density()
        metrics["tip_density"] = self.tip_density()
        metrics["growth_asymmetry"] = self.growth_asymmetry()
        metrics["perimeter_complexity"] = self.perimeter_complexity()
        metrics["growth_compactness"] = self.growth_compactness()
        metrics["radial_growth_variance"] = self.radial_growth_variance()
        metrics["skeleton_tortuosity"] = self.skeleton_tortuosity()
        metrics["multiscale_entropy"] = self.multiscale_entropy()
        metrics["interface_roughness"] = self.interface_roughness()
        metrics["shape_index"] = self.shape_index()

        # Literature-ish
        metrics["sdas_linear_um"] = self.secondary_dendrite_arm_spacing_linear()
        metrics["sdas_transform_um"] = self.secondary_dendrite_arm_spacing_transform()
        metrics["pdas_um"] = self.primary_dendrite_arm_spacing()
        metrics["tip_growth_activity"] = self.dendrite_tip_velocity_indicator()
        metrics["tip_radius_um"] = self.dendrite_tip_radius()

        sholl = self.sholl_analysis()
        metrics["sholl_max_intersections"] = float(sholl["max_intersections"])
        metrics["sholl_critical_radius_um"] = float(sholl["critical_radius"])
        metrics["sholl_ramification_index"] = float(sholl["ramification_index"])
        metrics["sholl_regression_coef"] = float(sholl["regression_coefficient"])
        metrics["sholl_total_intersections"] = float(sholl["total_intersections"])

        metrics["dendrite_coverage"] = self.dendrite_density()  # 0..1

        curv = self.principal_curvatures()
        metrics["mean_curvature"] = float(curv["mean_curvature"])
        metrics["curvature_variance"] = float(curv["curvature_variance"])
        metrics["max_curvature"] = float(curv["max_curvature"])

        topo = self.persistent_homology_features()
        metrics["num_components"] = float(topo["num_components"])
        metrics["num_holes"] = float(topo["num_holes"])
        metrics["euler_characteristic"] = float(topo["euler_characteristic"])
        metrics["betti_0"] = float(topo["betti_0"])
        metrics["betti_1"] = float(topo["betti_1"])

        metrics["fractal_dimension"] = self.fractal_dimension_boxcount()

        return metrics

    def calculate_severity_score(self, metrics: Dict[str, float]) -> Dict[str, float]:
        original_score = (
            metrics["branching_density"] * 0.15 +
            metrics["tip_density"] * 0.15 +
            metrics["growth_asymmetry"] * 10 +
            metrics["perimeter_complexity"] * 2 +
            (1 - metrics["growth_compactness"]) * 10 +
            metrics["radial_growth_variance"] * 0.01 +
            metrics["skeleton_tortuosity"] * 0.5 +
            metrics["multiscale_entropy"] * 10 +
            metrics["interface_roughness"] * 5
        )

        sdas_score = 100 / (metrics["sdas_linear_um"] + 1)

        sholl_score = (
            metrics["sholl_ramification_index"] * 0.3 +
            metrics["sholl_max_intersections"] * 0.1 +
            abs(metrics["sholl_regression_coef"]) * 100
        )

        tip_score = (
            metrics["tip_growth_activity"] * 50 +
            10 / (metrics["tip_radius_um"] + 0.1)
        )

        topo_score = (
            abs(metrics["euler_characteristic"]) * 2 +
            metrics["num_holes"] * 1
        )

        # total_score = (
        #     original_score * 0.3 +
        #     sdas_score * 0.2 +
        #     sholl_score * 0.25 +
        #     tip_score * 0.15 +
        #     topo_score * 0.1
        # )

        empirical_score = (
            min(metrics["branching_density"] * 0.5, 3) +
            # min(metrics["tip_density"] * 0.5, 3) +
            min(metrics["sholl_ramification_index"] * 0.5, 10) +
            metrics["sholl_max_intersections"] / 20 +
            metrics["sholl_total_intersections"] / 200 +
            metrics["interface_roughness"] * 2
        )

        return {
            "original_score": float(original_score),
            "sdas_score": float(sdas_score),
            "sholl_score": float(sholl_score),
            "tip_score": float(tip_score),
            "topo_score": float(topo_score),
            # "total_score": float(total_score),
            "empirical_score": empirical_score,
            # "morphology": metrics["morphology"],
        }

    def get_severity_level(self, total_score: float) -> str:
        if total_score < 7.5:
            return "None"
        if total_score < 12:
            return "Mild"
        elif total_score < 18:
            return "Moderate"
        elif total_score < 24:
            return "Severe"
        else:
            return "Extreme"

def _format_metrics(metrics: Dict[str, float], scores: Dict[str, float], severity: str) -> str:
    """
    Create a compact English metrics block for rendering in a figure.
    """
    lines = []
    lines.append("IMAGE")
    lines.append(f"- Size: {int(metrics.get('image_h', 0))} x {int(metrics.get('image_w', 0))} px")
    lines.append(f"- Pixel size: {metrics.get('pixel_size_um', 1.0):.3g} µm/px")
    lines.append(f"- Dendrite coverage: {metrics['dendrite_coverage']*100:.2f}%")
    lines.append("")
    lines.append("ORIGINAL METRICS")
    lines.append(f"- Branching density: {metrics['branching_density']:.4f}")
    lines.append(f"- Tip density: {metrics['tip_density']:.4f}")
    lines.append(f"- Growth asymmetry: {metrics['growth_asymmetry']:.4f}")
    lines.append(f"- Perimeter complexity: {metrics['perimeter_complexity']:.4f}")
    lines.append(f"- Growth compactness: {metrics['growth_compactness']:.4f}")
    lines.append(f"- Radial variance: {metrics['radial_growth_variance']:.4f}")
    lines.append(f"- Skeleton tortuosity: {metrics['skeleton_tortuosity']:.4f}")
    lines.append(f"- Multiscale entropy: {metrics['multiscale_entropy']:.4f}")
    lines.append(f"- Interface roughness: {metrics['interface_roughness']:.4f}")
    lines.append("")
    lines.append("LITERATURE-BASED (HEURISTICS)")
    lines.append(f"- SDAS (linear): {metrics['sdas_linear_um']:.2f} µm")
    lines.append(f"- SDAS (transform): {metrics['sdas_transform_um']:.2f} µm")
    lines.append(f"- PDAS: {metrics['pdas_um']:.2f} µm")
    lines.append(f"- Tip growth activity: {metrics['tip_growth_activity']:.4f}")
    lines.append(f"- Tip radius: {metrics['tip_radius_um']:.2f} µm")
    lines.append("")
    lines.append("SHOLL")
    lines.append(f"- Max intersections: {metrics['sholl_max_intersections']:.0f}")
    lines.append(f"- Critical radius: {metrics['sholl_critical_radius_um']:.2f} µm")
    lines.append(f"- Ramification index: {metrics['sholl_ramification_index']:.2f}")
    lines.append(f"- Regression coef: {metrics['sholl_regression_coef']:.4f}")
    lines.append(f"- Total intersections: {metrics['sholl_total_intersections']:.0f}")
    lines.append("")
    lines.append("MORPHOLOGY / TOPOLOGY / FRACTAL")
    lines.append(f"- Mean curvature: {metrics['mean_curvature']:.4f}")
    lines.append(f"- Curvature variance: {metrics['curvature_variance']:.4f}")
    lines.append(f"- Max curvature: {metrics['max_curvature']:.4f}")
    lines.append(f"- Components (β0): {metrics['num_components']:.0f}")
    lines.append(f"- Holes (β1): {metrics['num_holes']:.0f}")
    lines.append(f"- Euler characteristic: {metrics['euler_characteristic']:.0f}")
    lines.append(f"- Fractal dimension: {metrics['fractal_dimension']:.4f}")
    lines.append("")
    lines.append("SCORES")
    lines.append(f"- Original score: {scores['original_score']:.2f}")
    lines.append(f"- SDAS score: {scores['sdas_score']:.2f}")
    lines.append(f"- Sholl score: {scores['sholl_score']:.2f}")
    lines.append(f"- Tip score: {scores['tip_score']:.2f}")
    lines.append(f"- Topology score: {scores['topo_score']:.2f}")
    # lines.append(f"- TOTAL score: {scores['total_score']:.2f}")
    lines.append(f"- Empirical score: {scores['empirical_score']:.2f}")
    # lines.append(f"- Morphology: {scores['morphology']}")
    lines.append(f"- Severity: {severity}")

    return "\n".join(lines)

def generate_analysis_figure(
    image_or_path: Union[str, np.ndarray],
    pixel_size_um: float = 1.0,
    *,
    save: bool = False,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> Tuple[plt.Figure, Dict[str, float], Dict[str, float]]:
    """
    Generate ONE figure: visualization (left) + all metrics (right).

    Parameters
    ----------
    image_or_path : str | np.ndarray
        If str: expects a .npy file (same as your original code path).
        If ndarray: grayscale or RGB array.
    pixel_size_um : float
    save : bool
        Save the figure if True.
    save_path : str | None
        If None and save=True, auto-generate next to input file or current dir.
    show : bool
        Call plt.show() if True.
    title : str | None
        Optional suptitle.
    random_seed : int | None
        Only used if you later add randomness; kept for convenience.

    Returns
    -------
    fig, metrics, scores
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Load image
    base_name = "dendrite"
    if isinstance(image_or_path, str):
        base_name = os.path.splitext(os.path.basename(image_or_path))[0]
        img = np.load(image_or_path)[..., 0].astype(np.float32)
    else:
        img = image_or_path.astype(np.float32)
        if isinstance(img, np.ndarray) and img.ndim == 3:
            base_name = "dendrite_rgb"

    analyzer = ComprehensiveDendriteAnalyzer(img, pixel_size_um=pixel_size_um)
    metrics = analyzer.compute_all_metrics()
    scores = analyzer.calculate_severity_score(metrics)
    severity = analyzer.get_severity_level(scores["empirical_score"])

    # Add a few display-only fields
    metrics = dict(metrics)
    metrics["image_h"], metrics["image_w"] = analyzer.image_gray.shape[:2]
    metrics["pixel_size_um"] = float(pixel_size_um)

    # ---- Layout: left (3x4) + right text panel ----
    # Widen figure to make room for the text
    fig = plt.figure(figsize=(26, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.30, wspace=0.25, width_ratios=[1, 1, 1, 1, 1.45])

    # 1. Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(analyzer.image_gray, cmap="gray")
    ax1.set_title("Original Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. Binary
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(analyzer.binary, cmap="gray")
    ax2.set_title("Binary", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # 3. Skeleton
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(analyzer.skeleton, cmap="gray")
    ax3.set_title("Skeleton", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # 4. Distance transform
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(analyzer.dist_transform, cmap="hot")
    ax4.set_title("Distance Transform", fontsize=12, fontweight="bold")
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # Branch / Tip overlay prep
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    neighbor_count = ndimage.convolve(analyzer.skeleton.astype(int), kernel, mode="constant")
    branch_points = (neighbor_count >= 13) & analyzer.skeleton
    tips = (neighbor_count == 11) & analyzer.skeleton

    # 5. Branch points
    ax5 = fig.add_subplot(gs[1, 0])
    overlay_b = analyzer.image_rgb.copy()
    overlay_b[branch_points] = [255, 0, 0]
    ax5.imshow(overlay_b)
    ax5.set_title("Branch Points (Red)", fontsize=12, fontweight="bold")
    ax5.axis("off")

    # 6. Tip points
    ax6 = fig.add_subplot(gs[1, 1])
    overlay_t = analyzer.image_rgb.copy()
    overlay_t[tips] = [0, 255, 0]
    ax6.imshow(overlay_t)
    ax6.set_title("Tip Points (Green)", fontsize=12, fontweight="bold")
    ax6.axis("off")

    # 7. Sholl curve
    ax7 = fig.add_subplot(gs[1, 2:4])
    sholl = analyzer.sholl_analysis()
    ax7.plot(sholl["radii"], sholl["intersections"], marker="o", linewidth=2, markersize=4)
    ax7.axvline(x=sholl["critical_radius"], linestyle="--", linewidth=2,
                label=f"Critical radius: {sholl['critical_radius']:.2f} µm")
    ax7.set_xlabel("Distance from center (µm)", fontsize=11)
    ax7.set_ylabel("Intersections", fontsize=11)
    ax7.set_title("Sholl Profile", fontsize=12, fontweight="bold")
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # 8. Fractal (example box overlay)
    ax8 = fig.add_subplot(gs[2, 0])
    box_overlay = analyzer.image_rgb.copy()
    box_size = 20
    for i in range(0, analyzer.binary.shape[0], box_size):
        for j in range(0, analyzer.binary.shape[1], box_size):
            box = analyzer.binary[i:i + box_size, j:j + box_size]
            if np.any(box):
                cv2.rectangle(
                    box_overlay,
                    (j, i),
                    (min(j + box_size, box_overlay.shape[1]),
                     min(i + box_size, box_overlay.shape[0])),
                    (255, 255, 0),
                    1,
                )
    ax8.imshow(box_overlay)
    ax8.set_title(f"Fractal Box-Count (size={box_size})", fontsize=12, fontweight="bold")
    ax8.axis("off")

    # 9. Contours
    ax9 = fig.add_subplot(gs[2, 1])
    contours, _ = cv2.findContours(analyzer.binary.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    curvature_overlay = analyzer.image_rgb.copy()
    if len(contours) > 0:
        cv2.drawContours(curvature_overlay, contours, -1, (0, 255, 255), 2)
    ax9.imshow(curvature_overlay)
    ax9.set_title("Contour (Cyan)", fontsize=12, fontweight="bold")
    ax9.axis("off")

    # 10. Holes overlay
    ax10 = fig.add_subplot(gs[2, 2])
    filled = ndimage.binary_fill_holes(analyzer.binary)
    holes = filled & ~analyzer.binary
    overlay_h = analyzer.image_rgb.copy()
    overlay_h[holes] = [255, 0, 255]
    ax10.imshow(overlay_h)
    ax10.set_title("Holes (Magenta)", fontsize=12, fontweight="bold")
    ax10.axis("off")

    # 11. Radial circles
    ax11 = fig.add_subplot(gs[2, 3])
    h, w = analyzer.binary.shape
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - analyzer.centroid_y) ** 2 +
                        (x_coords - analyzer.centroid_x) ** 2)
    radial_overlay = analyzer.image_rgb.copy()
    for r in range(20, int(np.max(distances)), 40):
        cv2.circle(radial_overlay, (analyzer.centroid_x, analyzer.centroid_y), r, (255, 165, 0), 1)
    ax11.imshow(radial_overlay)
    ax11.set_title("Radial Rings", fontsize=12, fontweight="bold")
    ax11.axis("off")

    # Right-side metrics panel (spans all rows)
    ax_text = fig.add_subplot(gs[:, 4])
    ax_text.axis("off")
    metrics_text = _format_metrics(metrics, scores, severity)
    ax_text.text(
        0.0, 1.0, metrics_text,
        va="top", ha="left",
        fontsize=12,
        family="monospace",
        linespacing=1.25,
    )

    # Title
    if title is None:
        title = f"Comprehensive Dendrite Analysis — {base_name}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    # Save (optional)
    if save:
        if save_path is None:
            # default save path
            save_path = f"{base_name}_analysis.png" if not isinstance(image_or_path, str) else \
                os.path.join(os.path.dirname(image_or_path), f"{base_name}_analysis.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, metrics, scores


def random_visualize_from_glob(
    data_dir_or_pattern: Union[str, Sequence[str]],
    *,
    pixel_size_um: float = 1.0,
    k: int = 1,
    recursive: bool = True,
    seed: Optional[int] = None,
    save: bool = False,
    save_dir: str = "./output",
    show: bool = True,
) -> List[Tuple[str, Dict[str, float], Dict[str, float]]]:
    """
    Randomly select k items from a globbed list and visualize each with generate_analysis_figure().

    Parameters
    ----------
    data_dir_or_pattern : str | Sequence[str]
        - If str and is a directory: will search for **/*.npy under it.
        - If str and contains glob chars: uses it as a glob pattern.
        - If list/tuple: uses it directly as candidate paths.
    pixel_size_um : float
    k : int
        Number of random samples to visualize.
    recursive : bool
        Only applies if using glob.
    seed : int | None
    save : bool
    save_dir : str
    show : bool

    Returns
    -------
    list of (path, metrics, scores)
    """
    if seed is not None:
        random.seed(seed)

    # Build candidate list
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
    if len(candidates) == 0:
        raise FileNotFoundError(f"No files found from: {data_dir_or_pattern}")

    os.makedirs(save_dir, exist_ok=True)

    k = min(int(k), len(candidates))
    chosen = random.sample(candidates, k=k)

    results = []
    for p in chosen:
        base = os.path.splitext(os.path.basename(p))[0]
        save_path = os.path.join(save_dir, f"{base}_analysis.png") if save else None
        _, metrics, scores = generate_analysis_figure(
            p,
            pixel_size_um=pixel_size_um,
            save=save,
            save_path=save_path,
            show=show,
            title=None,
        )
        results.append((p, metrics, scores))
    return results

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

def plot_regression_summary(y_true: np.ndarray, y_pred: np.ndarray, prefix: str, save_dir: str=None, param_names=None):
    """
    Produces:
      1) MAE bar chart per parameter
      2) R2 bar chart per parameter
      3) Overall scatter (flattened)
      4) Residual histogram (flattened)
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    N, P = y_true.shape
    if param_names is None or len(param_names) != P:
        param_names = [f"p{i}" for i in range(P)]

    m = regression_metrics(y_true, y_pred)
    mae = np.array(m["per_dim"]["MAE"])
    r2 = np.array(m["per_dim"]["R2"])

    # 1) MAE bar
    plt.figure(figsize=(max(10, P * 0.5), 4))
    x = np.arange(P)
    plt.bar(x, mae)
    plt.xticks(x, param_names, rotation=60, ha="right")
    plt.ylabel("MAE")
    plt.title("Control parameter regression: MAE per parameter")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_mae_per_param.png"), dpi=300)
    else:
        plt.show()
    plt.close()

    # 2) R2 bar
    plt.figure(figsize=(max(10, P * 0.5), 4))
    plt.bar(x, r2)
    plt.xticks(x, param_names, rotation=60, ha="right")
    plt.ylabel("R²")
    plt.title("Control parameter regression: R² per parameter")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_r2_per_param.png"), dpi=300)
    else:
        plt.show()
    plt.close()

    # 3) Overall scatter (flatten)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    plt.figure(figsize=(5, 5))
    plt.scatter(yt, yp, s=6, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title("Overall true vs pred (all params flattened)")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_overall_scatter.png"), dpi=300)
    else:
        plt.show()
    plt.close()

    # 4) Residual histogram
    res = (y_pred - y_true).reshape(-1)
    plt.figure(figsize=(6, 4))
    plt.hist(res, bins=60)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title("Residual distribution (all params flattened)")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_residual_hist.png"), dpi=300)
    else:
        plt.show()
    plt.close()

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
    mean_c = conf_param.mean(axis=0)
    plt.figure(figsize=(max(10, P * 0.5), 4))
    x = np.arange(P)
    plt.bar(x, mean_c)
    plt.xticks(x, param_names, rotation=60, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean confidence")
    plt.title("Mean confidence per parameter")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_conf_param_mean.png"), dpi=300)
    else:
        plt.show()
    plt.close()

    # global confidence hist
    plt.figure(figsize=(6, 4))
    plt.hist(conf_global, bins=60)
    plt.xlabel("Global confidence")
    plt.ylabel("Count")
    plt.title("Global confidence distribution")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_conf_global_hist.png"), dpi=300)
    else:
        plt.show()
    plt.close()

    # flattened param confidence hist
    plt.figure(figsize=(6, 4))
    plt.hist(conf_param.reshape(-1), bins=60)
    plt.xlabel("Param confidence (flattened)")
    plt.ylabel("Count")
    plt.title("Param confidence distribution (all params flattened)")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{prefix}_conf_param_hist.png"), dpi=300)
    else:
        plt.show()
    plt.close()

# Example CLI usage:
if __name__ == "__main__":
    import sys

    # Usage:
    #   python step3_evaluate_metrics.py <data_dir> [pixel_size_um] [k] [save] [save_dir]
    #
    # Example:
    #   python step3_evaluate_metrics.py data/ 1.0 3 1 ./output
    if len(sys.argv) >= 2:
        data_dir = sys.argv[1]
        pixel = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0
        k = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
        save_flag = bool(int(sys.argv[4])) if len(sys.argv) >= 5 else False
        out_dir = sys.argv[5] if len(sys.argv) >= 6 else "./output"

        random_visualize_from_glob(
            data_dir,
            pixel_size_um=pixel,
            k=k,
            seed=None,
            save=save_flag,
            save_dir=out_dir,
            show=True,
        )
    else:
        print("Usage: python step3_evaluate_metrics.py <data_dir_or_glob> [pixel_size_um] [k] [save(0/1)] [save_dir]")
