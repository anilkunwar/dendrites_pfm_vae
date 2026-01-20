"""
枝晶生长综合分析系统
整合了原始指标和基于文献的标准指标

包含：
1. 原始9个指标（通用形态学分析）
2. 文献标准指标（材料科学/电化学/神经科学）
3. 综合评估和可视化
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ComprehensiveDendriteAnalyzer:
    """
    枝晶生长综合分析器
    整合所有指标和分析方法
    """

    def __init__(self, image, pixel_size_um=1.0):
        """
        初始化

        Parameters:
        -----------
        image : numpy.ndarray
            输入的枝晶生长图像 (可以是RGB或灰度图)
        pixel_size_um : float
            每像素对应的微米数，用于实际尺寸计算
        """
        if len(image.shape) == 3:
            self.image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            self.image_rgb = image.copy()
        else:
            self.image_gray = image.copy()
            self.image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        self.pixel_size = pixel_size_um

        # 二值化处理
        thresh = threshold_otsu(self.image_gray)
        self.binary = self.image_gray < thresh

        # 预计算骨架（多个指标会用到）
        self.skeleton = morphology.skeletonize(self.binary)

        # 预计算距离变换
        self.dist_transform = ndimage.distance_transform_edt(self.binary)

        # 预计算质心
        M = cv2.moments(self.binary.astype(np.uint8))
        if M['m00'] != 0:
            self.centroid_x = int(M['m10'] / M['m00'])
            self.centroid_y = int(M['m01'] / M['m00'])
        else:
            self.centroid_y, self.centroid_x = np.array(self.binary.shape) // 2

    # ========================================================================
    # 原始指标系统 (9个指标)
    # ========================================================================

    def branching_density(self):
        """
        分支密度指标
        计算枝晶的分支点数量与总面积的比值

        Returns:
        --------
        float : 分支密度值，值越大表示分支越密集
        """
        skeleton = self.skeleton

        # 检测分支点（8邻域中有3个或以上连接的点）
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])

        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        branch_points = np.sum((neighbor_count >= 13) & skeleton)

        total_area = np.sum(self.binary)

        if total_area == 0:
            return 0.0

        return branch_points / total_area * 10000

    def tip_density(self):
        """
        尖端密度指标
        计算枝晶生长尖端的数量，反映生长活跃程度

        Returns:
        --------
        float : 尖端密度值
        """
        skeleton = self.skeleton

        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])

        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        tip_points = np.sum((neighbor_count == 11) & skeleton)

        total_area = np.sum(self.binary)

        if total_area == 0:
            return 0.0

        return tip_points / total_area * 10000

    def growth_asymmetry(self):
        """
        生长不对称性指标
        计算枝晶生长的方向性偏差

        Returns:
        --------
        float : 不对称性值 (0-1)，越接近1表示越不对称
        """
        cx, cy = self.centroid_x, self.centroid_y

        # 计算四个象限的面积
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

    def perimeter_complexity(self):
        """
        周长复杂度指标
        通过周长与面积的比值评估边界复杂程度

        Returns:
        --------
        float : 复杂度值，值越大表示边界越复杂
        """
        contours, _ = cv2.findContours(self.binary.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return 0.0

        max_perimeter = max([cv2.arcLength(c, True) for c in contours])
        area = np.sum(self.binary)

        if area == 0:
            return 0.0

        complexity = max_perimeter / np.sqrt(area)

        return complexity

    def growth_compactness(self):
        """
        生长紧密度指标
        计算形状的紧凑程度，值越小表示越分散（生长越剧烈）

        Returns:
        --------
        float : 紧密度值 (0-1)
        """
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

    def radial_growth_variance(self):
        """
        径向生长方差指标
        计算从中心向外的生长距离方差

        Returns:
        --------
        float : 径向方差值
        """
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
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                distances.append(dist)

        if len(distances) == 0:
            return 0.0

        return np.var(distances)

    def skeleton_tortuosity(self):
        """
        骨架扭曲度指标
        计算枝晶骨架的弯曲程度

        Returns:
        --------
        float : 扭曲度值，越大表示枝晶越弯曲
        """
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

        tortuosity = skeleton_length / diagonal

        return tortuosity

    def multiscale_entropy(self, scales=[1, 2, 4, 8]):
        """
        多尺度熵指标
        在不同尺度下计算图像的信息熵，反映结构复杂度

        Parameters:
        -----------
        scales : list
            分析的尺度列表

        Returns:
        --------
        float : 平均多尺度熵值
        """
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
            entropies.append(entropy)

        return np.mean(entropies) if entropies else 0.0

    def interface_roughness(self):
        """
        界面粗糙度指标
        量化枝晶边界的不规则程度

        Returns:
        --------
        float : 粗糙度值
        """
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

            roughness_values.extend(angle_diff)

        return np.std(roughness_values) if roughness_values else 0.0

    # ========================================================================
    # 文献标准指标系统
    # ========================================================================

    def secondary_dendrite_arm_spacing_linear(self):
        """
        二次枝晶臂间距 - 线性截距法

        参考文献: Vandersluis & Ravindran (2017)

        Returns:
        --------
        float : SDAS值（微米）
        """
        skeleton = self.skeleton
        h, w = skeleton.shape

        spacings = []

        # 水平方向测量
        for y in range(0, h, 10):
            row = skeleton[y, :]
            transitions = np.diff(row.astype(int))
            arm_positions = np.where(np.abs(transitions) > 0)[0]
            if len(arm_positions) > 1:
                local_spacings = np.diff(arm_positions)
                spacings.extend(local_spacings)

        # 垂直方向测量
        for x in range(0, w, 10):
            col = skeleton[:, x]
            transitions = np.diff(col.astype(int))
            arm_positions = np.where(np.abs(transitions) > 0)[0]
            if len(arm_positions) > 1:
                local_spacings = np.diff(arm_positions)
                spacings.extend(local_spacings)

        if len(spacings) == 0:
            return 0.0

        sdas_pixels = np.median(spacings)
        return sdas_pixels * self.pixel_size

    def secondary_dendrite_arm_spacing_transform(self):
        """
        二次枝晶臂间距 - 间距变换法

        参考文献: Yang et al. (2025)

        Returns:
        --------
        float : SDAS值（微米）
        """
        dist_transform = self.dist_transform

        # 找到局部最大值（枝晶中心）
        local_max = morphology.local_maxima(dist_transform)

        coords = np.argwhere(local_max)
        if len(coords) < 2:
            return 0.0

        distances = pdist(coords)
        return np.median(distances) * self.pixel_size if len(distances) > 0 else 0.0

    def primary_dendrite_arm_spacing(self):
        """
        一次枝晶臂间距

        参考文献: Beltran-Sanchez & Stefanescu (2004)

        Returns:
        --------
        float : PDAS值（微米）
        """
        skeleton = self.skeleton

        labeled = measure.label(skeleton)
        regions = measure.regionprops(labeled)

        if len(regions) == 0:
            return 0.0

        centroids = [r.centroid for r in regions if r.area > 50]

        if len(centroids) < 2:
            return 0.0

        centroids = np.array(centroids)
        distances = pdist(centroids)

        return np.median(distances) * self.pixel_size if len(distances) > 0 else 0.0

    def dendrite_tip_velocity_indicator(self):
        """
        枝晶尖端生长速度指示器

        参考文献: LGK模型, Pei et al. (2020)

        Returns:
        --------
        float : 相对生长活性指标 (0-1)
        """
        skeleton = self.skeleton

        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])

        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        tips = (neighbor_count == 11) & skeleton

        tip_coords = np.argwhere(tips)

        if len(tip_coords) == 0:
            return 0.0

        sharpness_values = []
        for tip in tip_coords:
            y, x = tip
            y1, y2 = max(0, y - 5), min(skeleton.shape[0], y + 6)
            x1, x2 = max(0, x - 5), min(skeleton.shape[1], x + 6)
            local_region = skeleton[y1:y2, x1:x2]

            density = np.sum(local_region) / local_region.size
            sharpness_values.append(1 - density)

        return np.mean(sharpness_values) if sharpness_values else 0.0

    def dendrite_tip_radius(self):
        """
        枝晶尖端半径

        参考文献: Marginal Stability Criterion

        Returns:
        --------
        float : 平均尖端半径（微米）
        """
        dist_transform = self.dist_transform
        skeleton = self.skeleton

        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        tips = (neighbor_count == 11) & skeleton

        tip_radii = dist_transform[tips]

        if len(tip_radii) == 0:
            return 0.0

        return np.mean(tip_radii) * self.pixel_size

    def sholl_analysis(self, center=None, step_size=10):
        """
        Sholl分析

        参考文献: Sholl (1953), Langhammer et al. (2010)

        Parameters:
        -----------
        center : tuple, optional
            分析中心点，默认使用质心
        step_size : int
            同心圆间隔（像素）

        Returns:
        --------
        dict : 包含Sholl曲线数据和汇总统计
        """
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
            intersection_count = np.sum(skeleton & ring_mask)
            intersections.append(intersection_count)

        intersections = np.array(intersections)

        if len(intersections) > 0:
            max_intersections = np.max(intersections)
            critical_radius = radii[np.argmax(intersections)] if max_intersections > 0 else 0

            primary_dendrites = self._count_primary_dendrites(cy, cx)
            ramification_index = max_intersections / (primary_dendrites + 1)

            if len(intersections) > 1:
                valid_idx = intersections > 0
                if np.sum(valid_idx) > 1:
                    log_density = np.log10(intersections[valid_idx] / (np.pi * radii[valid_idx] ** 2) + 1e-6)
                    regression_coef = np.polyfit(radii[valid_idx], log_density, 1)[0]
                else:
                    regression_coef = 0.0
            else:
                regression_coef = 0.0
        else:
            max_intersections = 0
            critical_radius = 0
            ramification_index = 0
            regression_coef = 0.0

        return {
            'radii': radii * self.pixel_size,
            'intersections': intersections,
            'max_intersections': max_intersections,
            'critical_radius': critical_radius * self.pixel_size,
            'ramification_index': ramification_index,
            'regression_coefficient': regression_coef,
            'total_intersections': np.sum(intersections)
        }

    def _count_primary_dendrites(self, cy, cx, radius=20):
        """辅助函数：计算从中心发出的主要枝晶数"""
        skeleton = self.skeleton
        h, w = skeleton.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

        ring_mask = (distances >= radius - 2) & (distances < radius + 2)
        ring_skeleton = skeleton & ring_mask
        labeled = measure.label(ring_skeleton)

        return len(np.unique(labeled)) - 1

    def dendrite_density(self):
        """
        枝晶密度

        参考文献: Pei et al. (2020)

        Returns:
        --------
        float : 枝晶占据的面积比例
        """
        total_pixels = self.binary.size
        dendrite_pixels = np.sum(self.binary)
        return dendrite_pixels / total_pixels

    def principal_curvatures(self, num_samples=100):
        """
        主曲率分析

        参考文献: Scientific Reports (2015)

        Returns:
        --------
        dict : 曲率统计信息
        """
        contours, _ = cv2.findContours(
            self.binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            return {'mean_curvature': 0.0, 'curvature_variance': 0.0, 'max_curvature': 0.0}

        main_contour = max(contours, key=cv2.contourArea)

        if len(main_contour) < 3:
            return {'mean_curvature': 0.0, 'curvature_variance': 0.0, 'max_curvature': 0.0}

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
                curvatures.append(curvature)

        if len(curvatures) == 0:
            return {'mean_curvature': 0.0, 'curvature_variance': 0.0, 'max_curvature': 0.0}

        return {
            'mean_curvature': np.mean(curvatures),
            'curvature_variance': np.var(curvatures),
            'max_curvature': np.max(curvatures)
        }

    def persistent_homology_features(self):
        """
        持久同调特征（简化版）

        参考文献: Taylor & Francis (2025)

        Returns:
        --------
        dict : 拓扑特征
        """
        labeled = measure.label(self.binary)
        num_components = len(np.unique(labeled)) - 1

        filled = ndimage.binary_fill_holes(self.binary)
        holes = filled & ~self.binary
        labeled_holes = measure.label(holes)
        num_holes = len(np.unique(labeled_holes)) - 1

        euler_characteristic = num_components - num_holes

        return {
            'num_components': num_components,
            'num_holes': num_holes,
            'euler_characteristic': euler_characteristic,
            'betti_0': num_components,
            'betti_1': num_holes
        }

    def fractal_dimension_boxcount(self, max_box_size=None, min_box_size=2):
        """
        分形维度 - 盒计数法

        参考文献: Mandelbrot (1982)

        Parameters:
        -----------
        max_box_size : int, optional
            最大盒子尺寸
        min_box_size : int
            最小盒子尺寸

        Returns:
        --------
        float : 分形维度值
        """
        binary = self.binary

        if max_box_size is None:
            max_box_size = min(binary.shape) // 4

        sizes = []
        counts = []

        # 从大到小的盒子尺寸
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

        # 对数-对数拟合
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        fractal_dim = -coeffs[0]

        return fractal_dim

    # ========================================================================
    # 综合分析和评估
    # ========================================================================

    def compute_all_metrics(self):
        """
        计算所有指标（原始 + 文献标准）

        Returns:
        --------
        dict : 包含所有指标的字典
        """
        print("=" * 70)
        print("计算综合枝晶分析指标...")
        print("=" * 70)

        metrics = {}

        # 原始指标系统
        print("\n【原始指标系统】")
        print("  计算分支密度...")
        metrics['branching_density'] = self.branching_density()

        print("  计算尖端密度...")
        metrics['tip_density'] = self.tip_density()

        print("  计算生长不对称性...")
        metrics['growth_asymmetry'] = self.growth_asymmetry()

        print("  计算周长复杂度...")
        metrics['perimeter_complexity'] = self.perimeter_complexity()

        print("  计算生长紧密度...")
        metrics['growth_compactness'] = self.growth_compactness()

        print("  计算径向生长方差...")
        metrics['radial_growth_variance'] = self.radial_growth_variance()

        print("  计算骨架扭曲度...")
        metrics['skeleton_tortuosity'] = self.skeleton_tortuosity()

        print("  计算多尺度熵...")
        metrics['multiscale_entropy'] = self.multiscale_entropy()

        print("  计算界面粗糙度...")
        metrics['interface_roughness'] = self.interface_roughness()

        # 文献标准指标
        print("\n【文献标准指标】")
        print("  计算SDAS（线性法）...")
        metrics['sdas_linear'] = self.secondary_dendrite_arm_spacing_linear()

        print("  计算SDAS（变换法）...")
        metrics['sdas_transform'] = self.secondary_dendrite_arm_spacing_transform()

        print("  计算PDAS...")
        metrics['pdas'] = self.primary_dendrite_arm_spacing()

        print("  计算尖端特征...")
        metrics['tip_growth_indicator'] = self.dendrite_tip_velocity_indicator()
        metrics['tip_radius'] = self.dendrite_tip_radius()

        print("  执行Sholl分析...")
        sholl_results = self.sholl_analysis()
        metrics['sholl_max_intersections'] = sholl_results['max_intersections']
        metrics['sholl_critical_radius'] = sholl_results['critical_radius']
        metrics['sholl_ramification_index'] = sholl_results['ramification_index']
        metrics['sholl_regression_coef'] = sholl_results['regression_coefficient']
        metrics['sholl_total_intersections'] = sholl_results['total_intersections']

        print("  计算形态学特征...")
        metrics['dendrite_density'] = self.dendrite_density()

        curvature_results = self.principal_curvatures()
        metrics['mean_curvature'] = curvature_results['mean_curvature']
        metrics['curvature_variance'] = curvature_results['curvature_variance']
        metrics['max_curvature'] = curvature_results['max_curvature']

        print("  计算拓扑学特征...")
        topo_results = self.persistent_homology_features()
        metrics['num_components'] = topo_results['num_components']
        metrics['num_holes'] = topo_results['num_holes']
        metrics['euler_characteristic'] = topo_results['euler_characteristic']
        metrics['betti_0'] = topo_results['betti_0']
        metrics['betti_1'] = topo_results['betti_1']

        print("  计算分形维度...")
        metrics['fractal_dimension'] = self.fractal_dimension_boxcount()

        print("\n完成！")
        print("=" * 70)

        return metrics

    def calculate_severity_score(self, metrics):
        """
        计算综合剧烈程度评分

        Parameters:
        -----------
        metrics : dict
            所有指标的字典

        Returns:
        --------
        dict : 包含各类评分和总评分
        """
        # 原始指标评分
        original_score = (
                metrics['branching_density'] * 0.15 +
                metrics['tip_density'] * 0.15 +
                metrics['growth_asymmetry'] * 10 +
                metrics['perimeter_complexity'] * 2 +
                (1 - metrics['growth_compactness']) * 10 +
                metrics['radial_growth_variance'] * 0.01 +
                metrics['skeleton_tortuosity'] * 0.5 +
                metrics['multiscale_entropy'] * 10 +
                metrics['interface_roughness'] * 5
        )

        # 文献标准评分
        # SDAS越小越剧烈（取倒数）
        sdas_score = 100 / (metrics['sdas_linear'] + 1)

        # Sholl分析评分
        sholl_score = (
                metrics['sholl_ramification_index'] * 0.3 +
                metrics['sholl_max_intersections'] * 0.1 +
                abs(metrics['sholl_regression_coef']) * 100
        )

        # 尖端活性评分
        tip_score = (
                metrics['tip_growth_indicator'] * 50 +
                10 / (metrics['tip_radius'] + 0.1)
        )

        # 拓扑复杂度评分
        topo_score = (
                abs(metrics['euler_characteristic']) * 2 +
                metrics['num_holes'] * 1
        )

        # 综合评分（加权平均）
        total_score = (
                original_score * 0.3 +
                sdas_score * 0.2 +
                sholl_score * 0.25 +
                tip_score * 0.15 +
                topo_score * 0.1
        )

        return {
            'original_score': original_score,
            'sdas_score': sdas_score,
            'sholl_score': sholl_score,
            'tip_score': tip_score,
            'topo_score': topo_score,
            'total_score': total_score
        }

    def get_severity_level(self, total_score):
        """
        根据综合评分判断剧烈程度等级

        Parameters:
        -----------
        total_score : float
            综合评分

        Returns:
        --------
        str : 剧烈程度等级
        """
        if total_score < 20:
            return "轻度生长 (Mild)"
        elif total_score < 40:
            return "中度生长 (Moderate)"
        elif total_score < 60:
            return "剧烈生长 (Severe)"
        else:
            return "极度剧烈生长 (Extreme)"

    # ========================================================================
    # 可视化功能
    # ========================================================================

    def create_comprehensive_visualization(self, save_path=None):
        """
        创建综合可视化分析图

        Parameters:
        -----------
        save_path : str, optional
            保存路径
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. 原始图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.image_gray, cmap='gray')
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. 二值化图像
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.binary, cmap='gray')
        ax2.set_title('Binary Image', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 3. 骨架
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(self.skeleton, cmap='gray')
        ax3.set_title('Skeleton', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # 4. 距离变换
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(self.dist_transform, cmap='hot')
        ax4.set_title('Distance Transform', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)

        # 5. 分支点检测
        ax5 = fig.add_subplot(gs[1, 0])
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(self.skeleton.astype(int), kernel, mode='constant')
        branch_points = (neighbor_count >= 13) & self.skeleton
        overlay = self.image_rgb.copy()
        overlay[branch_points] = [255, 0, 0]
        ax5.imshow(overlay)
        ax5.set_title('Branch Points (Red)', fontsize=12, fontweight='bold')
        ax5.axis('off')

        # 6. 尖端检测
        ax6 = fig.add_subplot(gs[1, 1])
        tips = (neighbor_count == 11) & self.skeleton
        overlay = self.image_rgb.copy()
        overlay[tips] = [0, 255, 0]
        ax6.imshow(overlay)
        ax6.set_title('Tip Points (Green)', fontsize=12, fontweight='bold')
        ax6.axis('off')

        # 7. Sholl分析曲线
        ax7 = fig.add_subplot(gs[1, 2:])
        sholl_results = self.sholl_analysis()
        ax7.plot(sholl_results['radii'], sholl_results['intersections'],
                 marker='o', linewidth=2, markersize=4, color='blue')
        ax7.axvline(x=sholl_results['critical_radius'], color='r',
                    linestyle='--', linewidth=2,
                    label=f'Critical Radius: {sholl_results["critical_radius"]:.2f} μm')
        ax7.set_xlabel('Distance from Center (μm)', fontsize=11)
        ax7.set_ylabel('Number of Intersections', fontsize=11)
        ax7.set_title('Sholl Analysis Profile', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.legend()

        # 8. 分形维度可视化（盒计数）
        ax8 = fig.add_subplot(gs[2, 0])
        # 显示不同尺度的盒子覆盖
        box_overlay = self.image_rgb.copy()
        box_size = 20
        for i in range(0, self.binary.shape[0], box_size):
            for j in range(0, self.binary.shape[1], box_size):
                box = self.binary[i:i + box_size, j:j + box_size]
                if np.any(box):
                    cv2.rectangle(box_overlay, (j, i),
                                  (min(j + box_size, box_overlay.shape[1]),
                                   min(i + box_size, box_overlay.shape[0])),
                                  (255, 255, 0), 1)
        ax8.imshow(box_overlay)
        ax8.set_title(f'Fractal Box-Count (size={box_size})', fontsize=12, fontweight='bold')
        ax8.axis('off')

        # 9. 曲率分析
        ax9 = fig.add_subplot(gs[2, 1])
        contours, _ = cv2.findContours(self.binary.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        curvature_overlay = self.image_rgb.copy()
        if len(contours) > 0:
            cv2.drawContours(curvature_overlay, contours, -1, (0, 255, 255), 2)
        ax9.imshow(curvature_overlay)
        ax9.set_title('Contour & Curvature', fontsize=12, fontweight='bold')
        ax9.axis('off')

        # 10. 拓扑特征（孔洞）
        ax10 = fig.add_subplot(gs[2, 2])
        filled = ndimage.binary_fill_holes(self.binary)
        holes = filled & ~self.binary
        overlay = self.image_rgb.copy()
        overlay[holes] = [255, 0, 255]
        ax10.imshow(overlay)
        ax10.set_title('Holes (Magenta)', fontsize=12, fontweight='bold')
        ax10.axis('off')

        # 11. 径向分析
        ax11 = fig.add_subplot(gs[2, 3])
        h, w = self.binary.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - self.centroid_y) ** 2 +
                            (x_coords - self.centroid_x) ** 2)
        radial_overlay = self.image_rgb.copy()
        # 绘制同心圆
        for r in range(20, int(np.max(distances)), 40):
            cv2.circle(radial_overlay, (self.centroid_x, self.centroid_y),
                       r, (255, 165, 0), 1)
        ax11.imshow(radial_overlay)
        ax11.set_title('Radial Analysis', fontsize=12, fontweight='bold')
        ax11.axis('off')

        plt.suptitle('Comprehensive Dendrite Analysis', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n可视化结果已保存至: {save_path}")

        plt.show()

    def generate_analysis_report(self, metrics, scores, save_path=None):
        """
        生成详细的分析报告

        Parameters:
        -----------
        metrics : dict
            所有指标
        scores : dict
            评分结果
        save_path : str, optional
            保存路径
        """
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("枝晶生长综合分析报告")
        report_lines.append("COMPREHENSIVE DENDRITE GROWTH ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # 图像信息
        report_lines.append("【图像信息 / Image Information】")
        report_lines.append("-" * 80)
        report_lines.append(f"图像尺寸 / Image Size: {self.image_gray.shape}")
        report_lines.append(f"像素尺寸 / Pixel Size: {self.pixel_size} μm/pixel")
        report_lines.append(f"枝晶覆盖率 / Dendrite Coverage: {metrics['dendrite_density'] * 100:.2f}%")
        report_lines.append("")

        # 原始指标系统
        report_lines.append("【原始指标系统 / Original Metrics System】")
        report_lines.append("-" * 80)
        report_lines.append(f"1. 分支密度 / Branching Density:        {metrics['branching_density']:.4f}")
        report_lines.append(f"2. 尖端密度 / Tip Density:              {metrics['tip_density']:.4f}")
        report_lines.append(f"3. 生长不对称性 / Growth Asymmetry:      {metrics['growth_asymmetry']:.4f}")
        report_lines.append(f"4. 周长复杂度 / Perimeter Complexity:   {metrics['perimeter_complexity']:.4f}")
        report_lines.append(f"5. 生长紧密度 / Growth Compactness:     {metrics['growth_compactness']:.4f}")
        report_lines.append(f"6. 径向生长方差 / Radial Variance:      {metrics['radial_growth_variance']:.4f}")
        report_lines.append(f"7. 骨架扭曲度 / Skeleton Tortuosity:    {metrics['skeleton_tortuosity']:.4f}")
        report_lines.append(f"8. 多尺度熵 / Multiscale Entropy:       {metrics['multiscale_entropy']:.4f}")
        report_lines.append(f"9. 界面粗糙度 / Interface Roughness:    {metrics['interface_roughness']:.4f}")
        report_lines.append("")

        # 文献标准指标
        report_lines.append("【文献标准指标 / Literature-Based Metrics】")
        report_lines.append("-" * 80)
        report_lines.append("")

        report_lines.append("  材料科学指标 / Materials Science Metrics:")
        report_lines.append(f"    • SDAS (线性法 / Linear):              {metrics['sdas_linear']:.2f} μm")
        report_lines.append(f"    • SDAS (变换法 / Transform):           {metrics['sdas_transform']:.2f} μm")
        report_lines.append(f"    • PDAS:                                {metrics['pdas']:.2f} μm")
        report_lines.append(f"    • 尖端生长活性 / Tip Growth Activity:   {metrics['tip_growth_indicator']:.4f}")
        report_lines.append(f"    • 尖端半径 / Tip Radius:                {metrics['tip_radius']:.2f} μm")
        report_lines.append("")

        report_lines.append("  Sholl分析指标 / Sholl Analysis Metrics:")
        report_lines.append(f"    • 最大交叉数 / Max Intersections:       {metrics['sholl_max_intersections']:.0f}")
        report_lines.append(f"    • 临界半径 / Critical Radius:           {metrics['sholl_critical_radius']:.2f} μm")
        report_lines.append(f"    • 分支指数 / Ramification Index:        {metrics['sholl_ramification_index']:.2f}")
        report_lines.append(f"    • 回归系数 / Regression Coefficient:    {metrics['sholl_regression_coef']:.4f}")
        report_lines.append(f"    • 总交叉数 / Total Intersections:       {metrics['sholl_total_intersections']:.0f}")
        report_lines.append("")

        report_lines.append("  形态学指标 / Morphological Metrics:")
        report_lines.append(f"    • 平均曲率 / Mean Curvature:            {metrics['mean_curvature']:.4f}")
        report_lines.append(f"    • 曲率方差 / Curvature Variance:        {metrics['curvature_variance']:.4f}")
        report_lines.append(f"    • 最大曲率 / Max Curvature:             {metrics['max_curvature']:.4f}")
        report_lines.append("")

        report_lines.append("  拓扑学指标 / Topological Metrics:")
        report_lines.append(f"    • 连通组件数 / Components (β₀):         {metrics['num_components']:.0f}")
        report_lines.append(f"    • 孔洞数 / Holes (β₁):                   {metrics['num_holes']:.0f}")
        report_lines.append(f"    • 欧拉示性数 / Euler Characteristic:    {metrics['euler_characteristic']:.0f}")
        report_lines.append("")

        report_lines.append("  分形分析 / Fractal Analysis:")
        report_lines.append(f"    • 分形维度 / Fractal Dimension:         {metrics['fractal_dimension']:.4f}")
        report_lines.append("")

        # 综合评分
        report_lines.append("【综合评分 / Comprehensive Scores】")
        report_lines.append("-" * 80)
        report_lines.append(f"原始指标评分 / Original Score:          {scores['original_score']:.2f}")
        report_lines.append(f"SDAS评分 / SDAS Score:                  {scores['sdas_score']:.2f}")
        report_lines.append(f"Sholl评分 / Sholl Score:                {scores['sholl_score']:.2f}")
        report_lines.append(f"尖端活性评分 / Tip Activity Score:      {scores['tip_score']:.2f}")
        report_lines.append(f"拓扑复杂度评分 / Topology Score:        {scores['topo_score']:.2f}")
        report_lines.append("")
        report_lines.append(f"{'*' * 30}")
        report_lines.append(f"总评分 / TOTAL SCORE:                   {scores['total_score']:.2f}")
        report_lines.append(f"{'*' * 30}")
        report_lines.append("")

        # 剧烈程度等级
        severity_level = self.get_severity_level(scores['total_score'])
        report_lines.append("【剧烈程度评级 / Severity Level】")
        report_lines.append("-" * 80)
        report_lines.append(f">>> {severity_level} <<<")
        report_lines.append("")

        # 解释和建议
        report_lines.append("【关键指标解读 / Key Metrics Interpretation】")
        report_lines.append("-" * 80)

        if metrics['sdas_linear'] < 10:
            report_lines.append("✓ SDAS < 10 μm: 表明快速生长，枝晶臂间距非常小")
        elif metrics['sdas_linear'] < 50:
            report_lines.append("✓ SDAS 10-50 μm: 中等生长速率")
        else:
            report_lines.append("✓ SDAS > 50 μm: 较慢的生长速率")

        if metrics['sholl_ramification_index'] > 50:
            report_lines.append("✓ 分支指数 > 50: 极高的分支复杂度")
        elif metrics['sholl_ramification_index'] > 20:
            report_lines.append("✓ 分支指数 20-50: 高度分支结构")
        else:
            report_lines.append("✓ 分支指数 < 20: 相对简单的分支结构")

        if metrics['tip_growth_indicator'] > 0.7:
            report_lines.append("✓ 尖端活性 > 0.7: 生长高度活跃")
        elif metrics['tip_growth_indicator'] > 0.4:
            report_lines.append("✓ 尖端活性 0.4-0.7: 中等活性")
        else:
            report_lines.append("✓ 尖端活性 < 0.4: 生长较缓")

        if abs(metrics['euler_characteristic']) > 10:
            report_lines.append("✓ |欧拉示性数| > 10: 复杂的拓扑结构，多孔洞")

        report_lines.append("")

        # 文献引用建议
        report_lines.append("【推荐引用文献 / Recommended Citations】")
        report_lines.append("-" * 80)
        report_lines.append("1. SDAS测量方法:")
        report_lines.append("   Vandersluis & Ravindran (2017), Metall. Microstruct. Anal.")
        report_lines.append("")
        report_lines.append("2. Sholl分析:")
        report_lines.append("   Sholl (1953), J. Anatomy; Langhammer et al. (2010), Cytometry")
        report_lines.append("")
        report_lines.append("3. 枝晶生长理论:")
        report_lines.append("   Lipton-Glicksman-Kurz Model; Beltran-Sanchez & Stefanescu (2004)")
        report_lines.append("")
        report_lines.append("4. 拓扑分析:")
        report_lines.append("   Taylor & Francis (2025), Persistent Homology in Materials")
        report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("报告结束 / End of Report")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # 打印到控制台
        print(report_text)

        # 保存到文件
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n报告已保存至: {save_path}")

        return report_text


def analyze_dendrite_comprehensive(image_path, pixel_size_um=1.0,
                                   generate_visualization=True,
                                   generate_report=True,
                                   save_dir='./output'):
    """
    综合枝晶分析便捷函数

    Parameters:
    -----------
    image_path : str or numpy.ndarray
        图像路径或图像数组
    pixel_size_um : float
        像素尺寸（微米/像素）
    generate_visualization : bool
        是否生成可视化
    generate_report : bool
        是否生成报告
    save_dir : str
        保存目录

    Returns:
    --------
    tuple : (analyzer, metrics, scores)
    """
    import os

    # 读取图像
    if isinstance(image_path, str):
        image = np.load(image_path)[..., 0]
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = image.astype(np.float32)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
    else:
        image = image_path
        base_name = 'dendrite'

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建分析器
    print(f"\n初始化枝晶分析器...")
    print(f"图像尺寸: {image.shape}")
    print(f"像素尺寸: {pixel_size_um} μm/pixel")
    print()

    analyzer = ComprehensiveDendriteAnalyzer(image, pixel_size_um)

    # 计算所有指标
    metrics = analyzer.compute_all_metrics()

    # 计算评分
    print("\n计算综合评分...")
    scores = analyzer.calculate_severity_score(metrics)

    # 生成可视化
    if generate_visualization:
        viz_path = os.path.join(save_dir, f'{base_name}_comprehensive_analysis.png')
        print(f"\n生成可视化分析图...")
        analyzer.create_comprehensive_visualization(viz_path)

    # 生成报告
    if generate_report:
        report_path = os.path.join(save_dir, f'{base_name}_analysis_report.txt')
        print(f"\n生成分析报告...")
        analyzer.generate_analysis_report(metrics, scores, report_path)

    return analyzer, metrics, scores


# 使用示例
if __name__ == "__main__":
    import sys, os

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        pixel_size = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        save_dir = sys.argv[3] if len(sys.argv) > 3 else './output'

        for imagef in os.listdir(image_path):
            analyzer, metrics, scores = analyze_dendrite_comprehensive(
                os.path.join(image_path, imagef),
                pixel_size_um=pixel_size,
                generate_visualization=True,
                generate_report=True,
                save_dir=save_dir
            )

        print("\n" + "=" * 80)
        print("分析完成！所有结果已保存到:", save_dir)
        print("=" * 80)

    else:
        print("用法: python comprehensive_dendrite_analyzer.py <image_path> [pixel_size_um] [save_dir]")
        print("\n示例:")
        print("  python comprehensive_dendrite_analyzer.py dendrite.png 0.5 ./results")
        print("\n或在代码中使用:")
        print("  from comprehensive_dendrite_analyzer import analyze_dendrite_comprehensive")
        print("  analyzer, metrics, scores = analyze_dendrite_comprehensive('image.png')")