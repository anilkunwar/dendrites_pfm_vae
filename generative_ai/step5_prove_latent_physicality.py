# generative_ai/step6_compare_latent_physicality_all_splits.py
# -*- coding: utf-8 -*-
"""
Compare physical consistency of three representation spaces on ALL data:

1. MDN-VAE latent space
2. Linear-Regressor VAE latent space
3. PCA image-feature space + Ridge linear probe

This script evaluates train / val / test together, and also reports per-split
results. It does NOT use command-line arguments. Edit CONFIG below directly.

核心目的
--------
不要再只用 test set 的 4 条 did 来说明 latent 的物理性。
这里改为在 train + val + test 全部数据上做测试，并且比较：

    MDN latent
    Linear Regressor latent
    PCA baseline

评价维度
--------
A. Parameter prediction quality
   - pred_y vs true_y 的 MAE / MSE / R2 / Spearman

B. Same-trajectory temporal ordering
   - 同一 did 内 true_t 和 pred_t 是否一致
   - 同一 did 内 |delta_t| 和 representation distance 是否正相关
   - 相邻时间步距离是否小于非相邻时间步距离

C. Local physical consistency by kNN
   - representation 近邻是否比 random baseline 更接近：
       true_t
       pred_t
       true physical parameters
       pred physical parameters
       did
       morphology severity, if enabled

D. Optional morphology metrics
   - 如果 COMPUTE_MORPHOLOGY=True，会用 ComprehensiveDendriteAnalyzer 计算形貌严重度。
   - 这一步较慢；第一次可以设为 False，只比较物理参数和时间。
"""

import os
import json
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from src.dataloader import DendritePFMDataset, PARAM_RANGES
from src.modelv11 import mdn_point_and_confidence
from src.evaluate_metrics import ComprehensiveDendriteAnalyzer, get_severity_level


# ============================================================
# 0. CONFIG: 修改这里即可，不需要命令行参数
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # 数据设置
    # --------------------------------------------------------
    "split_json": "data/dataset_split.json",
    "splits": ["train", "val", "test"],
    "image_size": (3, 48, 48),
    "batch_size": 64,
    "num_workers": 4,

    # --------------------------------------------------------
    # 模型路径
    #
    # MDN 模型路径：
    #   MDN_MODEL_ROOT/ckpt/best.pt
    #
    # Linear 模型路径：
    #   LINEAR_MODEL_ROOT/ckpt/best.pt
    #
    # 你需要根据自己的实际训练结果修改。
    # --------------------------------------------------------
    "mdn_model_root": "results/good_v2",
    "linear_model_root": "results/VAEv12_LinearReg_src=mu_time=20260504_230843",

    "ckpt_name": "best.pt",

    # --------------------------------------------------------
    # latent 选择
    #
    # 推荐用 "mu"：
    #   mu_q 是 encoder 的确定性输出；
    #   z 带有 reparameterization noise，不适合作为几何分析的默认选择。
    # --------------------------------------------------------
    "latent_source": "mu",

    # --------------------------------------------------------
    # PCA baseline 设置
    #
    # PCA 是在 train split 的 clean target image xo 上 fit；
    # 然后 transform train/val/test 全部样本。
    #
    # Ridge probe 也是只在 train split 上训练，然后预测 train/val/test。
    # 这样可以测试 PCA 表征对物理量的泛化能力。
    # --------------------------------------------------------
    "pca_dim": 16,
    "pca_whiten": False,
    "ridge_alpha": 1.0,

    # --------------------------------------------------------
    # kNN 测试设置
    # --------------------------------------------------------
    "knn_k": 10,
    "n_random_trials": 20,

    # --------------------------------------------------------
    # MDN 置信度尺度。
    # 这里只影响 mdn_point_and_confidence 中 conf 的计算。
    # 对 theta_hat 混合均值本身没有影响。
    # --------------------------------------------------------
    "var_scale": 0.01,

    # --------------------------------------------------------
    # 是否计算形貌指标
    #
    # True:
    #   会计算 empirical_score / severity 等，结果更完整但慢。
    #
    # False:
    #   只比较物理参数、时间、did，不计算图像形貌严重度。
    # --------------------------------------------------------
    "compute_morphology": True,
    "morph_channel": 0,

    # --------------------------------------------------------
    # 输出位置
    # --------------------------------------------------------
    "out_dir": "results/latent_physicality_compare_all_splits",

    # --------------------------------------------------------
    # 随机种子
    # --------------------------------------------------------
    "seed": 0,
}


Y_NAMES = ["t"] + list(PARAM_RANGES.keys())
PARAM_NAMES = list(PARAM_RANGES.keys())
SEVERITY_MAP = ["None", "Mild", "Moderate", "Severe", "Extreme"]


# ============================================================
# 1. 通用工具函数
# ============================================================

def set_seed(seed: int):
    """固定随机种子，保证 kNN random baseline 和 PCA 结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> str:
    """创建目录并返回路径。"""
    os.makedirs(path, exist_ok=True)
    return path


def safe_spearman(x, y):
    """
    安全计算 Spearman correlation。
    如果样本太少或方差为 0，返回 NaN。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan, np.nan

    x = x[mask]
    y = y[mask]

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan

    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def safe_mannwhitneyu(x, y):
    """
    安全计算 Mann-Whitney U test。
    用于比较 kNN 与 random baseline 是否显著不同。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 4 or len(y) < 4:
        return np.nan, np.nan

    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    return float(stat), float(p)


def safe_r2(y_true, y_pred):
    """
    安全计算 R2。
    如果 y_true 方差过小，返回 NaN。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 4:
        return np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if np.std(y_true) < 1e-12:
        return np.nan

    return float(r2_score(y_true, y_pred))


def compute_morphology_summary(img2d: np.ndarray) -> dict:
    """
    用项目已有的 ComprehensiveDendriteAnalyzer 计算单通道形貌指标。

    注意：
    - img2d 通常是 xo 的第 0 通道。
    - 如果你知道锂浓度或枝晶主变量在别的 channel，修改 CONFIG["morph_channel"]。
    """
    img2d = np.asarray(img2d, dtype=np.float32)
    img2d = np.clip(img2d, 0.0, 1.0)

    try:
        analyzer = ComprehensiveDendriteAnalyzer(img2d)
        metrics = analyzer.compute_all_metrics()
        scores = analyzer.calculate_severity_score(metrics)
        severity = get_severity_level(scores["empirical_score"])

        if severity in SEVERITY_MAP:
            severity_id = SEVERITY_MAP.index(severity)
        else:
            severity_id = -1

        return {
            "empirical_score": float(scores["empirical_score"]),
            "severity": severity,
            "severity_id": int(severity_id),
            "dendrite_coverage": float(metrics.get("dendrite_coverage", np.nan)),
            "branching_density": float(metrics.get("branching_density", np.nan)),
            "tip_density": float(metrics.get("tip_density", np.nan)),
            "interface_roughness": float(metrics.get("interface_roughness", np.nan)),
            "sholl_ramification_index": float(metrics.get("sholl_ramification_index", np.nan)),
            "sholl_total_intersections": float(metrics.get("sholl_total_intersections", np.nan)),
            "fractal_dimension": float(metrics.get("fractal_dimension", np.nan)),
        }

    except Exception as exc:
        return {
            "empirical_score": np.nan,
            "severity": "ERROR",
            "severity_id": -1,
            "dendrite_coverage": np.nan,
            "branching_density": np.nan,
            "tip_density": np.nan,
            "interface_roughness": np.nan,
            "sholl_ramification_index": np.nan,
            "sholl_total_intersections": np.nan,
            "fractal_dimension": np.nan,
            "morphology_error": repr(exc),
        }


# ============================================================
# 2. 加载 train / val / test 全部数据
# ============================================================

def load_all_splits(config: dict):
    """
    加载 train / val / test。

    重要：
    这里 evaluation 不使用数据增强 transform。
    因为我们要评估真实样本的 latent 结构，而不是增强后的输入。
    """
    loaders = {}

    for split in config["splits"]:
        dataset = DendritePFMDataset(
            image_size=config["image_size"],
            json_path=config["split_json"],
            split=split,
            transform=None,
        )

        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )

        loaders[split] = loader

    return loaders


@torch.no_grad()
def collect_raw_dataset_table(loaders: dict, config: dict, device):
    """
    收集所有 split 的原始数据。

    返回：
    - df_base:
        每个样本一行，包含 split / did / true_y / morphology metrics。
    - images:
        clean target xo, shape [N, C, H, W]。
        PCA baseline 会使用这个图像。
    - x_inputs:
        model 输入 x，shape [N, C, H, W]。
        因为本脚本 evaluation 不使用 transform，所以 x 和 xo 通常一致。
    """
    rows = []
    images = []
    x_inputs = []

    sample_id = 0

    for split, loader in loaders.items():
        for x, y, did, xo in loader:
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            did_np = did.detach().cpu().numpy().astype(int)
            xo_np = xo.detach().cpu().numpy()

            batch_size = x_np.shape[0]

            for i in range(batch_size):
                row = {
                    "sample_id": int(sample_id),
                    "split": split,
                    "did": int(did_np[i]),
                }

                for j, name in enumerate(Y_NAMES):
                    row[f"true_{name}"] = float(y_np[i, j])

                if config["compute_morphology"]:
                    img2d = xo_np[i, config["morph_channel"]]
                    row.update(compute_morphology_summary(img2d))
                else:
                    row.update({
                        "empirical_score": np.nan,
                        "severity": "NA",
                        "severity_id": -1,
                    })

                rows.append(row)
                sample_id += 1

            images.append(xo_np)
            x_inputs.append(x_np)

    df_base = pd.DataFrame(rows)
    images = np.concatenate(images, axis=0)
    x_inputs = np.concatenate(x_inputs, axis=0)

    return df_base, images, x_inputs


# ============================================================
# 3. 收集 MDN-VAE 表征和预测
# ============================================================

@torch.no_grad()
def collect_mdn_representation(
    model,
    x_inputs: np.ndarray,
    df_base: pd.DataFrame,
    config: dict,
    device,
):
    """
    从 MDN-VAE 中收集：
    - representation:
        mu_q 或 sampled z
    - prediction:
        MDN mixture mean theta_hat

    这里为了几何分析稳定，默认 latent_source="mu"。
    如果使用 mu，则 MDN 预测也直接从 mu 输入 mdn_head 得到，
    避免 sampled z 带来的随机噪声。
    """
    model.eval()

    batch_size = config["batch_size"]
    latent_list = []
    pred_list = []
    conf_list = []

    for start in range(0, len(x_inputs), batch_size):
        end = min(start + batch_size, len(x_inputs))

        x = torch.tensor(x_inputs[start:end], dtype=torch.float32, device=device)

        recon, mu_q, logvar_q, mdn_out, z_sample = model(x)

        if config["latent_source"] == "mu":
            latent = mu_q
            pi, mdn_mu, log_sigma = model.mdn_head(mu_q)
        else:
            latent = z_sample
            pi, mdn_mu, log_sigma = mdn_out

        theta_hat, conf_param, conf_global, modes = mdn_point_and_confidence(
            pi,
            mdn_mu,
            log_sigma,
            var_scale=config["var_scale"],
            topk=3,
        )

        latent_list.append(latent.detach().cpu().numpy())
        pred_list.append(theta_hat.detach().cpu().numpy())
        conf_list.append(conf_global.detach().cpu().numpy())

    representations = np.concatenate(latent_list, axis=0)
    predictions = np.concatenate(pred_list, axis=0)
    confidence = np.concatenate(conf_list, axis=0)

    df = df_base.copy()
    df["model"] = "MDN_VAE"
    df["conf_global"] = confidence

    for j, name in enumerate(Y_NAMES):
        df[f"pred_{name}"] = predictions[:, j]
        df[f"abs_err_{name}"] = np.abs(df[f"pred_{name}"] - df[f"true_{name}"])

    return representations, predictions, df


# ============================================================
# 4. 收集 Linear-Regressor VAE 表征和预测
# ============================================================

@torch.no_grad()
def collect_linear_vae_representation(
    model,
    x_inputs: np.ndarray,
    df_base: pd.DataFrame,
    config: dict,
    device,
):
    """
    从 Linear-Regressor VAE 中收集：
    - representation:
        mu_q 或 sampled z
    - prediction:
        theta_hat = Linear(latent)

    上传的 linear model 中 forward 返回：
        recon, mu_q, logvar_q, theta_hat, z
    """
    model.eval()

    batch_size = config["batch_size"]
    latent_list = []
    pred_list = []

    for start in range(0, len(x_inputs), batch_size):
        end = min(start + batch_size, len(x_inputs))

        x = torch.tensor(x_inputs[start:end], dtype=torch.float32, device=device)

        recon, mu_q, logvar_q, theta_hat_forward, z_sample = model(x)

        if config["latent_source"] == "mu":
            latent = mu_q

            # 为了确定性，用 mu_q 直接过 regression_head。
            if hasattr(model, "regression_head"):
                theta_hat = model.regression_head(mu_q)
            else:
                theta_hat = theta_hat_forward
        else:
            latent = z_sample
            theta_hat = theta_hat_forward

        latent_list.append(latent.detach().cpu().numpy())
        pred_list.append(theta_hat.detach().cpu().numpy())

    representations = np.concatenate(latent_list, axis=0)
    predictions = np.concatenate(pred_list, axis=0)

    df = df_base.copy()
    df["model"] = "LinearReg_VAE"
    df["conf_global"] = np.nan

    for j, name in enumerate(Y_NAMES):
        df[f"pred_{name}"] = predictions[:, j]
        df[f"abs_err_{name}"] = np.abs(df[f"pred_{name}"] - df[f"true_{name}"])

    return representations, predictions, df


# ============================================================
# 5. PCA baseline: PCA image features + Ridge linear probe
# ============================================================

def collect_pca_representation(
    images: np.ndarray,
    df_base: pd.DataFrame,
    config: dict,
):
    """
    PCA baseline 的逻辑：

    1. 将 clean target image xo flatten 成向量。
    2. 只在 train split 上 fit StandardScaler 和 PCA。
    3. 用 PCA transform train / val / test 全部样本。
    4. 只在 train split 上训练 Ridge regression：
            PCA features -> y
    5. 预测 train / val / test 全部样本。

    这样 PCA 组可以和 MDN / LinearReg 一样进入：
    - prediction metric
    - trajectory temporal order
    - kNN physical consistency

    注意：
    PCA 没有 decoder，所以不做 latent interpolation 解码。
    """
    n = images.shape[0]
    x_flat = images.reshape(n, -1).astype(np.float32)

    train_mask = df_base["split"].to_numpy() == "train"

    y_all = df_base[[f"true_{name}" for name in Y_NAMES]].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_flat[train_mask])
    x_all_scaled = scaler.transform(x_flat)

    pca = PCA(
        n_components=config["pca_dim"],
        whiten=config["pca_whiten"],
        random_state=config["seed"],
    )

    z_train = pca.fit_transform(x_train_scaled)
    representations = pca.transform(x_all_scaled)

    reg = Ridge(alpha=config["ridge_alpha"])
    reg.fit(z_train, y_all[train_mask])

    predictions = reg.predict(representations)

    df = df_base.copy()
    df["model"] = "PCA_Ridge"
    df["conf_global"] = np.nan

    for j, name in enumerate(Y_NAMES):
        df[f"pred_{name}"] = predictions[:, j]
        df[f"abs_err_{name}"] = np.abs(df[f"pred_{name}"] - df[f"true_{name}"])

    pca_info = {
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "ridge_alpha": float(config["ridge_alpha"]),
        "pca_dim": int(config["pca_dim"]),
    }

    return representations, predictions, df, pca_info


# ============================================================
# 6. 参数预测性能
# ============================================================

def parameter_prediction_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 model、每个 split、每个参数，计算预测性能。

    输出包括：
    - MAE
    - MSE
    - R2
    - Spearman rho
    - Spearman p

    split_scope:
    - train
    - val
    - test
    - all
    """
    rows = []

    split_scopes = sorted(df["split"].unique().tolist()) + ["all"]

    for model_name, g_model in df.groupby("model"):
        for split in split_scopes:
            if split == "all":
                g = g_model
            else:
                g = g_model[g_model["split"] == split]

            if len(g) == 0:
                continue

            for name in Y_NAMES:
                y_true = g[f"true_{name}"].to_numpy(dtype=float)
                y_pred = g[f"pred_{name}"].to_numpy(dtype=float)

                mae = float(np.nanmean(np.abs(y_pred - y_true)))
                mse = float(np.nanmean((y_pred - y_true) ** 2))
                r2 = safe_r2(y_true, y_pred)
                rho, p = safe_spearman(y_true, y_pred)

                rows.append({
                    "model": model_name,
                    "split": split,
                    "target": name,
                    "n": int(len(g)),
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                    "spearman_rho": rho,
                    "spearman_p": p,
                })

            # 全部参数整体误差
            true_mat = g[[f"true_{name}" for name in Y_NAMES]].to_numpy(dtype=float)
            pred_mat = g[[f"pred_{name}" for name in Y_NAMES]].to_numpy(dtype=float)

            rows.append({
                "model": model_name,
                "split": split,
                "target": "__all_params__",
                "n": int(len(g)),
                "mae": float(np.nanmean(np.abs(pred_mat - true_mat))),
                "mse": float(np.nanmean((pred_mat - true_mat) ** 2)),
                "r2": np.nan,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
            })

    return pd.DataFrame(rows)


# ============================================================
# 7. 同一轨迹时间有序性测试
# ============================================================

def trajectory_temporal_order_test(
    representations: np.ndarray,
    df: pd.DataFrame,
    representation_name: str,
) -> pd.DataFrame:
    """
    对每个 model / split_scope / did 进行轨迹时间有序性测试。

    核心指标：
    1. rho_true_t_vs_pred_t
       同一 did 内，真实时间和预测时间是否单调一致。

    2. rho_abs_delta_t_vs_repr_distance
       同一 did 内，任意两个样本的时间间隔 |delta_t| 是否和
       representation distance 正相关。

    3. adjacent_distance_mean vs non_adjacent_distance_mean
       相邻时间步距离是否更小。

    注意：
    split_scope 包括：
    - train / val / test
    - all

    all 很重要，因为它使用 train+val+test 的完整轨迹。
    """
    rows = []
    split_scopes = sorted(df["split"].unique().tolist()) + ["all"]

    for split in split_scopes:
        if split == "all":
            g_split = df
        else:
            g_split = df[df["split"] == split]

        for did, g0 in g_split.groupby("did"):
            if len(g0) < 4:
                continue

            g = g0.sort_values("true_t")
            idx = g.index.to_numpy()

            z = representations[idx]
            true_t = g["true_t"].to_numpy(dtype=float)
            pred_t = g["pred_t"].to_numpy(dtype=float)

            rho_true_pred, p_true_pred = safe_spearman(true_t, pred_t)

            dt_list = []
            dz_list = []
            adjacent_dz = []
            non_adjacent_dz = []

            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    dt = abs(true_t[a] - true_t[b])
                    dz = float(np.linalg.norm(z[a] - z[b]))

                    dt_list.append(dt)
                    dz_list.append(dz)

                    if b == a + 1:
                        adjacent_dz.append(dz)
                    else:
                        non_adjacent_dz.append(dz)

            rho_dt_dz, p_dt_dz = safe_spearman(dt_list, dz_list)
            stat_adj, p_adj = safe_mannwhitneyu(adjacent_dz, non_adjacent_dz)

            rows.append({
                "model": representation_name,
                "split_scope": split,
                "did": int(did),
                "n_frames": int(len(g)),

                "rho_true_t_vs_pred_t": rho_true_pred,
                "p_true_t_vs_pred_t": p_true_pred,

                "rho_abs_delta_t_vs_repr_distance": rho_dt_dz,
                "p_abs_delta_t_vs_repr_distance": p_dt_dz,

                "adjacent_distance_mean": float(np.nanmean(adjacent_dz)) if len(adjacent_dz) else np.nan,
                "non_adjacent_distance_mean": float(np.nanmean(non_adjacent_dz)) if len(non_adjacent_dz) else np.nan,
                "adjacent_vs_non_adjacent_p": p_adj,

                "supports_pred_t_order": bool(np.isfinite(rho_true_pred) and rho_true_pred > 0),
                "supports_repr_temporal_geometry": bool(np.isfinite(rho_dt_dz) and rho_dt_dz > 0),
                "supports_adjacent_closer": bool(
                    len(adjacent_dz)
                    and len(non_adjacent_dz)
                    and np.nanmean(adjacent_dz) < np.nanmean(non_adjacent_dz)
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# 8. kNN 局部物理一致性测试
# ============================================================

def knn_physical_consistency_test(
    representations: np.ndarray,
    df: pd.DataFrame,
    representation_name: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    测试 representation 近邻是否比 random baseline 更物理相似。

    每个样本：
    - 找 k 个 representation 最近邻
    - 随机抽 k 个样本，重复 n_random_trials 次
    - 比较 kNN 和 random 在以下方面的差异：

        true_t difference
        pred_t difference
        true physical parameter difference
        pred physical parameter difference
        empirical_score difference, if available
        same_did_ratio

    输出：
    - per_sample_df:
        每个样本一行。
    - summary_df:
        每个指标一行，比较 kNN mean 和 random mean。
    """
    rng = np.random.default_rng(config["seed"])

    n = len(df)
    k = min(config["knn_k"], n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(representations)
    distances, indices = nn.kneighbors(representations)

    true_param_cols = [f"true_{name}" for name in PARAM_NAMES]
    pred_param_cols = [f"pred_{name}" for name in PARAM_NAMES]

    rows = []

    for i in range(n):
        knn_idx = indices[i, 1:]
        knn_dist = distances[i, 1:]

        candidates = np.setdiff1d(np.arange(n), np.array([i]))

        random_acc = defaultdict(list)

        for _ in range(config["n_random_trials"]):
            rand_idx = rng.choice(candidates, size=k, replace=False)

            random_acc["random_abs_true_t_diff"].append(
                np.mean(np.abs(df.loc[rand_idx, "true_t"].to_numpy() - df.loc[i, "true_t"]))
            )
            random_acc["random_abs_pred_t_diff"].append(
                np.mean(np.abs(df.loc[rand_idx, "pred_t"].to_numpy() - df.loc[i, "pred_t"]))
            )
            random_acc["random_same_did_ratio"].append(
                np.mean(df.loc[rand_idx, "did"].to_numpy() == df.loc[i, "did"])
            )

            true_param_diffs = []
            pred_param_diffs = []

            for c in true_param_cols:
                true_param_diffs.append(
                    np.mean(np.abs(df.loc[rand_idx, c].to_numpy() - df.loc[i, c]))
                )

            for c in pred_param_cols:
                pred_param_diffs.append(
                    np.mean(np.abs(df.loc[rand_idx, c].to_numpy() - df.loc[i, c]))
                )

            random_acc["random_abs_true_param_diff"].append(np.mean(true_param_diffs))
            random_acc["random_abs_pred_param_diff"].append(np.mean(pred_param_diffs))

            if "empirical_score" in df.columns and np.isfinite(df.loc[i, "empirical_score"]):
                random_acc["random_abs_empirical_score_diff"].append(
                    np.nanmean(
                        np.abs(
                            df.loc[rand_idx, "empirical_score"].to_numpy()
                            - df.loc[i, "empirical_score"]
                        )
                    )
                )

        true_param_diffs = []
        pred_param_diffs = []

        for c in true_param_cols:
            true_param_diffs.append(
                np.mean(np.abs(df.loc[knn_idx, c].to_numpy() - df.loc[i, c]))
            )

        for c in pred_param_cols:
            pred_param_diffs.append(
                np.mean(np.abs(df.loc[knn_idx, c].to_numpy() - df.loc[i, c]))
            )

        row = {
            "model": representation_name,
            "sample_id": int(df.loc[i, "sample_id"]),
            "split": df.loc[i, "split"],
            "did": int(df.loc[i, "did"]),

            "mean_knn_repr_distance": float(np.mean(knn_dist)),

            "knn_abs_true_t_diff": float(
                np.mean(np.abs(df.loc[knn_idx, "true_t"].to_numpy() - df.loc[i, "true_t"]))
            ),
            "knn_abs_pred_t_diff": float(
                np.mean(np.abs(df.loc[knn_idx, "pred_t"].to_numpy() - df.loc[i, "pred_t"]))
            ),
            "knn_same_did_ratio": float(
                np.mean(df.loc[knn_idx, "did"].to_numpy() == df.loc[i, "did"])
            ),
            "knn_abs_true_param_diff": float(np.mean(true_param_diffs)),
            "knn_abs_pred_param_diff": float(np.mean(pred_param_diffs)),
        }

        if "empirical_score" in df.columns and np.isfinite(df.loc[i, "empirical_score"]):
            row["knn_abs_empirical_score_diff"] = float(
                np.nanmean(
                    np.abs(
                        df.loc[knn_idx, "empirical_score"].to_numpy()
                        - df.loc[i, "empirical_score"]
                    )
                )
            )

        for key, values in random_acc.items():
            row[key] = float(np.nanmean(values))

        rows.append(row)

    per_sample_df = pd.DataFrame(rows)

    summary_rows = []

    metric_pairs = [
        ("knn_abs_true_t_diff", "random_abs_true_t_diff", "lower_is_better"),
        ("knn_abs_pred_t_diff", "random_abs_pred_t_diff", "lower_is_better"),
        ("knn_abs_true_param_diff", "random_abs_true_param_diff", "lower_is_better"),
        ("knn_abs_pred_param_diff", "random_abs_pred_param_diff", "lower_is_better"),
        ("knn_same_did_ratio", "random_same_did_ratio", "higher_is_better"),
        ("knn_abs_empirical_score_diff", "random_abs_empirical_score_diff", "lower_is_better"),
    ]

    split_scopes = sorted(per_sample_df["split"].unique().tolist()) + ["all"]

    for split in split_scopes:
        if split == "all":
            g = per_sample_df
        else:
            g = per_sample_df[per_sample_df["split"] == split]

        for knn_col, random_col, direction in metric_pairs:
            if knn_col not in g.columns or random_col not in g.columns:
                continue

            knn_values = g[knn_col].to_numpy(dtype=float)
            random_values = g[random_col].to_numpy(dtype=float)

            stat, p = safe_mannwhitneyu(knn_values, random_values)

            knn_mean = float(np.nanmean(knn_values))
            random_mean = float(np.nanmean(random_values))

            if direction == "lower_is_better":
                improvement = random_mean - knn_mean
            else:
                improvement = knn_mean - random_mean

            summary_rows.append({
                "model": representation_name,
                "split_scope": split,
                "knn_metric": knn_col,
                "random_metric": random_col,
                "direction": direction,
                "knn_mean": knn_mean,
                "random_mean": random_mean,
                "improvement": float(improvement),
                "mannwhitney_p": p,
                "supports_local_physicality": bool(np.isfinite(improvement) and improvement > 0),
                "n": int(len(g)),
            })

    summary_df = pd.DataFrame(summary_rows)
    return per_sample_df, summary_df


# ============================================================
# 9. 汇总比较表
# ============================================================

def summarize_trajectory_results(traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    将每个 did 的轨迹测试汇总到 model / split_scope 级别。
    """
    rows = []

    for (model, split_scope), g in traj_df.groupby(["model", "split_scope"]):
        rows.append({
            "model": model,
            "split_scope": split_scope,
            "n_trajectories": int(g["did"].nunique()),

            "mean_rho_true_t_vs_pred_t": float(np.nanmean(g["rho_true_t_vs_pred_t"])),
            "median_rho_true_t_vs_pred_t": float(np.nanmedian(g["rho_true_t_vs_pred_t"])),

            "mean_rho_abs_delta_t_vs_repr_distance": float(
                np.nanmean(g["rho_abs_delta_t_vs_repr_distance"])
            ),
            "median_rho_abs_delta_t_vs_repr_distance": float(
                np.nanmedian(g["rho_abs_delta_t_vs_repr_distance"])
            ),

            "fraction_positive_true_t_pred_t": float(np.nanmean(g["supports_pred_t_order"])),
            "fraction_positive_delta_t_repr_distance": float(
                np.nanmean(g["supports_repr_temporal_geometry"])
            ),
            "fraction_adjacent_closer": float(np.nanmean(g["supports_adjacent_closer"])),

            "mean_adjacent_distance": float(np.nanmean(g["adjacent_distance_mean"])),
            "mean_non_adjacent_distance": float(np.nanmean(g["non_adjacent_distance_mean"])),
        })

    return pd.DataFrame(rows)


def make_final_comparison_table(
    pred_metric_df: pd.DataFrame,
    traj_summary_df: pd.DataFrame,
    knn_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    生成一个最适合论文/汇报的总对比表。

    每个 model + split_scope 一行，包含：
    - all_params_mae
    - t_mae
    - t_spearman
    - trajectory temporal geometry rho
    - kNN true param improvement
    - kNN empirical score improvement
    """
    rows = []

    models = sorted(pred_metric_df["model"].unique().tolist())
    split_scopes = sorted(set(pred_metric_df["split"].unique().tolist()) | set(traj_summary_df["split_scope"].unique().tolist()))

    for model in models:
        for split in split_scopes:
            row = {
                "model": model,
                "split_scope": split,
            }

            # parameter prediction metrics
            g_pred = pred_metric_df[
                (pred_metric_df["model"] == model)
                & (pred_metric_df["split"] == split)
            ]

            hit_all = g_pred[g_pred["target"] == "__all_params__"]
            if len(hit_all):
                row["all_params_mae"] = float(hit_all.iloc[0]["mae"])
                row["all_params_mse"] = float(hit_all.iloc[0]["mse"])
            else:
                row["all_params_mae"] = np.nan
                row["all_params_mse"] = np.nan

            hit_t = g_pred[g_pred["target"] == "t"]
            if len(hit_t):
                row["t_mae"] = float(hit_t.iloc[0]["mae"])
                row["t_r2"] = float(hit_t.iloc[0]["r2"]) if np.isfinite(hit_t.iloc[0]["r2"]) else np.nan
                row["t_spearman_rho"] = float(hit_t.iloc[0]["spearman_rho"])
            else:
                row["t_mae"] = np.nan
                row["t_r2"] = np.nan
                row["t_spearman_rho"] = np.nan

            # trajectory metrics
            g_traj = traj_summary_df[
                (traj_summary_df["model"] == model)
                & (traj_summary_df["split_scope"] == split)
            ]

            if len(g_traj):
                r = g_traj.iloc[0]
                row["traj_mean_rho_true_t_vs_pred_t"] = r["mean_rho_true_t_vs_pred_t"]
                row["traj_mean_rho_delta_t_vs_repr_dist"] = r["mean_rho_abs_delta_t_vs_repr_distance"]
                row["traj_fraction_adjacent_closer"] = r["fraction_adjacent_closer"]
                row["traj_n"] = r["n_trajectories"]
            else:
                row["traj_mean_rho_true_t_vs_pred_t"] = np.nan
                row["traj_mean_rho_delta_t_vs_repr_dist"] = np.nan
                row["traj_fraction_adjacent_closer"] = np.nan
                row["traj_n"] = 0

            # kNN improvements
            g_knn = knn_summary_df[
                (knn_summary_df["model"] == model)
                & (knn_summary_df["split_scope"] == split)
            ]

            def get_improvement(metric_name):
                hit = g_knn[g_knn["knn_metric"] == metric_name]
                if len(hit):
                    return float(hit.iloc[0]["improvement"])
                return np.nan

            row["knn_true_t_improvement"] = get_improvement("knn_abs_true_t_diff")
            row["knn_pred_t_improvement"] = get_improvement("knn_abs_pred_t_diff")
            row["knn_true_param_improvement"] = get_improvement("knn_abs_true_param_diff")
            row["knn_pred_param_improvement"] = get_improvement("knn_abs_pred_param_diff")
            row["knn_same_did_improvement"] = get_improvement("knn_same_did_ratio")
            row["knn_empirical_score_improvement"] = get_improvement("knn_abs_empirical_score_diff")

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 10. 可视化
# ============================================================

def plot_final_comparison(final_df: pd.DataFrame, out_dir: str):
    """
    画几张简单对比图。
    """
    ensure_dir(out_dir)

    metrics_to_plot = [
        "all_params_mae",
        "t_mae",
        "t_spearman_rho",
        "traj_mean_rho_delta_t_vs_repr_dist",
        "knn_true_param_improvement",
        "knn_same_did_improvement",
    ]

    for metric in metrics_to_plot:
        if metric not in final_df.columns:
            continue

        g = final_df[final_df["split_scope"] == "all"].copy()
        if len(g) == 0:
            continue

        plt.figure(figsize=(6, 4))
        plt.bar(g["model"], g[metric])
        plt.ylabel(metric)
        plt.title(f"{metric} on all splits")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"compare_{metric}_all.png"), dpi=300)
        plt.close()


# ============================================================
# 11. 写 summary.txt
# ============================================================

def write_summary(
    out_path: str,
    config: dict,
    df_all_models: pd.DataFrame,
    final_df: pd.DataFrame,
    pca_info: dict,
):
    """
    写一个人类可读 summary。
    """
    lines = []

    lines.append("Latent Physicality Comparison on Train + Val + Test")
    lines.append("=" * 72)
    lines.append("")

    lines.append("Dataset")
    lines.append("-" * 72)
    base = df_all_models[df_all_models["model"] == df_all_models["model"].iloc[0]]
    lines.append(f"n_samples: {len(base)}")
    lines.append(f"n_did: {base['did'].nunique()}")
    lines.append(f"splits: {sorted(base['split'].unique().tolist())}")
    for split, g in base.groupby("split"):
        lines.append(f"  {split}: n={len(g)}, n_did={g['did'].nunique()}")
    lines.append("")

    lines.append("Compared representations")
    lines.append("-" * 72)
    lines.append("1. MDN_VAE: encoder latent from the MDN-VAE model")
    lines.append("2. LinearReg_VAE: encoder latent from the VAE with linear regression head")
    lines.append("3. PCA_Ridge: PCA image features with Ridge linear probe")
    lines.append("")

    lines.append("PCA info")
    lines.append("-" * 72)
    lines.append(json.dumps(pca_info, indent=2, ensure_ascii=False))
    lines.append("")

    lines.append("Final comparison, split_scope = all")
    lines.append("-" * 72)
    g_all = final_df[final_df["split_scope"] == "all"].copy()
    if len(g_all):
        lines.append(g_all.to_string(index=False))
    else:
        lines.append("No all-scope comparison found.")
    lines.append("")

    lines.append("Interpretation guide")
    lines.append("-" * 72)
    lines.append(
        "A physically meaningful representation should show low parameter prediction error, "
        "positive trajectory temporal geometry, and positive kNN improvements over random baseline."
    )
    lines.append(
        "The most important geometry metric is traj_mean_rho_delta_t_vs_repr_dist. "
        "If it is positive and large, then samples farther apart in physical time are also farther "
        "apart in the representation space within the same simulation trajectory."
    )
    lines.append(
        "The most important locality metric is knn_true_param_improvement. "
        "If it is positive, representation neighbors are closer in true physical parameters than "
        "random samples."
    )
    lines.append(
        "PCA_Ridge is a classical image-feature baseline. If MDN_VAE or LinearReg_VAE does not "
        "outperform PCA_Ridge on temporal geometry or kNN physical consistency, then the learned "
        "latent space may not provide additional physical organization beyond low-dimensional image variance."
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 12. Main: 无命令行参数，直接运行
# ============================================================

def main():
    config = CONFIG

    set_seed(config["seed"])

    out_dir = ensure_dir(config["out_dir"])
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] output dir = {out_dir}")

    # --------------------------------------------------------
    # 1. Load all data
    # --------------------------------------------------------
    print("[INFO] Loading train / val / test datasets...")
    loaders = load_all_splits(config)

    print("[INFO] Collecting raw data table and clean images...")
    df_base, images, x_inputs = collect_raw_dataset_table(
        loaders=loaders,
        config=config,
        device=device,
    )

    df_base.to_csv(os.path.join(out_dir, "base_samples_all_splits.csv"), index=False)
    print(f"[INFO] Total samples: {len(df_base)}")
    print(f"[INFO] Total did: {df_base['did'].nunique()}")

    # --------------------------------------------------------
    # 2. Load MDN model
    # --------------------------------------------------------
    mdn_model_path = os.path.join(
        config["mdn_model_root"],
        "ckpt",
        config["ckpt_name"],
    )

    print(f"[INFO] Loading MDN model: {mdn_model_path}")
    mdn_model = torch.load(
        mdn_model_path,
        map_location=device,
        weights_only=False,
    )
    mdn_model = mdn_model.to(device)
    mdn_model.eval()

    # --------------------------------------------------------
    # 3. Load Linear-Regressor VAE model
    # --------------------------------------------------------
    linear_model_path = os.path.join(
        config["linear_model_root"],
        "ckpt",
        config["ckpt_name"],
    )

    print(f"[INFO] Loading Linear-Regressor VAE model: {linear_model_path}")
    linear_model = torch.load(
        linear_model_path,
        map_location=device,
        weights_only=False,
    )
    linear_model = linear_model.to(device)
    linear_model.eval()

    # --------------------------------------------------------
    # 4. Collect representations and predictions
    # --------------------------------------------------------
    print("[INFO] Collecting MDN-VAE representations...")
    z_mdn, pred_mdn, df_mdn = collect_mdn_representation(
        model=mdn_model,
        x_inputs=x_inputs,
        df_base=df_base,
        config=config,
        device=device,
    )

    print("[INFO] Collecting Linear-Regressor VAE representations...")
    z_linear, pred_linear, df_linear = collect_linear_vae_representation(
        model=linear_model,
        x_inputs=x_inputs,
        df_base=df_base,
        config=config,
        device=device,
    )

    print("[INFO] Collecting PCA + Ridge baseline representations...")
    z_pca, pred_pca, df_pca, pca_info = collect_pca_representation(
        images=images,
        df_base=df_base,
        config=config,
    )

    # 保存 representation，方便后续复查。
    np.save(os.path.join(out_dir, "z_mdn.npy"), z_mdn)
    np.save(os.path.join(out_dir, "z_linear.npy"), z_linear)
    np.save(os.path.join(out_dir, "z_pca.npy"), z_pca)

    # 合并所有模型的逐样本结果。
    df_all_models = pd.concat(
        [df_mdn, df_linear, df_pca],
        axis=0,
        ignore_index=True,
    )

    df_all_models.to_csv(
        os.path.join(out_dir, "samples_all_models_all_splits.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # 5. Parameter prediction metrics
    # --------------------------------------------------------
    print("[INFO] Computing parameter prediction metrics...")
    pred_metric_df = parameter_prediction_metrics(df_all_models)
    pred_metric_df.to_csv(
        os.path.join(out_dir, "parameter_prediction_metrics.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # 6. Trajectory temporal order tests
    # --------------------------------------------------------
    print("[INFO] Computing trajectory temporal-order metrics...")

    traj_mdn = trajectory_temporal_order_test(
        representations=z_mdn,
        df=df_mdn,
        representation_name="MDN_VAE",
    )

    traj_linear = trajectory_temporal_order_test(
        representations=z_linear,
        df=df_linear,
        representation_name="LinearReg_VAE",
    )

    traj_pca = trajectory_temporal_order_test(
        representations=z_pca,
        df=df_pca,
        representation_name="PCA_Ridge",
    )

    traj_df = pd.concat(
        [traj_mdn, traj_linear, traj_pca],
        axis=0,
        ignore_index=True,
    )

    traj_df.to_csv(
        os.path.join(out_dir, "trajectory_temporal_order_all_models.csv"),
        index=False,
    )

    traj_summary_df = summarize_trajectory_results(traj_df)
    traj_summary_df.to_csv(
        os.path.join(out_dir, "trajectory_temporal_order_summary.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # 7. kNN physical consistency tests
    # --------------------------------------------------------
    print("[INFO] Computing kNN physical-consistency metrics...")

    knn_mdn, knn_summary_mdn = knn_physical_consistency_test(
        representations=z_mdn,
        df=df_mdn,
        representation_name="MDN_VAE",
        config=config,
    )

    knn_linear, knn_summary_linear = knn_physical_consistency_test(
        representations=z_linear,
        df=df_linear,
        representation_name="LinearReg_VAE",
        config=config,
    )

    knn_pca, knn_summary_pca = knn_physical_consistency_test(
        representations=z_pca,
        df=df_pca,
        representation_name="PCA_Ridge",
        config=config,
    )

    knn_per_sample_df = pd.concat(
        [knn_mdn, knn_linear, knn_pca],
        axis=0,
        ignore_index=True,
    )

    knn_summary_df = pd.concat(
        [knn_summary_mdn, knn_summary_linear, knn_summary_pca],
        axis=0,
        ignore_index=True,
    )

    knn_per_sample_df.to_csv(
        os.path.join(out_dir, "knn_physical_consistency_per_sample_all_models.csv"),
        index=False,
    )

    knn_summary_df.to_csv(
        os.path.join(out_dir, "knn_physical_consistency_summary_all_models.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # 8. Final compact comparison table
    # --------------------------------------------------------
    print("[INFO] Building final comparison table...")

    final_df = make_final_comparison_table(
        pred_metric_df=pred_metric_df,
        traj_summary_df=traj_summary_df,
        knn_summary_df=knn_summary_df,
    )

    final_df.to_csv(
        os.path.join(out_dir, "final_comparison_table.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # 9. Figures
    # --------------------------------------------------------
    print("[INFO] Plotting final comparison figures...")
    plot_final_comparison(final_df, fig_dir)

    # --------------------------------------------------------
    # 10. Summary
    # --------------------------------------------------------
    print("[INFO] Writing summary...")
    write_summary(
        out_path=os.path.join(out_dir, "summary.txt"),
        config=config,
        df_all_models=df_all_models,
        final_df=final_df,
        pca_info=pca_info,
    )

    print("")
    print("[DONE] All comparisons finished.")
    print(f"[DONE] Results saved to: {out_dir}")
    print("")
    print("Most important files:")
    print(f"  {os.path.join(out_dir, 'final_comparison_table.csv')}")
    print(f"  {os.path.join(out_dir, 'parameter_prediction_metrics.csv')}")
    print(f"  {os.path.join(out_dir, 'trajectory_temporal_order_summary.csv')}")
    print(f"  {os.path.join(out_dir, 'knn_physical_consistency_summary_all_models.csv')}")
    print(f"  {os.path.join(out_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()