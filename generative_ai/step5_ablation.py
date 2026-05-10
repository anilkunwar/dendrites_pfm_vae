# generative_ai/test_mdn_mae_hit_confidence_clean.py
# -*- coding: utf-8 -*-
"""
Clean MDN probabilistic ablation.

Main output:
1. Three-model comparison:
   - PCA_Ridge
   - Linear_Joint
   - MDN_sampling

   Metrics:
   - mae
   - t_mae
   - phys14_mae
   - hit_prob@0.01
   - hit_prob@0.02
   - hit_auc

2. MDN confidence vs hit:
   - Spearman(confidence, hit_prob@0.01)
   - Spearman(confidence, hit_prob@0.02)
   - confidence-bin table for manual inspection

Important definitions:
----------------------
For a normalized 15-D parameter vector y:

    vector_rmse = sqrt(mean_j((pred_j - true_j)^2))

For deterministic models:
    hit_prob@eps = fraction of samples with vector_rmse <= eps

For MDN:
    For each input x, sample S parameter vectors from p(y|x).
    hit_prob@eps = average fraction of sampled vectors with vector_rmse <= eps

hit_auc:
    Area under hit-probability curve over HIT_AUC_TOLS.
    This gives one compact score of overall neighborhood-hit ability.
"""

import os
import json
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.dataloader import DendritePFMDataset, PARAM_RANGES
from src.modelv11 import mdn_point_and_confidence
from src.model_linear_regression import VAE_LinearRegression  # needed for torch.load full Linear model


# ============================================================
# 0. Config
# ============================================================

CONFIG = {
    # Dataset
    "split_json": "data/dataset_split.json",
    "splits": ["train", "val", "test"],
    "image_size": (3, 48, 48),
    "batch_size": 64,
    "num_workers": 4,

    # Trained models
    "mdn_model_path": "results/joint_train_physicality_compare/ckpt/mdn_joint_best.pt",
    "linear_model_path": "results/joint_train_physicality_compare/ckpt/linear_joint_best.pt",

    # PCA baseline
    "pca_dim": 16,
    "pca_whiten": False,
    "ridge_alpha": 1.0,

    # MDN sampling
    "n_mdn_samples": 512,
    "clip_mdn_samples_to_01": False,

    # Main hit probabilities shown in final table
    "main_hit_tolerances": [0.01, 0.02],

    # Tolerances used only for hit_auc calculation
    # These will NOT all be printed as separate columns.
    "hit_auc_tolerances": [0.01, 0.02, 0.05, 0.10],

    # Only affects confidence mapping, not MDN sampling
    "var_scale": 1.0,

    "seed": 0,
    "out_dir": "results/mdn_mae_hit_confidence_clean",
}

Y_NAMES = ["t"] + list(PARAM_RANGES.keys())
N_PARAMS = len(Y_NAMES)  # should be 15


# ============================================================
# 1. Basic utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def safe_torch_load(path: str, device):
    """
    Compatible torch.load for different PyTorch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def safe_spearman(x, y) -> float:
    """
    Spearman correlation with simple validity checks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan

    x = x[mask]
    y = y[mask]

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    rho, _ = spearmanr(x, y)
    return float(rho)


def vector_rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Per-sample normalized vector RMSE.

    pred, true: [N, P]
    return: [N]
    """
    return np.sqrt(np.mean((pred - true) ** 2, axis=1))


def all_mae(pred: np.ndarray, true: np.ndarray) -> float:
    """
    MAE over all samples and all 15 normalized targets.
    """
    return float(np.mean(np.abs(pred - true)))


def t_mae(pred: np.ndarray, true: np.ndarray) -> float:
    """
    MAE for normalized time t only.
    """
    return float(np.mean(np.abs(pred[:, 0] - true[:, 0])))


def phys14_mae(pred: np.ndarray, true: np.ndarray) -> float:
    """
    MAE for the 14 physical parameters, excluding time t.
    """
    return float(np.mean(np.abs(pred[:, 1:] - true[:, 1:])))


def normalized_auc(x, y) -> float:
    """
    Normalized area under curve.

    x: tolerances
    y: hit probabilities

    Returns AUC divided by x-range, so result is in roughly [0, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan

    area = np.trapz(y, x)
    width = x.max() - x.min()

    if width <= 0:
        return np.nan

    return float(area / width)


# ============================================================
# 2. Data loading
# ============================================================

def load_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    loaders = {}

    for split in config["splits"]:
        dataset = DendritePFMDataset(
            image_size=config["image_size"],
            json_path=config["split_json"],
            split=split,
            transform=None,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            drop_last=False,
        )

    return loaders


@torch.no_grad()
def collect_data(loaders: Dict[str, DataLoader]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect all splits into arrays.

    Returns
    -------
    df:
        sample_id, split, did, true_*.
    x_inputs:
        input images for neural models.
    images:
        clean images for PCA baseline.
    y_true:
        normalized targets, [N, 15].
    """
    rows = []
    x_inputs = []
    images = []

    sample_id = 0

    for split, loader in loaders.items():
        for x, y, did, xo in loader:
            x_np = x.detach().cpu().numpy().astype(np.float32)
            xo_np = xo.detach().cpu().numpy().astype(np.float32)
            y_np = y.detach().cpu().numpy().astype(np.float32)
            did_np = did.detach().cpu().numpy().astype(int)

            for i in range(x_np.shape[0]):
                row = {
                    "sample_id": int(sample_id),
                    "split": split,
                    "did": int(did_np[i]),
                }

                for j, name in enumerate(Y_NAMES):
                    row[f"true_{name}"] = float(y_np[i, j])

                rows.append(row)
                sample_id += 1

            x_inputs.append(x_np)
            images.append(xo_np)

    df = pd.DataFrame(rows)
    x_inputs = np.concatenate(x_inputs, axis=0)
    images = np.concatenate(images, axis=0)

    y_true = df[[f"true_{name}" for name in Y_NAMES]].to_numpy(dtype=np.float32)

    return df, x_inputs, images, y_true


# ============================================================
# 3. PCA and Linear predictions
# ============================================================

def predict_pca_ridge(images: np.ndarray, y_true: np.ndarray, df: pd.DataFrame, config: Dict):
    """
    PCA_Ridge baseline.

    Fit scaler, PCA, and Ridge only on train split.
    """
    n = images.shape[0]
    x_flat = images.reshape(n, -1).astype(np.float32)

    train_mask = df["split"].to_numpy() == "train"

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_flat[train_mask])
    x_all_scaled = scaler.transform(x_flat)

    pca = PCA(
        n_components=config["pca_dim"],
        whiten=config["pca_whiten"],
        random_state=config["seed"],
    )

    z_train = pca.fit_transform(x_train_scaled)
    z_all = pca.transform(x_all_scaled)

    ridge = Ridge(alpha=config["ridge_alpha"])
    ridge.fit(z_train, y_true[train_mask])

    pred = ridge.predict(z_all).astype(np.float32)

    pca_info = {
        "pca_dim": int(config["pca_dim"]),
        "pca_whiten": bool(config["pca_whiten"]),
        "ridge_alpha": float(config["ridge_alpha"]),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }

    return pred, pca_info


@torch.no_grad()
def predict_linear(model, x_inputs: np.ndarray, config: Dict, device) -> np.ndarray:
    """
    Linear_Joint:
        x -> encoder -> mu_q -> regression_head -> y_hat
    """
    model.eval()

    preds = []
    bs = config["batch_size"]

    for start in range(0, len(x_inputs), bs):
        end = min(start + bs, len(x_inputs))

        x = torch.tensor(x_inputs[start:end], dtype=torch.float32, device=device)

        mu_q, _ = model.encoder(x)
        y_hat = model.regression_head(mu_q)

        preds.append(y_hat.detach().cpu().numpy())

    return np.concatenate(preds, axis=0).astype(np.float32)


# ============================================================
# 4. MDN sampling
# ============================================================

@torch.no_grad()
def sample_mdn(pi: torch.Tensor,
               mu: torch.Tensor,
               log_sigma: torch.Tensor,
               n_samples: int,
               clip_to_01: bool) -> torch.Tensor:
    """
    Sample from diagonal Gaussian mixture.

    Inputs
    ------
    pi: [B, K]
    mu: [B, K, P]
    log_sigma: [B, K, P]

    Returns
    -------
    samples: [S, B, P]
    """
    bsz, num_components, num_params = mu.shape

    pi = pi / (pi.sum(dim=-1, keepdim=True) + 1e-12)

    # Component index for each sample.
    comp_idx = torch.multinomial(
        pi,
        num_samples=n_samples,
        replacement=True,
    )  # [B, S]

    gather_idx = comp_idx.unsqueeze(-1).expand(bsz, n_samples, num_params)

    selected_mu = torch.gather(mu, dim=1, index=gather_idx)
    selected_log_sigma = torch.gather(log_sigma, dim=1, index=gather_idx)

    eps = torch.randn_like(selected_mu)
    samples = selected_mu + torch.exp(selected_log_sigma) * eps

    if clip_to_01:
        samples = torch.clamp(samples, 0.0, 1.0)

    return samples.permute(1, 0, 2).contiguous()  # [S, B, P]


@torch.no_grad()
def evaluate_mdn(model, x_inputs: np.ndarray, y_true: np.ndarray, config: Dict, device):
    """
    Evaluate MDN probabilistic prediction.

    For each input:
        1. encode x once
        2. get MDN distribution p(y|x)
        3. sample S parameter vectors
        4. compute hit probability at each tolerance
        5. collect confidence values
    """
    model.eval()

    bs = config["batch_size"]
    n_samples = config["n_mdn_samples"]

    # Use union of all needed tolerances.
    all_tols = sorted(set(config["main_hit_tolerances"] + config["hit_auc_tolerances"]))

    mixture_mean_preds = []
    hit_probs = {tol: [] for tol in all_tols}

    conf_param_mean_all = []
    conf_global_all = []
    conf_joint_all = []

    for start in range(0, len(x_inputs), bs):
        end = min(start + bs, len(x_inputs))

        x = torch.tensor(x_inputs[start:end], dtype=torch.float32, device=device)
        y = torch.tensor(y_true[start:end], dtype=torch.float32, device=device)

        # Joint training uses deterministic encoder mean for prediction head.
        mu_q, _ = model.encoder(x)
        pi, mdn_mu, log_sigma = model.mdn_head(mu_q)

        # Mixture mean and confidence.
        mixture_mean, conf_param, conf_global, _ = mdn_point_and_confidence(
            pi,
            mdn_mu,
            log_sigma,
            var_scale=config["var_scale"],
            topk=3,
        )

        # Draw samples from MDN distribution.
        samples = sample_mdn(
            pi=pi,
            mu=mdn_mu,
            log_sigma=log_sigma,
            n_samples=n_samples,
            clip_to_01=config["clip_mdn_samples_to_01"],
        )  # [S, B, P]

        # Distance from each sampled parameter vector to ground truth.
        dist = torch.sqrt(torch.mean((samples - y.unsqueeze(0)) ** 2, dim=-1))  # [S, B]

        # Hit probability for each input sample.
        for tol in all_tols:
            hp = torch.mean((dist <= tol).float(), dim=0)  # [B]
            hit_probs[tol].append(hp.detach().cpu().numpy())

        # Confidence definitions from mdn_point_and_confidence().
        conf_param_mean = conf_param.mean(dim=-1)   # [B]
        conf_joint = conf_param_mean * conf_global  # [B]

        mixture_mean_preds.append(mixture_mean.detach().cpu().numpy())
        conf_param_mean_all.append(conf_param_mean.detach().cpu().numpy())
        conf_global_all.append(conf_global.detach().cpu().numpy())
        conf_joint_all.append(conf_joint.detach().cpu().numpy())

    result = {
        "mixture_mean_pred": np.concatenate(mixture_mean_preds, axis=0).astype(np.float32),
        "conf_param_mean": np.concatenate(conf_param_mean_all, axis=0).astype(np.float32),
        "conf_global": np.concatenate(conf_global_all, axis=0).astype(np.float32),
        "conf_joint": np.concatenate(conf_joint_all, axis=0).astype(np.float32),
    }

    for tol in all_tols:
        result[f"hit_prob@{tol:g}"] = np.concatenate(hit_probs[tol], axis=0).astype(np.float32)

    return result


# ============================================================
# 5. Metrics and tables
# ============================================================

def deterministic_hit_probs(pred: np.ndarray, true: np.ndarray, tolerances) -> Dict[float, float]:
    """
    Hit rate for deterministic point prediction.
    """
    dist = vector_rmse(pred, true)
    return {tol: float(np.mean(dist <= tol)) for tol in tolerances}


def make_model_row(model_name: str,
                   split: str,
                   idx: np.ndarray,
                   pred: np.ndarray,
                   true: np.ndarray,
                   main_tols,
                   auc_tols,
                   mdn_hit_arrays: Dict[str, np.ndarray] = None) -> Dict:
    """
    Build one row for main table.

    For PCA / Linear:
        hit is computed from point prediction.

    For MDN:
        MAE uses mixture mean point prediction.
        hit is averaged from MDN sampling hit probabilities.
    """
    pred_i = pred[idx]
    true_i = true[idx]

    row = {
        "model": model_name,
        "split": split,
        "n": int(len(idx)),
        "mae": all_mae(pred_i, true_i),
        "t_mae": t_mae(pred_i, true_i),
        "phys14_mae": phys14_mae(pred_i, true_i),
    }

    if mdn_hit_arrays is None:
        # Deterministic model.
        hit_main = deterministic_hit_probs(pred_i, true_i, main_tols)
        hit_auc_values = deterministic_hit_probs(pred_i, true_i, auc_tols)

        for tol in main_tols:
            row[f"hit_prob@{tol:g}"] = hit_main[tol]

        y_auc = [hit_auc_values[tol] for tol in auc_tols]
        row["hit_auc"] = normalized_auc(auc_tols, y_auc)

    else:
        # MDN sampling model.
        for tol in main_tols:
            row[f"hit_prob@{tol:g}"] = float(np.mean(mdn_hit_arrays[f"hit_prob@{tol:g}"][idx]))

        y_auc = [
            float(np.mean(mdn_hit_arrays[f"hit_prob@{tol:g}"][idx]))
            for tol in auc_tols
        ]
        row["hit_auc"] = normalized_auc(auc_tols, y_auc)

    return row


def build_main_table(df: pd.DataFrame,
                     y_true: np.ndarray,
                     pred_pca: np.ndarray,
                     pred_linear: np.ndarray,
                     mdn_result: Dict,
                     config: Dict) -> pd.DataFrame:
    """
    Main table with only the requested compact metrics.
    """
    rows = []

    split_list = sorted(df["split"].unique().tolist()) + ["all"]
    main_tols = config["main_hit_tolerances"]
    auc_tols = config["hit_auc_tolerances"]

    for split in split_list:
        if split == "all":
            idx = np.arange(len(df))
        else:
            idx = np.where(df["split"].to_numpy() == split)[0]

        if len(idx) == 0:
            continue

        rows.append(
            make_model_row(
                model_name="PCA_Ridge",
                split=split,
                idx=idx,
                pred=pred_pca,
                true=y_true,
                main_tols=main_tols,
                auc_tols=auc_tols,
                mdn_hit_arrays=None,
            )
        )

        rows.append(
            make_model_row(
                model_name="Linear_Joint",
                split=split,
                idx=idx,
                pred=pred_linear,
                true=y_true,
                main_tols=main_tols,
                auc_tols=auc_tols,
                mdn_hit_arrays=None,
            )
        )

        rows.append(
            make_model_row(
                model_name="MDN_sampling",
                split=split,
                idx=idx,
                pred=mdn_result["mixture_mean_pred"],
                true=y_true,
                main_tols=main_tols,
                auc_tols=auc_tols,
                mdn_hit_arrays=mdn_result,
            )
        )

    out = pd.DataFrame(rows)

    model_order = {
        "PCA_Ridge": 0,
        "Linear_Joint": 1,
        "MDN_sampling": 2,
    }

    split_order = {
        "train": 0,
        "val": 1,
        "test": 2,
        "all": 3,
    }

    out["model_order"] = out["model"].map(model_order)
    out["split_order"] = out["split"].map(split_order)
    out = out.sort_values(["split_order", "model_order"])
    out = out.drop(columns=["model_order", "split_order"])

    return out


def build_per_sample_table(df: pd.DataFrame,
                           y_true: np.ndarray,
                           pred_pca: np.ndarray,
                           pred_linear: np.ndarray,
                           mdn_result: Dict,
                           config: Dict) -> pd.DataFrame:
    """
    Per-sample table for manual checking.
    """
    out = df.copy()

    out["pca_vector_rmse"] = vector_rmse(pred_pca, y_true)
    out["linear_vector_rmse"] = vector_rmse(pred_linear, y_true)
    out["mdn_mixture_mean_vector_rmse"] = vector_rmse(mdn_result["mixture_mean_pred"], y_true)

    out["conf_param_mean"] = mdn_result["conf_param_mean"]
    out["conf_global"] = mdn_result["conf_global"]
    out["conf_joint"] = mdn_result["conf_joint"]

    for tol in config["main_hit_tolerances"]:
        out[f"mdn_hit_prob@{tol:g}"] = mdn_result[f"hit_prob@{tol:g}"]

    return out


def build_confidence_vs_hit_table(per_sample: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Confidence vs MDN hit probability.

    Desired relationship:
        Spearman(confidence, hit_prob) > 0
    """
    rows = []

    confidence_cols = [
        "conf_param_mean",
        "conf_global",
        "conf_joint",
    ]

    split_list = sorted(per_sample["split"].unique().tolist()) + ["all"]

    for split in split_list:
        if split == "all":
            g = per_sample
        else:
            g = per_sample[per_sample["split"] == split]

        for tol in config["main_hit_tolerances"]:
            hit_col = f"mdn_hit_prob@{tol:g}"

            for conf_col in confidence_cols:
                rows.append({
                    "split": split,
                    "tolerance": tol,
                    "confidence": conf_col,
                    "n": int(len(g)),
                    "spearman_conf_vs_hit": safe_spearman(
                        g[conf_col].to_numpy(),
                        g[hit_col].to_numpy(),
                    ),
                })

    return pd.DataFrame(rows)


def build_confidence_bins(per_sample: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Bin samples by confidence and report average hit probability.

    For manual inspection:
        higher confidence bin should generally have higher mean_hit_prob.
    """
    rows = []

    # Previous results suggested conf_global is the most informative one.
    confidence_col = "conf_global"

    split_list = sorted(per_sample["split"].unique().tolist()) + ["all"]

    for split in split_list:
        if split == "all":
            g = per_sample.copy()
        else:
            g = per_sample[per_sample["split"] == split].copy()

        if len(g) < 10:
            continue

        g["conf_bin"] = pd.qcut(
            g[confidence_col],
            q=10,
            labels=False,
            duplicates="drop",
        )

        for tol in config["main_hit_tolerances"]:
            hit_col = f"mdn_hit_prob@{tol:g}"

            for bin_id, gb in g.groupby("conf_bin"):
                rows.append({
                    "split": split,
                    "tolerance": tol,
                    "confidence": confidence_col,
                    "bin": int(bin_id),
                    "n": int(len(gb)),
                    "conf_mean": float(gb[confidence_col].mean()),
                    "conf_min": float(gb[confidence_col].min()),
                    "conf_max": float(gb[confidence_col].max()),
                    "mean_hit_prob": float(gb[hit_col].mean()),
                })

    return pd.DataFrame(rows)


def write_summary(out_path: str,
                  df: pd.DataFrame,
                  pca_info: Dict,
                  main_table: pd.DataFrame,
                  conf_table: pd.DataFrame,
                  config: Dict):
    lines = []

    lines.append("Clean MDN MAE + Hit Probability + Confidence Test")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Main table columns:")
    lines.append("  mae: MAE over all normalized targets [t + 14 physical parameters].")
    lines.append("  t_mae: MAE for normalized time t.")
    lines.append("  phys14_mae: MAE over 14 physical parameters, excluding time.")
    lines.append("  hit_prob@0.01 / hit_prob@0.02:")
    lines.append("      Deterministic models: point-prediction hit rate.")
    lines.append("      MDN: sampling hit probability from p(y|x).")
    lines.append("  hit_auc: normalized AUC of hit probability over configured tolerances.")
    lines.append("")

    lines.append("Hit definition:")
    lines.append("  distance = sqrt(mean_j((pred_j - true_j)^2))")
    lines.append("  hit = distance <= tolerance")
    lines.append("")

    lines.append("Dataset")
    lines.append("-" * 72)
    lines.append(f"n_samples: {len(df)}")
    lines.append(f"n_did: {df['did'].nunique()}")
    for split, g in df.groupby("split"):
        lines.append(f"{split}: n={len(g)}, n_did={g['did'].nunique()}")
    lines.append("")

    lines.append("PCA info")
    lines.append("-" * 72)
    lines.append(json.dumps(pca_info, indent=2, ensure_ascii=False))
    lines.append("")

    lines.append("Main comparison, split = all")
    lines.append("-" * 72)
    lines.append(main_table[main_table["split"] == "all"].to_string(index=False))
    lines.append("")

    lines.append("Confidence vs hit, split = all")
    lines.append("-" * 72)
    lines.append(conf_table[conf_table["split"] == "all"].to_string(index=False))
    lines.append("")

    lines.append("Output files")
    lines.append("-" * 72)
    lines.append("  main_mae_hit_table.csv")
    lines.append("  confidence_vs_hit_spearman.csv")
    lines.append("  confidence_bins_hit.csv")
    lines.append("  per_sample_check.csv")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 6. Main
# ============================================================

def main():
    config = CONFIG

    set_seed(config["seed"])

    out_dir = ensure_dir(config["out_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[INFO] device = {device}")
    print(f"[INFO] output = {out_dir}")

    # ----------------------------
    # Load dataset
    # ----------------------------
    print("[INFO] Loading dataset...")
    loaders = load_dataloaders(config)
    df, x_inputs, images, y_true = collect_data(loaders)

    df.to_csv(os.path.join(out_dir, "base_samples_all_splits.csv"), index=False)

    print(f"[INFO] samples = {len(df)}")
    print(f"[INFO] did = {df['did'].nunique()}")

    # ----------------------------
    # Load models
    # ----------------------------
    print("[INFO] Loading models...")
    mdn_model = safe_torch_load(config["mdn_model_path"], device).to(device)
    mdn_model.eval()

    linear_model = safe_torch_load(config["linear_model_path"], device).to(device)
    linear_model.eval()

    # ----------------------------
    # Predict / sample
    # ----------------------------
    print("[INFO] PCA_Ridge...")
    pred_pca, pca_info = predict_pca_ridge(
        images=images,
        y_true=y_true,
        df=df,
        config=config,
    )

    print("[INFO] Linear_Joint...")
    pred_linear = predict_linear(
        model=linear_model,
        x_inputs=x_inputs,
        config=config,
        device=device,
    )

    print("[INFO] MDN sampling...")
    mdn_result = evaluate_mdn(
        model=mdn_model,
        x_inputs=x_inputs,
        y_true=y_true,
        config=config,
        device=device,
    )

    # ----------------------------
    # Tables
    # ----------------------------
    print("[INFO] Building tables...")
    main_table = build_main_table(
        df=df,
        y_true=y_true,
        pred_pca=pred_pca,
        pred_linear=pred_linear,
        mdn_result=mdn_result,
        config=config,
    )

    per_sample = build_per_sample_table(
        df=df,
        y_true=y_true,
        pred_pca=pred_pca,
        pred_linear=pred_linear,
        mdn_result=mdn_result,
        config=config,
    )

    conf_table = build_confidence_vs_hit_table(
        per_sample=per_sample,
        config=config,
    )

    conf_bins = build_confidence_bins(
        per_sample=per_sample,
        config=config,
    )

    # ----------------------------
    # Save files
    # ----------------------------
    main_table.to_csv(os.path.join(out_dir, "main_mae_hit_table.csv"), index=False)
    per_sample.to_csv(os.path.join(out_dir, "per_sample_check.csv"), index=False)
    conf_table.to_csv(os.path.join(out_dir, "confidence_vs_hit_spearman.csv"), index=False)
    conf_bins.to_csv(os.path.join(out_dir, "confidence_bins_hit.csv"), index=False)

    np.save(os.path.join(out_dir, "pred_pca.npy"), pred_pca)
    np.save(os.path.join(out_dir, "pred_linear.npy"), pred_linear)
    np.save(os.path.join(out_dir, "pred_mdn_mixture_mean.npy"), mdn_result["mixture_mean_pred"])

    write_summary(
        out_path=os.path.join(out_dir, "summary.txt"),
        df=df,
        pca_info=pca_info,
        main_table=main_table,
        conf_table=conf_table,
        config=config,
    )

    print("")
    print("[DONE] Clean MDN MAE + hit + confidence test finished.")
    print(f"[DONE] Results saved to: {out_dir}")
    print("")

    print("[MAIN TABLE: all split]")
    print(main_table[main_table["split"] == "all"].to_string(index=False))
    print("")

    print("[CONFIDENCE VS HIT: all split]")
    print(conf_table[conf_table["split"] == "all"].to_string(index=False))
    print("")

    print("Main files:")
    print("  main_mae_hit_table.csv")
    print("  confidence_vs_hit_spearman.csv")
    print("  confidence_bins_hit.csv")
    print("  per_sample_check.csv")
    print("  summary.txt")


if __name__ == "__main__":
    main()