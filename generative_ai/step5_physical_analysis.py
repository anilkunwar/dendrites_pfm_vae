# -*- coding: utf-8 -*-
"""
Single-figure latent traversal for five descriptors:

    1. time_norm                       -> t
    2. POT_LEFT                        -> U
    3. Noise                           -> psi
    4. electrode free-energy density   -> f^e
    5. electrolyte free-energy density -> f^s

Run inside generative_ai/:
    python test.py

Output:
    t_U_psi_fe_fs_traversal.png

Only one image is saved.

Method:
    For each descriptor target y_k:
        1. Fit linear probe:
               y_k = w_k^T z + b_k
        2. Traverse:
               z(alpha) = z0 + alpha * w_k / ||w_k||
        3. Decode each z(alpha)

The bottom labels are normalized target values:
    0.00, 0.20, 0.40, 0.60, 0.80, 1.00

Free-energy density definitions:

    Electrode / solid phase:
        f^e = fo * [ As * (c - cseq)^2 + Bs * (c - cseq) + Cs ] / Vm

    Electrolyte / solution phase:
        f^s = fo * [ Al * (c - cleq)^2 + Bl * (c - cleq) + Cl ] / Vm

For each sample, each free-energy density target is converted to a scalar by:
        mean(abs(f))

and then min-max normalized to [0, 1].
"""

import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from src.dataloader import DendritePFMDataset, PARAM_RANGES


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # Dataset split file, relative to generative_ai/
    "SPLIT_JSON": os.path.join("data", "dataset_split.json"),

    # Trained model checkpoint
    "CKPT_PATH": os.path.join(
        "results",
        "final_model",
        "ckpt",
        "best.pt",
    ),

    # Use all splits for stable latent directions
    "SPLITS_TO_USE": ["train", "val", "test"],

    # Must match training
    "IMAGE_SIZE": (3, 48, 48),
    "BATCH_SIZE": 256,
    "NUM_WORKERS": 2,

    # Output image
    "OUT_PATH": "t_U_psi_fe_fs_traversal.png",

    # Image channel to visualize
    # Usually:
    #   channel 0 = phase/order parameter
    #   channel 1 = concentration
    #   channel 2 = potential
    "PLOT_CHANNEL": 0,

    # Concentration channel used for free-energy density
    "CONCENTRATION_CHANNEL": 1,

    # Normalized target values shown at the bottom
    "TARGET_VALUES": [0.00, 0.20, 0.40, 0.60, 0.80, 1.00],

    # Linear probe
    "RIDGE_ALPHA": 1.0,

    # Alpha clipping / warning
    "HARD_CLIP_ALPHA": False,
    "MAX_ABS_ALPHA_CLIP": 8.0,
    "MAX_ABS_ALPHA_WARNING": 8.0,

    # Center point:
    #   "global_center": same representative latent point for all rows
    #   "median_target": each row chooses sample nearest median target
    #   "fixed_index": use FIXED_CENTER_INDEX
    "CENTER_STRATEGY": "global_center",
    "FIXED_CENTER_INDEX": 0,

    # Free-energy constant
    "VM": 1.0,

    # Concentration handling:
    #   "clip_0_1": clip concentration to [0,1]
    #   "decoded": use concentration channel directly
    "CONCENTRATION_MODE": "clip_0_1",

    # Use actual dataset field or reconstructed field to compute free-energy targets.
    # "xo" is recommended because it uses the actual dataset fields.
    "ENERGY_TARGET_FIELD": "xo",  # "xo" or "recon"

    # Figure appearance
    "DPI": 500,
    "CMAP": "coolwarm",

    # Color scale:
    #   "fixed_0_1": same scale for all panels
    #   "per_row_percentile": better contrast within each row
    "COLOR_SCALE": "per_row_percentile",
    "PERCENTILE_LOW": 1,
    "PERCENTILE_HIGH": 99,

    # Layout
    "PANEL_SIZE": 1.72,
    "LEFT_LABEL_WIDTH": 1.12,
    "BOTTOM_LABEL_HEIGHT": 0.62,

    # Font sizes
    "ROW_LABEL_FONT_SIZE": 34,
    "BOTTOM_LABEL_FONT_SIZE": 24,

    # Row labels
    "ROW_LABELS": {
        "time_norm": r"$t$",
        "POT_LEFT": r"$U$",
        "Noise": r"$\psi$",
        "electrode_free_energy_density": r"$f^{e}$",
        "electrolyte_free_energy_density": r"$f^{s}$",
    },

    # Bottom labels are normalized target values
    "BOTTOM_LABEL_PREFIX": "",

    # Random seed
    "SEED": 0,
}


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_param_names():
    """
    Dataset returns:
        y = [time_norm] + list(PARAM_RANGES.keys())
    """
    return ["time_norm"] + list(PARAM_RANGES.keys())


def param_index(param_names, name):
    if name not in param_names:
        raise KeyError(f"Parameter {name} not found. param_names={param_names}")
    return param_names.index(name)


def inv_scale_param_value(param_name: str, norm_value: float):
    """
    Convert normalized parameter value back to its physical value.
    time_norm is kept unchanged.
    """
    norm_value = float(norm_value)

    if param_name == "time_norm":
        return norm_value

    if param_name not in PARAM_RANGES:
        return norm_value

    lo, hi = PARAM_RANGES[param_name]
    return norm_value * (hi - lo) + lo


def inv_scale_matrix(Y_norm, param_names):
    """
    Convert normalized parameter matrix to physical parameter matrix.
    """
    Y_norm = np.asarray(Y_norm, dtype=np.float64)
    Y_real = np.zeros_like(Y_norm, dtype=np.float64)

    for j, name in enumerate(param_names):
        Y_real[:, j] = [inv_scale_param_value(name, v) for v in Y_norm[:, j]]

    return Y_real


def minmax01(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))

    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float64)

    return (x - lo) / (hi - lo)


def load_model(device):
    ckpt_path = CONFIG["CKPT_PATH"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{ckpt_path}\n\n"
            "Please edit CONFIG['CKPT_PATH']."
        )

    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def build_loader(split: str):
    ds = DendritePFMDataset(
        CONFIG["IMAGE_SIZE"],
        CONFIG["SPLIT_JSON"],
        split=split,
        transform=None,
    )

    loader = DataLoader(
        ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"],
    )

    return ds, loader


@torch.no_grad()
def collect_latents_and_fields(model, splits):
    """
    Dataset returns:
        x, y, case_id, xo

    x:
        model input

    y:
        [time_norm] + normalized physical parameters

    xo:
        original smooth-scaled tensor from dataset
    """
    device = next(model.parameters()).device

    Z_list = []
    Y_list = []
    XO_list = []
    RECON_list = []

    for split in splits:
        _, loader = build_loader(split)

        for x, y, case_id, xo in loader:
            x = x.to(device)

            recon, mu_q, logvar_q, mdn_out, z_sample = model(x)

            Z_list.append(mu_q.detach().cpu().numpy())
            Y_list.append(y.detach().cpu().numpy())
            XO_list.append(xo.detach().cpu().numpy())
            RECON_list.append(recon.detach().cpu().numpy())

    Z = np.vstack(Z_list).astype(np.float32)
    Y = np.vstack(Y_list).astype(np.float32)
    XO = np.vstack(XO_list).astype(np.float32)
    RECON = np.vstack(RECON_list).astype(np.float32)

    return Z, Y, XO, RECON


# ============================================================
# Free-energy density targets
# ============================================================

def get_concentration(field):
    """
    field: [N, C, H, W]
    """
    c = field[:, int(CONFIG["CONCENTRATION_CHANNEL"])].astype(np.float64)

    mode = CONFIG.get("CONCENTRATION_MODE", "clip_0_1")

    if mode == "clip_0_1":
        c = np.clip(c, 0.0, 1.0)
    elif mode == "decoded":
        pass
    else:
        raise ValueError(f"Unknown CONCENTRATION_MODE: {mode}")

    return c


def compute_free_energy_density_targets(field, Y_norm, param_names):
    """
    Compute two scalar free-energy density descriptors:

        electrode_free_energy_density:
            f^e = fo * [ As * (c - cseq)^2 + Bs * (c - cseq) + Cs ] / Vm

        electrolyte_free_energy_density:
            f^s = fo * [ Al * (c - cleq)^2 + Bl * (c - cleq) + Cl ] / Vm

    Then:
        E = mean(abs(f))
        E_norm = minmax(E)

    Returns:
        electrode_energy_norm, electrolyte_energy_norm
    """
    Y_real = inv_scale_matrix(Y_norm, param_names)

    idx_fo = param_index(param_names, "fo")

    idx_Al = param_index(param_names, "Al")
    idx_Bl = param_index(param_names, "Bl")
    idx_Cl = param_index(param_names, "Cl")
    idx_cleq = param_index(param_names, "cleq")

    idx_As = param_index(param_names, "As")
    idx_Bs = param_index(param_names, "Bs")
    idx_Cs = param_index(param_names, "Cs")
    idx_cseq = param_index(param_names, "cseq")

    fo = Y_real[:, idx_fo]

    Al = Y_real[:, idx_Al]
    Bl = Y_real[:, idx_Bl]
    Cl = Y_real[:, idx_Cl]
    cleq = Y_real[:, idx_cleq]

    As = Y_real[:, idx_As]
    Bs = Y_real[:, idx_Bs]
    Cs = Y_real[:, idx_Cs]
    cseq = Y_real[:, idx_cseq]

    c = get_concentration(field)

    Vm = float(CONFIG.get("VM", 1.0))
    eps = 1e-12

    # Electrode / solid phase free-energy density: f^e
    f_e = fo[:, None, None] * (
        As[:, None, None] * (c - cseq[:, None, None]) ** 2
        + Bs[:, None, None] * (c - cseq[:, None, None])
        + Cs[:, None, None]
    ) / max(Vm, eps)

    # Electrolyte / solution phase free-energy density: f^s
    f_s = fo[:, None, None] * (
        Al[:, None, None] * (c - cleq[:, None, None]) ** 2
        + Bl[:, None, None] * (c - cleq[:, None, None])
        + Cl[:, None, None]
    ) / max(Vm, eps)

    electrode_energy_raw = np.mean(np.abs(f_e), axis=(1, 2))
    electrolyte_energy_raw = np.mean(np.abs(f_s), axis=(1, 2))

    electrode_energy_norm = minmax01(electrode_energy_raw).astype(np.float32)
    electrolyte_energy_norm = minmax01(electrolyte_energy_raw).astype(np.float32)

    return electrode_energy_norm, electrolyte_energy_norm


def build_descriptor_targets(Y, field, param_names):
    """
    Five descriptors in required order:
        time_norm                       -> t
        POT_LEFT                        -> U
        Noise                           -> psi
        electrode_free_energy_density   -> f^e
        electrolyte_free_energy_density -> f^s
    """
    targets = {}

    time_idx = param_index(param_names, "time_norm")
    pot_idx = param_index(param_names, "POT_LEFT")
    noise_idx = param_index(param_names, "Noise")

    targets["time_norm"] = Y[:, time_idx].astype(np.float32)
    targets["POT_LEFT"] = Y[:, pot_idx].astype(np.float32)
    targets["Noise"] = Y[:, noise_idx].astype(np.float32)

    f_e_norm, f_s_norm = compute_free_energy_density_targets(
        field=field,
        Y_norm=Y,
        param_names=param_names,
    )

    targets["electrode_free_energy_density"] = f_e_norm
    targets["electrolyte_free_energy_density"] = f_s_norm

    return targets


# ============================================================
# Traversal
# ============================================================

def choose_global_center_index(Z_scaled):
    center = np.mean(Z_scaled, axis=0, keepdims=True)
    dist = np.sum((Z_scaled - center) ** 2, axis=1)
    return int(np.argmin(dist))


def choose_center_index(Z_scaled, target):
    strategy = CONFIG.get("CENTER_STRATEGY", "global_center")

    if strategy == "global_center":
        return choose_global_center_index(Z_scaled)

    if strategy == "median_target":
        return int(np.argmin(np.abs(target - np.median(target))))

    if strategy == "fixed_index":
        idx = int(CONFIG.get("FIXED_CENTER_INDEX", 0))
        if idx < 0 or idx >= Z_scaled.shape[0]:
            raise ValueError(f"FIXED_CENTER_INDEX={idx} out of range.")
        return idx

    raise ValueError(f"Unknown CENTER_STRATEGY: {strategy}")


@torch.no_grad()
def traverse_one_descriptor(model, Z, target, target_name):
    """
    Fit:
        target = w^T z + b

    Traverse:
        z(alpha) = z0 + alpha * w / ||w||
    """
    device = next(model.parameters()).device

    z_scaler = StandardScaler().fit(Z)
    Zs = z_scaler.transform(Z)

    reg = Ridge(alpha=float(CONFIG.get("RIDGE_ALPHA", 1.0)))
    reg.fit(Zs, target.astype(np.float32))

    pred_all = reg.predict(Zs).astype(np.float32)

    probe_r2 = float(r2_score(target, pred_all))
    probe_mae = float(mean_absolute_error(target, pred_all))

    w = reg.coef_.astype(np.float32)
    w_norm = float(np.linalg.norm(w))

    if w_norm < 1e-12:
        raise RuntimeError(f"Near-zero Ridge direction for {target_name}")

    direction = w / w_norm

    center_idx = choose_center_index(Zs, target)
    z0 = Zs[center_idx]

    pred_center = float(reg.predict(z0[None, :])[0])

    target_values = np.asarray(
        CONFIG.get("TARGET_VALUES", [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        dtype=np.float32,
    )

    alphas = (target_values - pred_center) / w_norm

    if CONFIG.get("HARD_CLIP_ALPHA", False):
        clip_value = float(CONFIG.get("MAX_ABS_ALPHA_CLIP", 8.0))
        alphas = np.clip(alphas, -clip_value, clip_value)
        target_values = pred_center + alphas * w_norm

    Z_path_scaled = np.stack(
        [z0 + float(alpha) * direction for alpha in alphas],
        axis=0,
    ).astype(np.float32)

    pred_values = reg.predict(Z_path_scaled).astype(np.float32)

    Z_path = z_scaler.inverse_transform(Z_path_scaled).astype(np.float32)

    z_tensor = torch.tensor(Z_path, dtype=torch.float32, device=device)
    decoded = model.decoder(z_tensor).detach().cpu().numpy().astype(np.float32)

    max_abs_alpha = float(np.max(np.abs(alphas)))

    print(
        f"{target_name:36s} | "
        f"R2={probe_r2:.4f}, MAE={probe_mae:.4f}, "
        f"||w||={w_norm:.4f}, max|alpha|={max_abs_alpha:.4f}"
    )

    if max_abs_alpha > float(CONFIG.get("MAX_ABS_ALPHA_WARNING", 8.0)):
        warnings.warn(
            f"{target_name}: large max |alpha| = {max_abs_alpha:.3f}. "
            "This row may include stronger latent extrapolation."
        )

    return {
        "target_name": target_name,
        "decoded": decoded,
        "target_values": target_values,
        "pred_values": pred_values,
        "alphas": alphas.astype(np.float32),
        "probe_r2": probe_r2,
        "probe_mae": probe_mae,
        "ridge_coef_norm": w_norm,
    }


# ============================================================
# Plot single figure
# ============================================================

def get_row_vmin_vmax(decoded):
    channel = int(CONFIG.get("PLOT_CHANNEL", 0))

    if CONFIG.get("COLOR_SCALE", "per_row_percentile") == "fixed_0_1":
        return 0.0, 1.0

    vals = decoded[:, channel].reshape(-1)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0.0, 1.0

    if CONFIG.get("COLOR_SCALE", "per_row_percentile") == "per_row_percentile":
        vmin = float(np.percentile(vals, CONFIG.get("PERCENTILE_LOW", 1)))
        vmax = float(np.percentile(vals, CONFIG.get("PERCENTILE_HIGH", 99)))

        if vmax <= vmin:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))

        if vmax <= vmin:
            vmax = vmin + 1e-6

        return vmin, vmax

    raise ValueError(f"Unknown COLOR_SCALE: {CONFIG.get('COLOR_SCALE')}")


def get_bottom_display_value(value):
    """
    Bottom labels are shared normalized target values.
    """
    return f"{value:.2f}"


def plot_single_traversal_figure(results, out_path):
    # f^e and f^s are placed as the last two rows.
    row_order = [
        "time_norm",
        "Noise",
        "POT_LEFT",
        "electrode_free_energy_density",
        "electrolyte_free_energy_density",
    ]

    result_map = {r["target_name"]: r for r in results}
    ordered_results = [result_map[name] for name in row_order if name in result_map]

    if len(ordered_results) == 0:
        raise RuntimeError("No traversal results to plot.")

    n_rows = len(ordered_results)
    n_cols = len(CONFIG.get("TARGET_VALUES", [0, 0.2, 0.4, 0.6, 0.8, 1.0]))

    panel = float(CONFIG.get("PANEL_SIZE", 1.72))
    left_w = float(CONFIG.get("LEFT_LABEL_WIDTH", 1.12))
    bottom_h = float(CONFIG.get("BOTTOM_LABEL_HEIGHT", 0.62))

    fig_w = left_w + n_cols * panel
    fig_h = n_rows * panel + bottom_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(CONFIG.get("DPI", 500)))

    gs = GridSpec(
        nrows=n_rows + 1,
        ncols=n_cols + 1,
        figure=fig,
        width_ratios=[left_w] + [panel] * n_cols,
        height_ratios=[panel] * n_rows + [bottom_h],
        wspace=0.018,
        hspace=0.018,
    )

    ax_empty = fig.add_subplot(gs[-1, 0])
    ax_empty.set_axis_off()

    channel = int(CONFIG.get("PLOT_CHANNEL", 0))

    for r_idx, result in enumerate(ordered_results):
        target_name = result["target_name"]
        decoded = result["decoded"]

        vmin, vmax = get_row_vmin_vmax(decoded)

        # Left row label
        ax_label = fig.add_subplot(gs[r_idx, 0])
        ax_label.set_axis_off()

        label_text = CONFIG.get("ROW_LABELS", {}).get(target_name, target_name)

        ax_label.text(
            0.92,
            0.50,
            label_text,
            ha="right",
            va="center",
            fontsize=int(CONFIG.get("ROW_LABEL_FONT_SIZE", 34)),
            fontweight="bold",
            linespacing=0.9,
        )

        # Image panels
        for c_idx in range(n_cols):
            ax = fig.add_subplot(gs[r_idx, c_idx + 1])

            ax.imshow(
                decoded[c_idx, channel],
                cmap=CONFIG.get("CMAP", "coolwarm"),
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()

    # Bottom column labels
    target_values = np.asarray(
        CONFIG.get("TARGET_VALUES", [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        dtype=np.float32,
    )

    for c_idx in range(n_cols):
        ax = fig.add_subplot(gs[-1, c_idx + 1])
        ax.set_axis_off()

        value_text = get_bottom_display_value(float(target_values[c_idx]))

        if CONFIG.get("BOTTOM_LABEL_PREFIX", ""):
            value_text = f"{CONFIG.get('BOTTOM_LABEL_PREFIX')}{value_text}"

        ax.text(
            0.5,
            0.56,
            value_text,
            ha="center",
            va="center",
            fontsize=int(CONFIG.get("BOTTOM_LABEL_FONT_SIZE", 24)),
            fontweight="bold",
        )

    plt.subplots_adjust(
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
        wspace=0.018,
        hspace=0.018,
    )

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    set_seed(int(CONFIG.get("SEED", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("Single figure traversal: t, U, psi, f^e, f^s")
    print("=" * 88)
    print(f"Device:     {device}")
    print(f"Checkpoint: {CONFIG['CKPT_PATH']}")
    print(f"Output:     {CONFIG['OUT_PATH']}")
    print("=" * 88)

    model = load_model(device)

    Z, Y, XO, RECON = collect_latents_and_fields(
        model=model,
        splits=CONFIG.get("SPLITS_TO_USE", ["train", "val", "test"]),
    )

    param_names = get_param_names()

    if len(param_names) > Y.shape[1]:
        param_names = param_names[:Y.shape[1]]
    elif len(param_names) < Y.shape[1]:
        param_names = param_names + [
            f"theta_{i}" for i in range(len(param_names), Y.shape[1])
        ]

    print(f"Latent Z shape:  {Z.shape}")
    print(f"Control Y shape: {Y.shape}")
    print(f"XO field shape:  {XO.shape}")
    print("")

    if CONFIG.get("ENERGY_TARGET_FIELD", "xo") == "xo":
        energy_field = XO
    elif CONFIG.get("ENERGY_TARGET_FIELD", "xo") == "recon":
        energy_field = RECON
    else:
        raise ValueError(f"Unknown ENERGY_TARGET_FIELD: {CONFIG.get('ENERGY_TARGET_FIELD')}")

    targets = build_descriptor_targets(
        Y=Y,
        field=energy_field,
        param_names=param_names,
    )

    results = []

    for target_name in [
        "time_norm",
        "POT_LEFT",
        "Noise",
        "electrode_free_energy_density",
        "electrolyte_free_energy_density",
    ]:
        print(f"[Traversal] {target_name}")

        try:
            result = traverse_one_descriptor(
                model=model,
                Z=Z,
                target=targets[target_name],
                target_name=target_name,
            )
            results.append(result)

        except Exception as e:
            warnings.warn(f"Skip {target_name}: {e}")

    plot_single_traversal_figure(
        results=results,
        out_path=CONFIG.get("OUT_PATH", "t_U_psi_fe_fs_traversal.png"),
    )

    print("")
    print("=" * 88)
    print("Done.")
    print(f"Saved figure: {CONFIG.get('OUT_PATH', 't_U_psi_fe_fs_traversal.png')}")
    print("=" * 88)


if __name__ == "__main__":
    main()