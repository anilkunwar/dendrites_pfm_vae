# train_vae_mdn.py
import os
import math
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import albumentations as A

from src.dataloader import DendritePFMDataset
from src.modelv11 import VAE_MDN, mdn_point_and_confidence

import json

def save_run_args(args, save_root: str):
    """Save CLI args for reproducibility."""
    os.makedirs(save_root, exist_ok=True)
    args_path = os.path.join(save_root, "run_args.json")
    # argparse Namespace isn't directly JSON serializable if it contains tuples; convert carefully.
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def save_image_grid(tensor, path, nrow=3):
    grid = vutils.make_grid(
        tensor,
        nrow=nrow,
        normalize=True,
        scale_each=True
    )
    vutils.save_image(grid, path)

def plot_all_metrics_separately(
    df_train,
    df_val=None,
    save_root="plots",
    xcol="epoch",
    drop_top=0.05,      # 删除最大的 5%
):
    """
    Plot every numeric column (except xcol) as a separate figure.
    Train / Val are filtered independently.
    """

    os.makedirs(save_root, exist_ok=True)

    def _filter_top(x, y, drop_top):
        x = np.asarray(x)
        y = np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(y) == 0:
            return x, y
        thr = np.percentile(y, 100 * (1 - drop_top))
        keep = y <= thr
        return x[keep], y[keep]

    # 选出所有数值列
    numeric_cols = [
        c for c in df_train.columns
        if c != xcol and pd.api.types.is_numeric_dtype(df_train[c])
    ]

    for col in numeric_cols:
        fig = plt.figure()
        ax = plt.gca()

        # -------- train --------
        x_t = df_train[xcol].values
        y_t = df_train[col].values
        x_t, y_t = _filter_top(x_t, y_t, drop_top)
        if len(y_t) > 0:
            ax.plot(x_t, y_t, label="train", alpha=0.7)

        # -------- val --------
        if df_val is not None and col in df_val.columns:
            x_v = df_val[xcol].values
            y_v = df_val[col].values
            x_v, y_v = _filter_top(x_v, y_v, drop_top)
            if len(y_v) > 0:
                ax.plot(x_v, y_v, label="val", alpha=0.7)

        ax.set_xlabel(xcol)
        ax.set_ylabel(col)
        ax.set_title(f"{col} (drop top {int(drop_top*100)}%)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(save_root, f"{col}.png"),
            dpi=300
        )
        plt.close(fig)

def multiscale_recon_loss(x_pred, x_true, num_scales=4, scale_weight=0.5):
    total, weight = 0.0, 0.0
    cur_p, cur_t = x_pred, x_true
    w = 1.0
    for _ in range(num_scales):
        # per-element MSE: mean over (C,H,W) and batch
        total += w * F.l1_loss(cur_p, cur_t, reduction="mean")
        weight += w
        cur_p = F.avg_pool2d(cur_p, 2)
        cur_t = F.avg_pool2d(cur_t, 2)
        w *= scale_weight
    return total / weight


def kl_div_loss(mu_q, logvar_q, prior_var=0.1):
    var_q = torch.exp(logvar_q)
    kl_per_sample = 0.5 * torch.sum(
        var_q / prior_var
        + mu_q ** 2 / prior_var
        - 1.0
        + math.log(prior_var)
        - logvar_q,
        dim=1
    )  # [B]
    return torch.mean(kl_per_sample) / mu_q.shape[1]

def mdn_nll_loss(pi, mu, log_sigma, y, eps: float = 1e-9):
    """
    y: [B, P]
    pi: [B, K]
    mu/log_sigma: [B, K, P]
    return: scalar (mean over batch)
    """
    B, K, P = mu.shape
    y = y.unsqueeze(1).expand(B, K, P)  # [B, K, P]

    # log N(y | mu, sigma^2) for diagonal Gaussian
    # = -0.5 * [sum((y-mu)^2/sigma^2 + 2log(sigma) + log(2pi))]
    sigma = torch.exp(log_sigma) + eps
    log_prob = -0.5 * (
        torch.sum(((y - mu) / sigma) ** 2, dim=-1)
        + 2.0 * torch.sum(torch.log(sigma), dim=-1)
        + P * math.log(2.0 * math.pi)
    )  # [B, K]

    log_pi = torch.log(pi + eps)  # [B, K]
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # [B]
    nll = -torch.mean(log_mix)
    return nll

def sharp_interface_loss(img, alpha=1.0, beta=10.0, eps=1e-6):
    """
    Per-sample min/max normalization with eps in denominator,
    plus a simple penalty if a sample is (almost) flat.
    img: (B,H,W)
    flat_thr: treat (max-min) <= flat_thr as invalid/flat
    """
    img_min = img.amin(dim=(-2, -1), keepdim=True)  # (B,1,1)
    img_max = img.amax(dim=(-2, -1), keepdim=True)  # (B,1,1)
    denom = img_max - img_min                       # (B,1,1)

    # per-sample normalize; denom+eps avoids div-by-zero
    m = (img - img_min) / (denom + eps)

    # interface term
    dx = m[..., 1:] - m[..., :-1]
    dy = m[..., 1:, :] - m[..., :-1, :]
    interface = (dx * dx).mean() + (dy * dy).mean()

    # two-phase term
    two_phase = (m * m * (1 - m) * (1 - m)).mean()

    loss = alpha * interface + beta * two_phase
    return loss, {
        "interface": interface.item(),
        "two_phase": two_phase.item()
    }

def beta_warmup(epoch, beta_max, warmup_epochs):
    if epoch >= warmup_epochs:
        return beta_max
    return beta_max * 0.5 * (1.0 - math.cos(math.pi * epoch / warmup_epochs))

# ==========================================================
# Main
# ==========================================================
def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------
    # Experiment name
    # --------------------------
    exp_name = (
        f"VAEv12_MDN_"
        # f"lat={args.latent_size}_var_scale={args.var_scale}"
        # f"K={args.mdn_components}_"
        # f"beta={args.beta}_warm={args.beta_warmup_ratio}_"
        # f"gamma={args.gamma}_warm={args.gamma_warmup_ratio}_"
        # f"phy_weight={args.phy_weight}_phy_alpha={args.phy_alpha}_phy_beta={args.phy_beta}_"
        # f"scale_weight={args.scale_weight}_"
        f"time={datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"Experiment name: {exp_name}")
    save_root = os.path.join(args.save_root, exp_name)
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, "ckpt"), exist_ok=True)

    # Save CLI args
    save_run_args(args, save_root)

    # --------------------------
    # Dataset
    # --------------------------
    train_ds = DendritePFMDataset(
        args.image_size,
        os.path.join("data", "dataset_split.json"),
        split="train",
        transform=A.Compose([
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.01, 0.1),
                hole_width_range=(0.01, 0.1),
                p=0.1
            ),
            A.GaussNoise(p=0.8),
        ])
    )
    val_ds = DendritePFMDataset(
        args.image_size,
        os.path.join("data", "dataset_split.json"),
        split="val"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --------------------------
    # Fixed visualization batch
    # --------------------------
    vis_n = 9
    vis_x, _, _, vis_xo = next(iter(val_loader))
    vis_x = vis_x[:vis_n].to(device)
    vis_xo = vis_xo[:vis_n].to(device)
    fixed_z = torch.randn(vis_n, args.latent_size, device=device)

    # --------------------------
    # Model
    # --------------------------
    # model = VAE_MDN(
    #     image_size=args.image_size,
    #     latent_size=args.latent_size,
    #     hidden_dimension=args.hidden_dim,
    #     num_params=args.num_params,
    #     mdn_components=args.mdn_components,
    #     mdn_hidden=args.mdn_hidden,
    # ).to(device)
    if args.init_model_path:
        model = torch.load(args.init_model_path, weights_only=False)
    else:
        model = VAE_MDN(
            image_size=args.image_size,
            latent_size=args.latent_size,
            hidden_dimension=args.hidden_dim,
            num_params=args.num_params,
            mdn_components=args.mdn_components,
            mdn_hidden=args.mdn_hidden,
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)

    beta_warup_epochs = int(args.epochs * args.beta_warmup_ratio)
    gamma_warmup_epochs = int(args.epochs * args.gamma_warmup_ratio)

    train_logs, val_logs = [], []
    best_val, no_imp = float("inf"), 0

    # 用于把“方差 -> 置信度”的尺度做个稳定参考
    var_scale = args.var_scale

    # ======================================================
    # Training loop
    # ======================================================
    for epoch in range(1, args.epochs+1):

        beta_t = beta_warmup(epoch, args.beta, beta_warup_epochs)
        gamma_t = beta_warmup(epoch, args.gamma, gamma_warmup_epochs)

        # ---------------------- Train ----------------------
        model.train()
        tstat = defaultdict(list)

        for x, y, _, xo in train_loader:
            x, xo, y = x.to(device), xo.to(device), y.to(device)

            recon, mu_q, logvar_q, mdn_out, z = model(x)
            pi, mu, log_sigma = mdn_out

            recon_loss = multiscale_recon_loss(recon, xo, scale_weight=args.scale_weight)
            kl_loss = kl_div_loss(mu_q, logvar_q, prior_var=var_scale)
            ctr_nll = mdn_nll_loss(pi, mu, log_sigma, y)

            with torch.no_grad():
                theta_hat, conf_param, conf_global, _ = mdn_point_and_confidence(
                    pi, mu, log_sigma, var_scale=var_scale, topk=3
                )
                ctr_mse_monitor = F.mse_loss(theta_hat, y)

            # prior smoothness
            bsz = x.size(0)
            z_prior = torch.randn(bsz, args.latent_size, device=device) * math.sqrt(var_scale)
            prior_img = model.decoder(z_prior)
            sm_loss, sm_info = sharp_interface_loss(prior_img[:, 0], args.phy_alpha, args.phy_beta)

            total = (
                recon_loss
                + beta_t * kl_loss
                + gamma_t * ctr_nll
                + args.phy_weight * sm_loss
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            tstat["total"].append(total.item())
            tstat["recon"].append(recon_loss.item())
            tstat["kl"].append(kl_loss.item())
            tstat["ctr_nll"].append(ctr_nll.item())
            tstat["ctr_mse_monitor"].append(ctr_mse_monitor.item())
            tstat["conf_global_mean"].append(conf_global.mean().item())
            tstat["phy"].append(sm_loss.item())
            tstat["interface"].append(sm_info["interface"])
            tstat["two_phase"].append(sm_info["two_phase"])

        train_epoch = {k: float(np.mean(v)) for k, v in tstat.items()}
        train_epoch["epoch"] = epoch
        train_epoch["beta"] = beta_t
        train_epoch["gamma"] = gamma_t
        train_logs.append(train_epoch)

        # -------------------- Validation --------------------
        model.eval()
        vstat = defaultdict(list)

        with torch.no_grad():
            for x, y, _, xo in val_loader:
                x, xo, y = x.to(device), xo.to(device), y.to(device)

                recon, mu_q, logvar_q, mdn_out, z = model(x)
                pi, mu, log_sigma = mdn_out

                recon_loss = multiscale_recon_loss(recon, xo, scale_weight=args.scale_weight)
                kl_loss = kl_div_loss(mu_q, logvar_q, prior_var=var_scale)
                ctr_nll = mdn_nll_loss(pi, mu, log_sigma, y)

                theta_hat, conf_param, conf_global, _ = mdn_point_and_confidence(
                    pi, mu, log_sigma, var_scale=var_scale, topk=3
                )
                ctr_mse_monitor = F.mse_loss(theta_hat, y)

                bsz = x.size(0)
                z_prior = torch.randn(bsz, args.latent_size, device=device) * math.sqrt(var_scale)
                prior_img = model.decoder(z_prior)
                sm_loss, sm_info = sharp_interface_loss(prior_img[:, 0], args.phy_alpha, args.phy_beta)

                total = (
                    recon_loss
                    + beta_t * kl_loss
                    + gamma_t * ctr_nll
                    + args.phy_weight * sm_loss
                )

                vstat["total"].append(total.item())
                vstat["recon"].append(recon_loss.item())
                vstat["kl"].append(kl_loss.item())
                vstat["ctr_nll"].append(ctr_nll.item())
                vstat["ctr_mse_monitor"].append(ctr_mse_monitor.item())
                vstat["conf_global_mean"].append(conf_global.mean().item())
                vstat["phy"].append(sm_loss.item())
                vstat["interface"].append(sm_info["interface"])
                vstat["two_phase"].append(sm_info["two_phase"])

        lr_scheduler.step(np.mean(vstat["total"]))

        val_epoch = {k: float(np.mean(v)) for k, v in vstat.items()}
        val_epoch["epoch"] = epoch
        val_epoch["beta"] = beta_t
        val_epoch["gamma"] = gamma_t
        val_logs.append(val_epoch)

        print(
            f"[Epoch {epoch:04d}] beta={beta_t:.3f} gamma={gamma_t:.3f} | "
            f"Train total={train_epoch['total']:.4f} | "
            f"Val total={val_epoch['total']:.4f} | "
            f"Val ctr_nll={val_epoch['ctr_nll']:.4f} | "
            f"Val ctr_mse={val_epoch['ctr_mse_monitor']:.6f} | "
            f"Val conf={val_epoch['conf_global_mean']:.3f}"
        )

        # -------------------- Save images --------------------
        epoch_dir = os.path.join(save_root, f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)
        with torch.no_grad():
            # recon
            recon_vis, *_ = model(vis_x)
            save_image_grid(recon_vis, os.path.join(epoch_dir, "recon.png"))
            # prior
            save_image_grid(model.decoder(fixed_z), os.path.join(epoch_dir, "prior.png"))
            # input/target
            save_image_grid(vis_x, os.path.join(epoch_dir, "input.png"))
            save_image_grid(vis_xo, os.path.join(epoch_dir, "target.png"))

        # -------------------- Save logs --------------------
        df_train = pd.DataFrame(train_logs)
        df_val = pd.DataFrame(val_logs)
        df_train.to_csv(os.path.join(save_root, "train_epoch.csv"), index=False)
        df_val.to_csv(os.path.join(save_root, "val_epoch.csv"), index=False)

        plot_all_metrics_separately(df_train, df_val, save_root=save_root)

        # -------------------- Early stopping --------------------
        if val_epoch["total"] < best_val:
            best_val = val_epoch["total"]
            no_imp = 0
            torch.save(model, os.path.join(save_root, "ckpt", "best.pt"))
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print("🛑 Early stopping")
                break


# ==========================================================
# Args
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_factor", type=float, default=0.75)
    parser.add_argument("--lr_patience", type=float, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--image_size", type=tuple, default=(3, 48, 48))
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_params", type=int, default=15)

    # MDN
    parser.add_argument("--mdn_components", type=int, default=16)
    parser.add_argument("--mdn_hidden", type=int, default=256)

    # VAE losses
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--beta_warmup_ratio", type=float, default=0.1)

    # weights
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--gamma_warmup_ratio", type=float, default=0.1)

    parser.add_argument("--phy_weight", type=float, default=0.01) # 设置为0可以实现重建效果
    parser.add_argument("--phy_alpha", type=float, default=1)
    parser.add_argument("--phy_beta", type=float, default=1)

    parser.add_argument("--scale_weight", type=float, default=0.1)

    # confidence scaling
    parser.add_argument("--var_scale", type=float, default=0.1)

    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--save_root", type=str, default="results")

    parser.add_argument("--init_model_path", type=str,
                        default=None,
                        # default="/home/xtanghao/THPycharm/dendrites_pfm_vae/tmp/oldv12/VAEv12_MDN_lat=16_var_scale=0.1K=16_beta=0.01_warm=0.1_gamma=0.001_warm=0.1_phy_weight=0.0_phy_alpha=1_phy_beta=1_scale_weight=0.1_time=20260124_055835/ckpt/best.pt",
                        help="Optional path to a saved .pt model to initialize (torch.load). If not set, build a fresh VAE_MDN.")

    args = parser.parse_args()
    main(args)
