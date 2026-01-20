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
from src.modelv11 import VAE_MDN, mdn_nll_loss, mdn_point_and_confidence


def save_image_grid(tensor, path, nrow=3):
    grid = vutils.make_grid(
        tensor,
        nrow=nrow,
        normalize=True,
        scale_each=True
    )
    vutils.save_image(grid, path)


def multiscale_recon_loss(x_pred, x_true, num_scales=4, scale_weight=0.5):
    total, weight = 0.0, 0.0
    cur_p, cur_t = x_pred, x_true
    w = 1.0
    for _ in range(num_scales):
        total += w * F.mse_loss(cur_p, cur_t, reduction="sum")
        weight += w
        cur_p = F.avg_pool2d(cur_p, 2)
        cur_t = F.avg_pool2d(cur_t, 2)
        w *= scale_weight
    return total / weight


def kl_div_loss(mu_q, logvar_q):
    return 0.5 * torch.sum(
        torch.exp(logvar_q) + mu_q ** 2 - 1.0 - logvar_q
    )


def total_variation(x):
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


def smoothness_loss(img, var_floor=0.02):
    tv = total_variation(img)
    var = img.var(dim=(1, 2, 3), unbiased=False).mean()
    var_pen = F.relu(var_floor - var) ** 2
    return tv + var_pen, {
        "tv": tv.item(),
        "var": var.item(),
        "var_pen": var_pen.item()
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
        f"VAEv11_MDN_"
        f"lat={args.latent_size}_"
        f"K={args.mdn_components}_"
        f"beta={args.beta}_warm={args.beta_warmup_ratio}_"
        f"ctr={args.ctr_weight}_smooth={args.smooth_weight}_"
        f"time={datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_root = os.path.join(args.save_root, exp_name)
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, "ckpt"), exist_ok=True)

    # --------------------------
    # Dataset
    # --------------------------
    train_ds = DendritePFMDataset(
        args.image_size,
        os.path.join("data", "dataset_split.json"),
        split="train",
        transform=A.Compose([A.GaussNoise(p=0.5)])
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
    model = VAE_MDN(
        image_size=args.image_size,
        latent_size=args.latent_size,
        hidden_dimension=args.hidden_dim,
        num_params=args.num_params,
        mdn_components=args.mdn_components,
        mdn_hidden=args.mdn_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    warmup_epochs = int(args.epochs * args.beta_warmup_ratio)

    train_logs, val_logs = [], []
    best_val, no_imp = float("inf"), 0

    # 用于把“方差 -> 置信度”的尺度做个稳定参考
    # 简单做法：固定为 1.0；更稳做法：用训练初期估个 var 中位数
    var_scale = args.var_scale

    # ======================================================
    # Training loop
    # ======================================================
    for epoch in range(args.epochs):

        beta_t = beta_warmup(epoch, args.beta, warmup_epochs)

        # ---------------------- Train ----------------------
        model.train()
        tstat = defaultdict(list)

        for x, y, _, xo in train_loader:
            x, xo, y = x.to(device), xo.to(device), y.to(device)

            recon, mu_q, logvar_q, mdn_out, z = model(x)
            pi, mu, log_sigma = mdn_out

            recon_loss = multiscale_recon_loss(recon, xo)
            kl_loss = kl_div_loss(mu_q, logvar_q)

            # 核心：MDN NLL 代替 MSE
            ctr_nll = mdn_nll_loss(pi, mu, log_sigma, y)

            # 仅用于监控：用混合均值当点估计，算一下 MSE（不反传）
            with torch.no_grad():
                theta_hat, conf_param, conf_global, _ = mdn_point_and_confidence(
                    pi, mu, log_sigma, var_scale=var_scale, topk=3
                )
                ctr_mse_monitor = F.mse_loss(theta_hat, y)

            # prior smoothness（保持你原始正则）
            bsz = x.size(0)
            z_prior = torch.randn(bsz, args.latent_size, device=device)
            prior_img = model.decoder(z_prior)
            sm_loss, sm_info = smoothness_loss(prior_img)

            total = (
                recon_loss
                + beta_t * kl_loss
                + args.ctr_weight * ctr_nll
                + args.smooth_weight * sm_loss
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
            tstat["smooth"].append(sm_loss.item())
            tstat["tv"].append(sm_info["tv"])
            tstat["var"].append(sm_info["var"])
            tstat["var_pen"].append(sm_info["var_pen"])

        train_epoch = {k: float(np.mean(v)) for k, v in tstat.items()}
        train_epoch["epoch"] = epoch
        train_epoch["beta"] = beta_t
        train_logs.append(train_epoch)

        # -------------------- Validation --------------------
        model.eval()
        vstat = defaultdict(list)

        with torch.no_grad():
            for x, y, _, xo in val_loader:
                x, xo, y = x.to(device), xo.to(device), y.to(device)

                recon, mu_q, logvar_q, mdn_out, z = model(x)
                pi, mu, log_sigma = mdn_out

                recon_loss = multiscale_recon_loss(recon, xo)
                kl_loss = kl_div_loss(mu_q, logvar_q)
                ctr_nll = mdn_nll_loss(pi, mu, log_sigma, y)

                theta_hat, conf_param, conf_global, _ = mdn_point_and_confidence(
                    pi, mu, log_sigma, var_scale=var_scale, topk=3
                )
                ctr_mse_monitor = F.mse_loss(theta_hat, y)

                bsz = x.size(0)
                z_prior = torch.randn(bsz, args.latent_size, device=device)
                prior_img = model.decoder(z_prior)
                sm_loss, sm_info = smoothness_loss(prior_img)

                total = (
                    recon_loss
                    + beta_t * kl_loss
                    + args.ctr_weight * ctr_nll
                    + args.smooth_weight * sm_loss
                )

                vstat["total"].append(total.item())
                vstat["recon"].append(recon_loss.item())
                vstat["kl"].append(kl_loss.item())
                vstat["ctr_nll"].append(ctr_nll.item())
                vstat["ctr_mse_monitor"].append(ctr_mse_monitor.item())
                vstat["conf_global_mean"].append(conf_global.mean().item())
                vstat["smooth"].append(sm_loss.item())
                vstat["tv"].append(sm_info["tv"])
                vstat["var"].append(sm_info["var"])
                vstat["var_pen"].append(sm_info["var_pen"])

        val_epoch = {k: float(np.mean(v)) for k, v in vstat.items()}
        val_epoch["epoch"] = epoch
        val_epoch["beta"] = beta_t
        val_logs.append(val_epoch)

        print(
            f"[Epoch {epoch:04d}] beta={beta_t:.3f} | "
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

        # -------------------- Plot losses --------------------
        plt.figure()
        plt.plot(df_train["epoch"], df_train["total"], label="train")
        plt.plot(df_val["epoch"], df_val["total"], label="val")
        plt.legend(); plt.title("Total Loss")
        plt.savefig(os.path.join(save_root, "loss_total.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(df_train["epoch"], df_train["recon"], label="recon")
        plt.plot(df_train["epoch"], df_train["kl"], label="kl")
        plt.plot(df_train["epoch"], df_train["ctr_nll"], label="ctr_nll")
        plt.legend(); plt.title("Train Main Losses")
        plt.savefig(os.path.join(save_root, "loss_train_main.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(df_train["epoch"], df_train["ctr_mse_monitor"], label="ctr_mse_monitor")
        plt.plot(df_train["epoch"], df_train["conf_global_mean"], label="conf_global_mean")
        plt.legend(); plt.title("CTR Monitor (MSE) + Global Confidence")
        plt.savefig(os.path.join(save_root, "ctr_monitor.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(df_train["epoch"], df_train["smooth"], label="smooth")
        plt.plot(df_train["epoch"], df_train["tv"], label="tv")
        plt.plot(df_train["epoch"], df_train["var_pen"], label="var_pen")
        plt.legend(); plt.title("Smoothness Losses")
        plt.savefig(os.path.join(save_root, "loss_smooth.png"), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(df_train["epoch"], df_train["beta"], label="beta")
        plt.title("Beta Schedule")
        plt.savefig(os.path.join(save_root, "beta_schedule.png"), dpi=300)
        plt.close()

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

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--image_size", type=tuple, default=(3, 64, 64))
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_params", type=int, default=15)

    # MDN
    parser.add_argument("--mdn_components", type=int, default=8)
    parser.add_argument("--mdn_hidden", type=int, default=256)

    # VAE losses
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--beta_warmup_ratio", type=float, default=0.3)

    # weights
    parser.add_argument("--ctr_weight", type=float, default=0.8)
    parser.add_argument("--smooth_weight", type=float, default=2.0)

    # confidence scaling
    parser.add_argument("--var_scale", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--save_root", type=str, default="results")

    args = parser.parse_args()
    main(args)
