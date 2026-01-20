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

import albumentations as A

from src.dataloader import DendritePFMDataset
from src.modelv10 import VAE


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def regression_metrics(y_true, y_pred, tol_list=(0.01, 0.02, 0.05)):
    """
    y_true, y_pred: [N, P] torch tensors on CPU
    return dict with global + per-dim metrics
    """
    y_true = y_true.float()
    y_pred = y_pred.float()
    err = y_pred - y_true
    abs_err = err.abs()

    # Global MAE / RMSE
    mae = abs_err.mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()

    # R2 (global, treating all dims together)
    y_mean = y_true.mean(dim=0, keepdim=True)
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_mean) ** 2).sum().item()
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    # Pearson correlation per-dim (and mean)
    # corr(x,y)=cov/(stdx*stdy)
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    cov = (yt * yp).mean(dim=0)
    stdt = yt.std(dim=0, unbiased=False).clamp_min(1e-12)
    stdp = yp.std(dim=0, unbiased=False).clamp_min(1e-12)
    pearson = (cov / (stdt * stdp)).cpu().numpy()  # [P]
    pearson_mean = float(np.nanmean(pearson))

    # Per-dim MAE/RMSE
    per_mae = abs_err.mean(dim=0).cpu().numpy()
    per_rmse = torch.sqrt((err ** 2).mean(dim=0)).cpu().numpy()

    # Within tolerance "accuracy"
    tol_acc = {}
    for tol in tol_list:
        tol_acc[f"acc_tol_{tol}"] = (abs_err <= tol).float().mean().item()

    out = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_mean": pearson_mean,
        **tol_acc,
        "per_mae": per_mae,
        "per_rmse": per_rmse,
        "per_pearson": pearson,
    }
    return out


def save_curves(df, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)

    # Main curves
    plt.figure()
    plt.plot(df["epoch"], df["train_mae"], label="train_mae")
    plt.plot(df["epoch"], df["val_mae"], label="val_mae")
    plt.legend()
    plt.title("MAE")
    plt.savefig(os.path.join(save_dir, f"{prefix}_mae.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_rmse"], label="train_rmse")
    plt.plot(df["epoch"], df["val_rmse"], label="val_rmse")
    plt.legend()
    plt.title("RMSE")
    plt.savefig(os.path.join(save_dir, f"{prefix}_rmse.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_r2"], label="train_r2")
    plt.plot(df["epoch"], df["val_r2"], label="val_r2")
    plt.legend()
    plt.title("R2")
    plt.savefig(os.path.join(save_dir, f"{prefix}_r2.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_pearson_mean"], label="train_pearson_mean")
    plt.plot(df["epoch"], df["val_pearson_mean"], label="val_pearson_mean")
    plt.legend()
    plt.title("Pearson (mean over params)")
    plt.savefig(os.path.join(save_dir, f"{prefix}_pearson.png"), dpi=300)
    plt.close()

    # tol acc curves if exist
    for col in df.columns:
        if col.startswith("train_acc_tol_"):
            suf = col.replace("train_", "")
            vcol = "val_" + suf
            if vcol in df.columns:
                plt.figure()
                plt.plot(df["epoch"], df[col], label=col)
                plt.plot(df["epoch"], df[vcol], label=vcol)
                plt.legend()
                plt.title(suf)
                plt.savefig(os.path.join(save_dir, f"{prefix}_{suf}.png"), dpi=300)
                plt.close()


@torch.no_grad()
def save_scatter_and_error_hist(y_true, y_pred, save_dir, max_dims=12):
    """
    保存：
    - 每个维度 true vs pred 散点（最多 max_dims 个）
    - 每个维度误差直方图（最多 max_dims 个）
    """
    os.makedirs(save_dir, exist_ok=True)
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    P = y_true.shape[1]
    show_dims = list(range(min(P, max_dims)))

    for i in show_dims:
        # scatter
        plt.figure()
        plt.scatter(y_true[:, i], y_pred[:, i], s=6)
        plt.xlabel("true")
        plt.ylabel("pred")
        plt.title(f"Param[{i}] scatter")
        plt.savefig(os.path.join(save_dir, f"scatter_param_{i}.png"), dpi=300)
        plt.close()

        # error hist
        err = y_pred[:, i] - y_true[:, i]
        plt.figure()
        plt.hist(err, bins=60)
        plt.xlabel("pred-true")
        plt.title(f"Param[{i}] error hist")
        plt.savefig(os.path.join(save_dir, f"err_hist_param_{i}.png"), dpi=300)
        plt.close()


def set_requires_grad(m, flag: bool):
    for p in m.parameters():
        p.requires_grad = flag


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Experiment dir
    # -------------------------
    exp_name = (
        f"CTR_ONLY_lat={args.latent_size}_p={args.num_params}_"
        f"lr={args.lr}_wd={args.weight_decay}_"
        f"tol={','.join(map(str,args.tols))}_"
        f"time={datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_root = os.path.join(args.save_root, exp_name)
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, "ckpt"), exist_ok=True)

    # -------------------------
    # Dataset
    # -------------------------
    train_ds = DendritePFMDataset(
        args.image_size,
        os.path.join("data", "dataset_split.json"),
        split="train",
        transform=A.Compose([A.GaussNoise(p=0.5)]) if args.use_aug else None
    )
    val_ds = DendritePFMDataset(
        args.image_size,
        os.path.join("data", "dataset_split.json"),
        split="val"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), drop_last=False
    )

    # # -------------------------
    # # Model
    # # -------------------------
    # model = VAE(
    #     image_size=args.image_size,
    #     latent_size=args.latent_size,
    #     hidden_dimension=args.hidden_dim,
    #     num_params=args.num_params
    # ).to(device)
    #
    # # optional: load pretrained full model checkpoint
    # if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
    #     obj = torch.load(args.pretrained_ckpt, map_location="cpu")
    #     # 兼容 torch.save(model) 或 state_dict 两种保存方式
    #     if isinstance(obj, torch.nn.Module):
    #         model.load_state_dict(obj.state_dict(), strict=False)
    #     elif isinstance(obj, dict) and "state_dict" in obj:
    #         model.load_state_dict(obj["state_dict"], strict=False)
    #     elif isinstance(obj, dict):
    #         model.load_state_dict(obj, strict=False)
    model = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
    print(f"Loaded pretrained: {args.pretrained_ckpt}")

    # Freeze encoder/decoder, train only ctr_head
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.decoder, False)
    set_requires_grad(model.ctr_head, True)

    # 训练时 encoder/decoder 用 eval，避免 BN 统计漂移
    model.encoder.eval()
    model.decoder.eval()
    model.ctr_head.train()

    # Optimizer only for ctr_head
    optimizer = torch.optim.AdamW(
        model.ctr_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Optional LR scheduler
    scheduler = None
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
        )

    # -------------------------
    # Training loop
    # -------------------------
    best_val = float("inf")
    no_imp = 0
    logs = []

    for epoch in range(args.epochs):
        # ---- train ----
        model.ctr_head.train()
        train_true, train_pred = [], []
        train_loss_meter = []

        for x, y, _, _xo in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            with torch.no_grad():
                mu, logvar = model.encoder(x)
                z = mu if args.use_mu else model.reparameterize(mu, logvar)

            pred = model.ctr_head(z)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.ctr_head.parameters(), args.grad_clip)
            optimizer.step()

            train_loss_meter.append(loss.item())
            train_true.append(y.detach().cpu())
            train_pred.append(pred.detach().cpu())

        train_true = torch.cat(train_true, dim=0)
        train_pred = torch.cat(train_pred, dim=0)
        train_m = regression_metrics(train_true, train_pred, tol_list=tuple(args.tols))
        train_m["mse"] = float(np.mean(train_loss_meter))

        # ---- val ----
        model.ctr_head.eval()
        val_true, val_pred = [], []
        val_loss_meter = []
        with torch.no_grad():
            for x, y, _, _xo in val_loader:
                x = x.to(device)
                y = y.to(device).float()
                mu, logvar = model.encoder(x)
                z = mu if args.use_mu else model.reparameterize(mu, logvar)

                pred = model.ctr_head(z)
                loss = F.mse_loss(pred, y)

                val_loss_meter.append(loss.item())
                val_true.append(y.cpu())
                val_pred.append(pred.cpu())

        val_true = torch.cat(val_true, dim=0)
        val_pred = torch.cat(val_pred, dim=0)
        val_m = regression_metrics(val_true, val_pred, tol_list=tuple(args.tols))
        val_m["mse"] = float(np.mean(val_loss_meter))

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_mse": train_m["mse"],
            "val_mse": val_m["mse"],
            "train_mae": train_m["mae"],
            "val_mae": val_m["mae"],
            "train_rmse": train_m["rmse"],
            "val_rmse": val_m["rmse"],
            "train_r2": train_m["r2"],
            "val_r2": val_m["r2"],
            "train_pearson_mean": train_m["pearson_mean"],
            "val_pearson_mean": val_m["pearson_mean"],
        }
        for tol in args.tols:
            row[f"train_acc_tol_{tol}"] = train_m[f"acc_tol_{tol}"]
            row[f"val_acc_tol_{tol}"] = val_m[f"acc_tol_{tol}"]

        logs.append(row)
        df = pd.DataFrame(logs)
        df.to_csv(os.path.join(save_root, "epoch_metrics.csv"), index=False)
        save_curves(df, save_root, prefix="ctr_only")

        # 每个 epoch 也保存 per-dim 指标（便于你做“每个控制参数的准确度”分析）
        per_df = pd.DataFrame({
            "param_idx": np.arange(args.num_params),
            "train_mae": train_m["per_mae"],
            "val_mae": val_m["per_mae"] * 0 + np.nan,  # placeholder (we'll save val separately below)
        })
        per_df["val_mae"] = val_m["per_mae"]
        per_df["train_rmse"] = train_m["per_rmse"]
        per_df["val_rmse"] = val_m["per_rmse"]
        per_df["train_pearson"] = train_m["per_pearson"]
        per_df["val_pearson"] = val_m["per_pearson"]
        per_df.to_csv(os.path.join(save_root, "per_param_metrics.csv"), index=False)

        print(
            f"[Epoch {epoch:04d}] "
            f"train MAE={row['train_mae']:.6f} val MAE={row['val_mae']:.6f} | "
            f"train R2={row['train_r2']:.4f} val R2={row['val_r2']:.4f} | "
            f"val acc@tol{args.tols[0]}={row[f'val_acc_tol_{args.tols[0]}']:.4f}"
        )

        # ---- save best ----
        if row["val_mae"] < best_val:
            best_val = row["val_mae"]
            no_imp = 0
            torch.save(
                {
                    "epoch": epoch,
                    "ctr_head_state_dict": model.ctr_head.state_dict(),
                    "args": vars(args),
                    "best_val_mae": best_val,
                },
                os.path.join(save_root, "ckpt", "best_ctr_head.pt")
            )

            # best 时保存散点/误差分布图（验证集）
            save_scatter_and_error_hist(
                val_true, val_pred,
                save_dir=os.path.join(save_root, "best_vis"),
                max_dims=min(args.num_params, args.max_vis_dims)
            )
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print("🛑 Early stopping")
                break

    print(f"Done. Best val MAE = {best_val:.6f}")
    print(f"Saved to: {save_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # train
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_root", type=str, default="results")

    # model
    p.add_argument("--image_size", type=tuple, default=(3, 64, 64))
    p.add_argument("--latent_size", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_params", type=int, default=15)

    # behavior
    p.add_argument("--pretrained_ckpt", type=str, default="results/V9_lat=8_beta=0.1_warm=0.3_ctr=1.0_smooth=0.05_time=20260105_012629/ckpt/best.pt")
    p.add_argument("--use_mu", action="store_true", help="用 mu 做确定性回归（推荐），否则用 reparameterize(z) 有噪声")
    p.add_argument("--use_aug", action="store_true", help="训练集加噪声增强（和你原脚本一致）")

    # eval
    p.add_argument("--tols", type=float, nargs="+", default=[0.01, 0.02, 0.05],
                   help="容差准确率阈值列表：abs(pred-true)<=tol")
    p.add_argument("--max_vis_dims", type=int, default=12)

    args = p.parse_args()
    main(args)
