import math
import os
import time
from datetime import datetime

import torch
import albumentations as A
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd

from src.lossv4 import PhysicsConstrainedVAELoss
from src.dataloader import DendritePFMDataset
from src.modelv5 import VAE


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------
    # 输出目录
    # --------------------------
    exp_name = (
        f"V5_"
        f"latent_size{args.latent_size}_"
        f"noise{args.noise_prob}_"
        f"beta_start{args.beta_start}_"
        f"beta_end{args.beta_end}_"
        f"phy{args.w_phy}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    save_root = os.path.join(args.fig_root, exp_name)
    os.makedirs(save_root, exist_ok=True)
    os.mkdir(os.path.join(save_root, "ckpt"))

    print("📁 实验目录:", save_root)

    # --------------------------
    # 数据集（加入动态噪声）
    # --------------------------
    train_dataset = DendritePFMDataset(
        args.image_size,
        os.path.join("data", "dataset_split.json"),
        split="train",
        # transform=A.Compose([
        #     A.CoarseDropout(
        #         num_holes_range=(1, 8),
        #         hole_height_range=(0.01, 0.1),
        #         hole_width_range=(0.01, 0.1),
        #         p=0.1
        #     ),
        #     A.PixelDropout(dropout_prob=0.05, p=0.1),
        #     A.GaussNoise(p=args.noise_prob),
        # ])
    )

    valid_dataset = DendritePFMDataset(
        args.image_size,
        os.path.join("data", "dataset_split.json"),
        split="test"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # --------------------------
    # 模型
    # --------------------------
    vae = VAE(
        image_size=args.image_size,
        latent_size=args.latent_size,
        hidden_dimension=args.hidden_dimension,
        num_params=args.num_params
    ).to(device)

    # --------------------------
    # 可调权重损失
    # --------------------------
    loss_fn = PhysicsConstrainedVAELoss(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        w_grad=args.w_phy
    )

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    # --------------------------
    # 日志结构（记录全部损失）
    # --------------------------
    logs = {
        "train": defaultdict(list),
        "valid": defaultdict(list)
    }

    best_val = float("inf")
    patience = 30
    no_imp = 0

    # ======================================================
    #                     训练循环
    # ======================================================
    for epoch in range(args.epochs):

        # ==================================================
        #                     Train
        # ==================================================
        vae.train()
        for it, (x, y, did, xo) in enumerate(train_loader):
            x, y, xo = x.to(device), y.to(device), xo.to(device)

            recon_x, mu_x, log_var_x, z = vae(x, y)

            total_loss, loss_dict = loss_fn(
                recon_x.view(xo.shape),
                xo,
                mu_x,
                log_var_x
            )

            # ---- 记录损失（所有字段）----
            for k, v in loss_dict.items():
                logs["train"][k].append(v if isinstance(v, float) else float(v))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if it % args.print_every == 0:
                print(f"[Train] Epoch {epoch} It {it}/{len(train_loader)} "
                      f"Loss = {total_loss.item():.4f} "
                      f"Recon = {loss_dict['recon']:.4f} "
                      f"KL = {loss_dict['kl']:.4f} "
                      # f"Phy = {loss_dict['grad']:.4f}"
                )

        # ==================================================
        #                     Valid
        # ==================================================
        vae.eval()
        val_epoch_losses = []

        with torch.no_grad():
            for it, (x, y, did, xo) in enumerate(valid_loader):
                x, y, xo = x.to(device), y.to(device), xo.to(device)

                recon_x, mu_x, log_var_x, z = vae(x, y)

                total_loss, loss_dict = loss_fn(
                    recon_x.view(xo.shape),
                    xo,
                    mu_x,
                    log_var_x,
                    freeze=True
                )

                # ---- 记录损失（所有字段）----
                for k, v in loss_dict.items():
                    logs["valid"][k].append(v if isinstance(v, float) else float(v))

                val_epoch_losses.append(total_loss.item())

                # ------------------------------------
                # 保存图片（保持原样）
                # ------------------------------------
                plt.figure()
                for p in range(min(9, recon_x.shape[0])):
                    plt.subplot(3, 3, p + 1)
                    plt.text(
                        0, 0, f"t={y[p][0].item()} did={did[p]}",
                        color='black', backgroundcolor='white', fontsize=8
                    )
                    plt.imshow(
                        recon_x[p].view(args.image_size)
                        .detach().cpu().numpy().transpose(1, 2, 0)
                    )
                    plt.axis("off")

                plt.savefig(os.path.join(save_root, f"E{epoch}_I{it}.png"), dpi=300)
                plt.close()

        avg_val = sum(val_epoch_losses) / len(val_epoch_losses)
        print(f"[Valid] Epoch {epoch} AvgLoss = {avg_val:.4f}")

        # ==================================================
        #                    Early Stopping
        # ==================================================
        if avg_val < best_val:
            best_val = avg_val
            no_imp = 0
            torch.save(vae, os.path.join(save_root, "ckpt", "CVAE.ckpt"))
            print("✔ Model improved & saved")
        else:
            no_imp += 1
            print(f"No improvement {no_imp}/{patience}")

        if no_imp >= patience or math.isnan(avg_val):
            print("🛑 Early stopping")
            break

        # ==================================================
        #               每个 epoch 保存损失曲线
        # ==================================================

        # --- train ---
        df_train = pd.DataFrame(logs["train"])
        df_train.to_csv(os.path.join(save_root, "train_loss.csv"), index=False)

        plt.figure()
        for k in df_train.columns:
            plt.plot(df_train[k], label=k)
        plt.legend()
        plt.title("Train Loss")
        plt.savefig(os.path.join(save_root, "train_loss.png"), dpi=300)
        plt.close()

        # --- valid ---
        df_valid = pd.DataFrame(logs["valid"])
        df_valid.to_csv(os.path.join(save_root, "valid_loss.csv"), index=False)

        plt.figure()
        for k in df_valid.columns:
            plt.plot(df_valid[k], label=k)
        plt.legend()
        plt.title("Valid Loss")
        plt.savefig(os.path.join(save_root, "valid_loss.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--image_size", type=tuple, default=(3, 64, 64))
    parser.add_argument("--hidden_dimension", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--num_params", type=int, default=15)
    parser.add_argument("--print_every", type=int, default=10)

    # 动态参数
    parser.add_argument("--noise_prob", type=float, default=1.0)
    parser.add_argument("--beta_start", type=float, default=0)
    parser.add_argument("--beta_end", type=float, default=0)
    parser.add_argument("--anneal_steps", type=int, default=100)
    parser.add_argument("--w_phy", type=float, default=0)

    parser.add_argument("--fig_root", type=str, default="results")

    args = parser.parse_args()
    main(args)
