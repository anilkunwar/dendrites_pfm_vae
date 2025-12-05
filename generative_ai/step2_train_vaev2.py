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
import torch.nn as nn
import torch.nn.functional as F

from src.dataloader import DendritePFMDataset
from src.modelv6 import VAE

import numpy as np
class ChannelDropout(A.ImageOnlyTransform):
    """
    随机将一个或多个通道置零（或掩码掉）
    """

    def __init__(self, drop_prob=0.5, num_drop_channels=1, always_apply=False, p=0.5):
        super(ChannelDropout, self).__init__(always_apply, p)
        self.drop_prob = drop_prob
        self.num_drop_channels = num_drop_channels

    def apply(self, img, **params):
        # img: H x W x C
        h, w, c = img.shape

        # 每个通道被选为drop的概率
        if np.random.rand() < self.drop_prob:
            drop_channels = np.random.choice(c, self.num_drop_channels, replace=False)
            img = img.copy()
            img[:, :, drop_channels] = 0  # 掩码
        return img

def multiscale_recon_loss(
    x_recon_logits,
    x_true,
    num_scales=4,
    scale_weight=1
):
    """
    多尺度重建损失
    - 每往下采样一倍，分辨率变为上一层的一半
    - 低分辨率 -> 强调形状 / 连通性
    - 高分辨率 -> 保留局部细节

    参数：
        x_recon_logits: B x 1 x H x W  (你的模型输出 logits)
        x_true:         B x 1 x H x W  (ground truth)
        num_scales:     一共使用多少尺度
        loss_type:      "bce" 或 "mse"
        scale_weight:   (0~1) coarse 层权重随尺度递减的倍数
                        e.g. 0.5 → coarse 层权重要比细层大
    """

    # 初始损失
    total_loss = 0.0
    weight_sum = 0.0

    # 当前尺度输入
    cur_pred = x_recon_logits
    cur_true = x_true

    # 权重：粗尺度更重要（形态），细尺度稍弱（像素对齐）
    # e.g. scale_weight = 0.5 → 权重依次：1, 0.5, 0.25, 0.125 ...
    cur_weight = 1.0

    for scale in range(num_scales):

        loss = F.mse_loss(cur_pred, cur_true, reduction="sum")

        total_loss += cur_weight * loss
        weight_sum += cur_weight

        # 下一个尺度：尺寸减半
        # 注意：这里我们用 avg_pool 下采样
        cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2)
        cur_true = F.avg_pool2d(cur_true, kernel_size=2, stride=2)

        # 更新权重
        cur_weight *= scale_weight

    # 平均化权重
    return total_loss / weight_sum

class PhysicsConstrainedVAELoss(nn.Module):
    """
    Physics-constrained loss with annealed beta-VAE KL term.
    """

    def __init__(self,
                 beta_start=0.0,         # 初始 β
                 beta_end=4.0,           # 最终 β
                 anneal_steps=1000,     # β 从 start 增到 end 的步数
                 w_grad=0.01,
                 scale_weight=1,
                 device="cuda"):
        super().__init__()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_steps = anneal_steps
        self.current_step = 0

        self.w_grad = w_grad
        self.scale_weight = scale_weight

        # gradient kernels
        self.kernel_grady = torch.tensor(
            [[[[1.], [-1.]]]] * 3, device=device
        )
        self.kernel_gradx = torch.tensor(
            [[[[1., -1.]]]] * 3, device=device
        )

    # ----------------------------
    #  计算当前 β (线性 Annealing)
    # ----------------------------
    def compute_beta(self):
        t = min(self.current_step / self.anneal_steps, 1)
        beta = self.beta_start + t * (self.beta_end - self.beta_start)
        return beta

    def grad_loss(self, input, target):
        """
        对 dx 和 dy 的误差做 0–1 逐通道归一化。
        归一化区间来自 target 的每个通道的 min/max。
        """

        C = input.shape[1]

        # dx: (B, C, H, W-1)
        input_dx = F.conv2d(input, self.kernel_gradx, padding=0, groups=C)
        target_dx = F.conv2d(target, self.kernel_gradx, padding=0, groups=C)

        # dy: (B, C, H-1, W)
        input_dy = F.conv2d(input, self.kernel_grady, padding=0, groups=C)
        target_dy = F.conv2d(target, self.kernel_grady, padding=0, groups=C)

        # # --------------------------------------------------
        # # 每通道: 用 target_dx/target_dy 的 min/max 做 0-1 归一化
        # # --------------------------------------------------
        # # dx 通道的 min/max
        # dx_min = target_dx.amin(dim=(2, 3), keepdim=True)
        # dx_max = target_dx.amax(dim=(2, 3), keepdim=True)
        # dx_scale = (dx_max - dx_min).clamp_min(1e-6)
        #
        # # dy 通道的 min/max
        # dy_min = target_dy.amin(dim=(2, 3), keepdim=True)
        # dy_max = target_dy.amax(dim=(2, 3), keepdim=True)
        # dy_scale = (dy_max - dy_min).clamp_min(1e-6)
        #
        # # 归一化
        # input_dx_norm = (input_dx - dx_min) / dx_scale
        # target_dx_norm = (target_dx - dx_min) / dx_scale
        #
        # input_dy_norm = (input_dy - dy_min) / dy_scale
        # target_dy_norm = (target_dy - dy_min) / dy_scale

        # 归一化后做 L1
        dx_loss = F.mse_loss(input_dx, target_dx, reduction="sum")
        dy_loss = F.mse_loss(input_dy, target_dy, reduction="sum")
        return dx_loss + dy_loss

    def forward(self, recon_x, x, kl, freeze=False):

        if not freeze:
            self.current_step += 1

        beta = self.compute_beta()

        # # ---------------------------------------
        # # 逐通道 0-1 归一化（参考 ground truth）
        # # x_norm, recon_norm : [B,C,H,W]
        # # ---------------------------------------
        # x_min = x.amin(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        # x_max = x.amax(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        #
        # range_ = x_max - x_min  # [B,C,1,1]
        # is_const = (range_ < 1e-12)  # 判断是否是常数通道（max==min）
        #
        # # 常数通道 scale = 1，否则正常 scale
        # scale = torch.where(is_const, torch.ones_like(range_), range_.clamp_min(1e-6))
        #
        # x_norm = (x - x_min) / scale
        # recon_norm = (recon_x - x_min) / scale

        # 重建损失（在归一化空间中算 L1）
        recon_loss = multiscale_recon_loss(recon_x, x, scale_weight=self.scale_weight)

        # KL 损失
        kl = kl.mean()
        kl_loss = beta * kl

        # 梯度损失（逐通道 normalization）
        grad_loss = self.grad_loss(recon_x, x) * self.w_grad

        total_loss = recon_loss + kl_loss + grad_loss

        return total_loss, {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "kl": kl.item(),
            "beta": beta,
            "grad": grad_loss.item()
        }

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------
    # 输出目录
    # --------------------------
    exp_name = (
        f"V6_"
        f"latent_size{args.latent_size}_"
        f"noise{args.noise_prob}_"
        f"beta_start{args.beta_start}_"
        f"beta_end{args.beta_end}_"
        f"phy{args.w_phy}_"
        f"ncomp{args.n_components}_"
        f"scale_weight{args.scale_weight}_"
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
        transform=A.Compose([
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.01, 0.1),
                hole_width_range=(0.01, 0.1),
                p=0.1
            ),
            A.PixelDropout(dropout_prob=0.05, p=0.1),
            ChannelDropout(p=0.5, num_drop_channels=1),
            A.GaussNoise(p=args.noise_prob),
        ])
    )

    valid_dataset = DendritePFMDataset(
        args.image_size,
        os.path.join("data", "dataset_split.json"),
        split="val"
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
        num_params=args.num_params,
        n_components=args.n_components
    ).to(device)

    # --------------------------
    # 可调权重损失
    # --------------------------
    loss_fn = PhysicsConstrainedVAELoss(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        w_grad=args.w_phy,
        scale_weight=args.scale_weight
    )

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.anneal_steps)

    # --------------------------
    # 日志结构（记录全部损失）
    # --------------------------
    logs = {
        "train": defaultdict(list),
        "valid": defaultdict(list)
    }

    best_val = float("inf")
    patience = 50
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

            recon_x, mu_x, log_var_x, kl = vae(x, y)

            total_loss, loss_dict = loss_fn(
                recon_x.view(xo.shape),
                xo,
                kl
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
                      f"Phy = {loss_dict['grad']:.4f}"
                )

        # ==================================================
        #                     Valid
        # ==================================================
        vae.eval()
        val_epoch_losses = []

        with torch.no_grad():
            for it, (x, y, did, xo) in enumerate(valid_loader):
                x, y, xo = x.to(device), y.to(device), xo.to(device)

                recon_x, mu_x, log_var_x, kl = vae(x, y)

                total_loss, loss_dict = loss_fn(
                    recon_x.view(xo.shape),
                    xo,
                    kl,
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

        lr_scheduler.step()

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
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--image_size", type=tuple, default=(3, 64, 64))
    parser.add_argument("--hidden_dimension", type=int, default=128)
    parser.add_argument("--latent_size", type=int, default=32)  # 128 -> 64
    parser.add_argument("--num_params", type=int, default=15)
    parser.add_argument("--print_every", type=int, default=10)

    # 动态参数
    parser.add_argument("--noise_prob", type=float, default=0.8)
    parser.add_argument("--beta_start", type=float, default=0.1)
    parser.add_argument("--beta_end", type=float, default=500.0)
    parser.add_argument("--anneal_steps", type=int, default=1500)
    parser.add_argument("--w_phy", type=float, default=0.25)

    parser.add_argument("--n_components", type=int, default=64)
    parser.add_argument("--scale_weight", type=float, default=0.5)

    parser.add_argument("--fig_root", type=str, default="results")

    args = parser.parse_args()
    main(args)
