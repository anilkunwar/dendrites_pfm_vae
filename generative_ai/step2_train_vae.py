import os
import time
from random import random
import math

import numpy as np
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import fftpack
from skimage import filters
from skimage.restoration import denoise_bilateral
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict

from torchvision import models

from src.dataloader import DendritePFMDataset
from src.modelv4 import VAE

class Loss(nn.Module):
    """
    带余弦退火 β 的 VAE 综合损失函数：
      - L1 重构损失
      - KL 散度（带余弦退火系数 β）
      - 梯度损失（增强边缘细节）
    """

    def __init__(self, device="cuda", max_beta=1.0, total_epochs=100):
        super(Loss, self).__init__()
        self.device = device
        self.max_beta = max_beta
        self.total_epochs = total_epochs
        self.current_epoch = 0  # 需要在训练循环中每个 epoch 更新一次

        # 梯度卷积核
        self.kernel_grady = torch.tensor(
            [[[[1.], [-1.]]]] * 3, device=device
        )
        self.kernel_gradx = torch.tensor(
            [[[[1., -1.]]]] * 3, device=device
        )

    # ===============================
    # β 的余弦退火函数
    # ===============================
    def get_beta(self):
        """
        根据当前 epoch 动态计算 β 值
        β 从 0 → max_beta 按余弦曲线平滑上升
        """
        ratio = self.current_epoch / self.total_epochs
        beta = 0.5 * self.max_beta * (1 - math.cos(math.pi * ratio))
        return beta

    # ===============================
    # 梯度损失
    # ===============================
    def grad_loss(self, input, target):
        input_rectangles_h = F.conv2d(input, self.kernel_grady, padding=0, groups=3)
        target_rectangles_h = F.conv2d(target, self.kernel_grady, padding=0, groups=3)
        loss_h = torch.sum(torch.abs(input_rectangles_h - target_rectangles_h) * (target_rectangles_h.abs().exp()))

        input_rectangles_o = F.conv2d(input, self.kernel_gradx, padding=0, groups=3)
        target_rectangles_o = F.conv2d(target, self.kernel_gradx, padding=0, groups=3)
        loss_o = torch.sum(torch.abs(input_rectangles_o - target_rectangles_o) * (target_rectangles_o.abs().exp()))

        return loss_h + loss_o

    # ===============================
    # 前向计算
    # ===============================
    def forward(self, recon_x, x, mean, log_var):
        # --- 基本损失 ---
        recon_loss = F.l1_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        g_loss = self.grad_loss(recon_x, x)

        # --- 余弦退火 β ---
        beta = self.get_beta()

        total_loss = recon_loss + beta * kl_loss + 0.1 * g_loss
        return total_loss / x.size(0), kl_loss, recon_loss, g_loss

    # ===============================
    # 在训练循环中更新 epoch
    # ===============================
    def step_epoch(self, epoch):
        """每个 epoch 调用一次以更新 β"""
        self.current_epoch = min(epoch, self.total_epochs)

class Lossv2(nn.Module):
    """
    综合损失（带 β 余弦退火）：
      - L1 重构损失
      - KL 散度（β 余弦退火）
      - 梯度损失（边缘/纹理）
      - VGG 感知损失（改善细节与清晰度）
    """

    def __init__(self, device="cuda", max_beta=1.0, total_epochs=100):
        super(Lossv2, self).__init__()
        self.device = device
        self.max_beta = max_beta        # KL 的最大权重
        self.total_epochs = total_epochs
        self.current_epoch = 0          # 当前训练 epoch

        # ============ 梯度卷积核 ============
        self.kernel_grady = torch.tensor([[[[1.], [-1.]]]] * 3).to(device)
        self.kernel_gradx = torch.tensor([[[[1., -1.]]]] * 3).to(device)

        # ============ 预训练 VGG 模型 ============
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False

        # 取前几个卷积层（感知特征）
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16])  # conv3_3 输出

    # -------------------------------
    # β 的余弦退火函数
    # -------------------------------
    def get_beta(self):
        """根据当前 epoch 动态计算 β 值"""
        # ratio = self.current_epoch / self.total_epochs
        # beta = 0.5 * self.max_beta * (1 - math.cos(math.pi * min(ratio, 1.0)))
        # return beta
        return 0.1

    def step_epoch(self, epoch):
        """训练循环中每个 epoch 调用以更新 β"""
        self.current_epoch = epoch

    # -------------------------------
    # 梯度损失
    # -------------------------------
    def grad_loss(self, input, target):
        input_rectangles_h = F.conv2d(input, self.kernel_grady, padding=0, groups=3)
        target_rectangles_h = F.conv2d(target, self.kernel_grady, padding=0, groups=3)
        loss_h = torch.sum(torch.abs(input_rectangles_h - target_rectangles_h) * (target_rectangles_h.abs().exp()))

        input_rectangles_o = F.conv2d(input, self.kernel_gradx, padding=0, groups=3)
        target_rectangles_o = F.conv2d(target, self.kernel_gradx, padding=0, groups=3)
        loss_o = torch.sum(torch.abs(input_rectangles_o - target_rectangles_o) * (target_rectangles_o.abs().exp()))

        return loss_h + loss_o

    # -------------------------------
    # VGG 感知损失
    # -------------------------------
    def vgg_loss(self, input, target):
        # 输入归一化到 [0,1]
        input = torch.clamp((input + 1) / 2, 0, 1)
        target = torch.clamp((target + 1) / 2, 0, 1)

        # 提取感知特征
        feat_input = self.vgg_layers(input)
        feat_target = self.vgg_layers(target)

        # L1 特征差
        return F.l1_loss(feat_input, feat_target, reduction='sum')

    # -------------------------------
    # 总损失
    # -------------------------------
    def forward(self, recon_x, x, mean, log_var):
        # 基础项
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # 余弦退火 β
        beta = self.get_beta()

        # 梯度与感知损失
        g_loss = 0.1 * self.grad_loss(recon_x, x)
        vgg_loss = 0.01 * self.vgg_loss(recon_x, x)

        # 总和
        total = recon_loss + beta * kl_loss + g_loss + vgg_loss
        return total / x.size(0), kl_loss, recon_loss, g_loss, vgg_loss

class PhysicsAugmentedVAELoss(nn.Module):
    """
    Physics-augmented VAE loss combining ELBO with custom physics-driven losses.
    Based on: "Combining Variational Autoencoders and Physical Bias for
    Improved Microscopy Data Analysis" by Arpan Biswas

    All the samples in a batch should have similarities, like from the same simulation.
    In this case we can limit their latent shape.
    """

    def __init__(self, w1=1.0, w2=0.0, use_edge_loss=True, use_fft_loss=False):
        """
        Args:
            w1: Weight for custom loss (default: 1.0)
            w2: Additional weight parameter (default: 0.0)
            use_edge_loss: Whether to use edge magnitude loss (default: True)
            use_fft_loss: Whether to use FFT-based loss (default: False)
        """
        super(PhysicsAugmentedVAELoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.use_edge_loss = use_edge_loss
        self.use_fft_loss = use_fft_loss

    def forward(self, recon_x, x, mean, log_var):
        """
        Compute physics-augmented VAE loss.

        Args:
            recon_x: Reconstructed images [batch_size, channels, H, W]
            x: Original images [batch_size, channels, H, W]
            mean: Latent mean [batch_size, latent_dim]
            log_var: Latent log variance [batch_size, latent_dim]

        Returns:
            total_loss: Weighted total loss (ELBO + custom physics loss)
            elbo_loss: Standard ELBO loss
            custom_loss: Physics-driven custom loss
        """
        batch_size = x.size(0)

        # 1. Reconstruction loss (BCE or MSE)
        recon_loss = nn.functional.l1_loss(recon_x, x, reduction='sum')

        # 2. KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # 3. Standard ELBO loss (negative ELBO)
        elbo_loss = (recon_loss + kl_loss) / batch_size

        # 4. Custom physics-driven loss
        custom_loss = 0.0
        if self.use_edge_loss:
            custom_loss += self.edge_magnitude_loss(mean)
        if self.use_fft_loss:
            custom_loss += self.fft_loss(mean)

        # Scale custom loss (as in original code)
        custom_loss = custom_loss * 1e4

        # 5. Augmented loss: ELBO weighted by (w1 + custom_loss)
        # Based on the original implementation's aug_particle calculation
        total_loss = elbo_loss * (self.w1 + custom_loss)

        return total_loss, elbo_loss, custom_loss

    def edge_magnitude_loss(self, z_mean):
        """
        Compute edge magnitude loss using Scharr filter on latent dimensions.
        Loss = mean of sum of edge magnitudes (to minimize sharp edges)

        Args:
            z_mean: Latent mean [batch_size, latent_dim]
        """
        batch_size = z_mean.size(0)
        d1 = int(np.sqrt(batch_size))
        d2 = d1

        # Check if batch can form a square image
        if d1 * d2 != batch_size:
            return 0.0

        # Use latent dimensions 2 and 3 (as in original code)
        if z_mean.size(1) < 4:
            return 0.0

        z1 = z_mean[:, 2].reshape(d1, d2)
        z2 = z_mean[:, 3].reshape(d1, d2)

        # Normalize to [0, 1]
        z1 = (z1 - z1.min()) / (z1.max() - z1.min() + 1e-8)
        z2 = (z2 - z2.min()) / (z2.max() - z2.min() + 1e-8)

        # Convert to numpy for scikit-image processing
        z1_np = z1.detach().cpu().numpy()
        z2_np = z2.detach().cpu().numpy()

        # Denoise using bilateral filter
        z1_denoised = denoise_bilateral(z1_np, sigma_color=None, sigma_spatial=5)
        z2_denoised = denoise_bilateral(z2_np, sigma_color=None, sigma_spatial=5)

        # Compute edge magnitude using Scharr filter
        emag_z1 = filters.scharr(z1_denoised)
        emag_z2 = filters.scharr(z2_denoised)

        # Loss is mean of sum of edge magnitudes
        loss = (np.sum(emag_z1) + np.sum(emag_z2)) / 2

        return loss

    def fft_loss(self, z_mean):
        """
        Compute FFT-based loss: ratio of intensity outside central peak
        to total intensity.

        Args:
            z_mean: Latent mean [batch_size, latent_dim]
        """
        batch_size = z_mean.size(0)
        d1 = int(np.sqrt(batch_size))
        d2 = d1

        # Check if batch can form a square image
        if d1 * d2 != batch_size:
            return 0.0

        # Use latent dimensions 2 and 3
        if z_mean.size(1) < 4:
            return 0.0

        z1 = z_mean[:, 2].reshape(d1, d2)
        z2 = z_mean[:, 3].reshape(d1, d2)

        # Convert to numpy for FFT
        z1_np = z1.detach().cpu().numpy()
        z2_np = z2.detach().cpu().numpy()

        # Compute FFT
        Fz1 = fftpack.fft2(z1_np)
        Fz2 = fftpack.fft2(z2_np)

        # Shift zero frequency to center
        F_shiftz1 = fftpack.fftshift(Fz1)
        F_shiftz2 = fftpack.fftshift(Fz2)

        # Log magnitude spectrum
        F_shiftz1 = np.log(np.abs(F_shiftz1) + 1)
        F_shiftz2 = np.log(np.abs(F_shiftz2) + 1)

        # Total intensity
        total_int = (np.sum(F_shiftz1) + np.sum(F_shiftz2)) / 2

        # Define central peak region (3x3 around center)
        lim_kx_min, lim_kx_max = int(d1 / 2 - 1), int(d1 / 2 + 2)
        lim_ky_min, lim_ky_max = int(d2 / 2 - 1), int(d2 / 2 + 2)

        # Central peak intensity
        central_peak_intz1 = np.sum(F_shiftz1[lim_kx_min:lim_kx_max, lim_ky_min:lim_ky_max])
        central_peak_intz2 = np.sum(F_shiftz2[lim_kx_min:lim_kx_max, lim_ky_min:lim_ky_max])

        # Intensity outside central peak
        outcentral_peak_intz1 = np.sum(F_shiftz1) - central_peak_intz1
        outcentral_peak_intz2 = np.sum(F_shiftz2) - central_peak_intz2

        # Mean ratio (minimize intensity outside central peak)
        outcentral_peak_int_mean = (outcentral_peak_intz1 + outcentral_peak_intz2) / 2
        loss = outcentral_peak_int_mean / (total_int + 1e-8)

        return loss

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()
    if not os.path.exists(os.path.join(args.fig_root, str(ts))):
        if not (os.path.exists(os.path.join(args.fig_root))):
            os.mkdir(os.path.join(args.fig_root))
        os.mkdir(os.path.join(args.fig_root, str(ts)))

    train_dataset = DendritePFMDataset(args.image_size, os.path.join("data", "dataset_split.json"), split="train")
    valid_dataset = DendritePFMDataset(args.image_size, os.path.join("data", "dataset_split.json"), split="test")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

    vae = VAE(
        image_size=args.image_size,
        latent_size=args.latent_size,
        hidden_dimension=args.hidden_dimension,
        num_params=args.num_params).to(device)

    loss_fn = Lossv2(total_epochs=args.epochs)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    logs = defaultdict(list)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 5
    for epoch in range(args.epochs):

        # tracker_epoch = defaultdict(lambda: defaultdict(dict))

        vae.train()
        for iteration, (x, y, did) in enumerate(train_dataloader):

            # image and control variables
            x, y = x.to(device), y.to(device)

            # if random() < 0.5:
            #     x_s = x[torch.randperm(x.size(0))]
            # else:
            #     x_s = x
            recon_x, mean, log_var, z = vae(x, y)

            # pca = PCA(n_components=2)
            # z_2d = pca.fit_transform(z.detach().cpu().numpy())
            # for i, yi in enumerate(y):
            #     id = len(tracker_epoch)
            #     tracker_epoch[id]['x'] = z_2d[i, 0].item()
            #     tracker_epoch[id]['y'] = z_2d[i, 1].item()
            #     tracker_epoch[id]['did'] = did[i]
            #     tracker_epoch[id]['label'] = f"t={yi[0].item()} did={did[i]}"   # label for each sample

            loss = loss_fn(recon_x.view(x.shape), x, mean, log_var)
            logs['train_total_loss'].append(loss[0].item())
            logs['train_kl_loss'].append(loss[1].item())
            logs['train_recon_loss'].append(loss[2].item())
            logs['train_phy_loss'].append(loss[3].item())
            logs['train_vgg_loss'].append(loss[4].item())

            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()

            logs['train_loss'].append(loss[0].item())

            if iteration % args.print_every == 0 or iteration == len(train_dataloader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Train Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(train_dataloader)-1, loss[0].item()))

        # evaluate
        vae.eval()
        with torch.no_grad():
            for iteration, (x, y, did) in enumerate(valid_dataloader):

                # image and control variables
                x, y = x.to(device), y.to(device)

                # shuffle the input images
                # x_s = x[torch.randperm(x.size(0))]
                recon_x, mean, log_var, z = vae(x, y)

                loss = loss_fn(recon_x.view(x.shape), x, mean, log_var)
                logs['valid_total_loss'].append(loss[0].item())
                logs['valid_kl_loss'].append(loss[1].item())
                logs['valid_recon_loss'].append(loss[2].item())
                logs['valid_phy_loss'].append(loss[3].item())

                plt.figure()
                for p in range(min(9, recon_x.shape[0])):
                    plt.subplot(3, 3, p+1)
                    plt.text(
                        0, 0, f"t={y[p][0].item()}_did={did[p]}", color='black',
                        backgroundcolor='white', fontsize=8)
                    plt.imshow(recon_x[p].view(args.image_size).detach().cpu().numpy().transpose(1, 2, 0))
                    plt.axis('off')

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

        lr_scheduler.step()
        loss_fn.step_epoch(epoch)

        # cal eval loss
        avg_valid_loss = sum(logs['valid_total_loss'][-len(valid_dataloader):]) / len(valid_dataloader)
        print(f"Epoch {epoch}: Avg Valid Loss = {avg_valid_loss:.4f}")
        # check improved
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            epochs_no_improve = 0
            # save model
            torch.save(vae, "ckpt/CVAE.ckpt")
            print(f"✅ Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epochs.")

        # early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"🛑 Early stopping triggered after {early_stop_patience} epochs without improvement.")
            break

        # # select 100 samples for each simulation
        # df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        # # build platte
        # df['group'] = df['label'].str.split(' ').str[1]
        # main_colors = sns.color_palette('tab10', n_colors=df['group'].nunique())
        # palette = {}
        # for color, g in zip(main_colors, df['group'].unique()):
        #     # 给每个小类分配同色系的不同亮度
        #     shades = sns.light_palette(color, n_colors=sum(df['group'] == g) + 1)[1:]
        #     sublabels = df[df['group'] == g]['label'].unique()
        #     for s, c in zip(sublabels, shades):
        #         palette[s] = c
        # # plot result
        # g = sns.lmplot(
        #     x='x', y='y', hue='label', data=df,
        #     fit_reg=False, legend=True, palette=palette
        # )
        # g.savefig(os.path.join(
        #     args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
        #     dpi=300)

        # show loss
        plt.figure()
        plt.plot(list(range(1, len(logs["train_total_loss"]) + 1)), logs["train_total_loss"], label='total loss')
        plt.plot(list(range(1, len(logs["train_kl_loss"]) + 1)), logs["train_kl_loss"], label='kl loss')
        plt.plot(list(range(1, len(logs["train_recon_loss"]) + 1)), logs["train_recon_loss"], label='recon loss')
        plt.plot(list(range(1, len(logs["train_phy_loss"]) + 1)), logs["train_phy_loss"], label='phy loss')
        plt.plot(list(range(1, len(logs["train_vgg_loss"]) + 1)), logs["train_vgg_loss"], label='vgg loss')
        plt.legend(loc='upper right')
        plt.title("train_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(
                args.fig_root, str(ts), "train_loss.png"), dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.plot(list(range(1, len(logs["valid_total_loss"]) + 1)), logs["valid_total_loss"], label='total loss')
        plt.plot(list(range(1, len(logs["valid_kl_loss"]) + 1)), logs["valid_kl_loss"], label='kl loss')
        plt.plot(list(range(1, len(logs["valid_recon_loss"]) + 1)), logs["valid_recon_loss"], label='recon loss')
        plt.plot(list(range(1, len(logs["valid_phy_loss"]) + 1)), logs["valid_phy_loss"], label='phy loss')
        plt.legend(loc='upper right')
        plt.title("valid_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(
            args.fig_root, str(ts), "valid_loss.png"), dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--image_size", type=tuple, default=(3, 128, 128))
    parser.add_argument("--hidden_dimension", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--num_params", type=int, default=15)    # another param is t (included here)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--fig_root", type=str, default='figs')

    args = parser.parse_args()

    main(args)
