import os
import time
from random import random
import math

import numpy as np
import torch
import albumentations as A
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

from generative_ai.src.loss import AdaptivePhysicsVAELoss, PhysicsConstrainedVAELoss
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

    train_dataset = DendritePFMDataset(args.image_size, os.path.join("data", "dataset_split.json"), split="train",
                                       transform=A.Compose([
                                           A.CoarseDropout(
                                               num_holes_range=(1, 8),
                                               hole_height_range=(0.01, 0.1),
                                               hole_width_range=(0.01, 0.1),
                                               p=0.1
                                           ),
                                           A.PixelDropout(dropout_prob=0.05, p=0.1),
                                           A.GaussNoise(p=0.9),
                                        ]))
    valid_dataset = DendritePFMDataset(args.image_size, os.path.join("data", "dataset_split.json"), split="test")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

    vae = VAE(
        image_size=args.image_size,
        latent_size=args.latent_size,
        hidden_dimension=args.hidden_dimension,
        num_params=args.num_params).to(device)

    # loss_fn = Loss()
    loss_fn = PhysicsConstrainedVAELoss()
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    logs = defaultdict(list)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 5
    for epoch in range(args.epochs):

        # loss_fn.set_epoch(epoch)

        vae.train()
        for iteration, (x, y, did, xo) in enumerate(train_dataloader):

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

            total_loss, loss_dict = loss_fn(recon_x.view(xo.shape), xo, mean, log_var)
            logs['train_total_loss'].append(total_loss.item())
            logs['train_kl_loss'].append(loss_dict['kl'])
            logs['train_recon_loss'].append(loss_dict['recon'])
            logs['train_phy_loss'].append(loss_dict['physics'])
            logs['train_edge_loss'].append(loss_dict['edge'])

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            logs['train_loss'].append(total_loss.item())

            if iteration % args.print_every == 0 or iteration == len(train_dataloader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Train Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(train_dataloader)-1, total_loss.item()))

        # evaluate
        vae.eval()
        with torch.no_grad():
            for iteration, (x, y, did, xo) in enumerate(valid_dataloader):

                # image and control variables
                x, y = x.to(device), y.to(device)

                # shuffle the input images
                # x_s = x[torch.randperm(x.size(0))]
                recon_x, mean, log_var, z = vae(x, y)

                total_loss, loss_dict = loss_fn(recon_x.view(xo.shape), xo, mean, log_var)
                logs['valid_total_loss'].append(total_loss.item())
                logs['valid_kl_loss'].append(loss_dict['kl'])
                logs['valid_recon_loss'].append(loss_dict['recon'])
                logs['valid_phy_loss'].append(loss_dict['physics'])
                logs['valid_edge_loss'].append(loss_dict['edge'])

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

        # cal eval loss
        avg_valid_loss = sum(logs['valid_total_loss'][-len(valid_dataloader):]) / len(valid_dataloader)
        print(f"Epoch {epoch}: Avg Valid Loss = {avg_valid_loss:.4f}")

        lr_scheduler.step()

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
            # if epochs_no_improve >= early_stop_patience / 2:
            #     optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
            #     lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--image_size", type=tuple, default=(3, 32, 32))
    parser.add_argument("--hidden_dimension", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--num_params", type=int, default=15)    # another param is t (included here)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--fig_root", type=str, default='figs')

    args = parser.parse_args()

    main(args)
