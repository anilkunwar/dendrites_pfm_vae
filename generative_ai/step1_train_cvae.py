import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict

from src.dataloader import DendritePFMDataset
from src.models import VAE

from skimage.restoration import denoise_bilateral
from skimage import filters
from scipy import fftpack

def loss_fn(recon_x, x, mean, log_var, use_fft_loss=False, w_phys=0.1):
    """
    Physics-driven VAE loss supporting 3-channel independent images.

    Parameters
    ----------
    recon_x : torch.Tensor
        Decoder output, shape (B, 3, H, W)
    x : torch.Tensor
        Ground-truth input, shape (B, 3, H, W)
    mean : torch.Tensor
        Encoder mean (B, latent_dim)
    log_var : torch.Tensor
        Encoder log-variance (B, latent_dim)
    use_fft_loss : bool
        Whether to include the FFT-based physical loss (SL₂)
    w_phys : float
        Weight (slack) for the physical loss Ψ, 0 ≤ w ≤ 0.5

    Returns
    -------
    total_loss, recon_loss, kl_loss, phys_loss
    """

    # -----------------------------
    # 1. Reconstruction loss (independent per channel)
    # -----------------------------
    recon_loss = 0
    for c in range(3):
        recon_loss += F.mse_loss(recon_x[:, c, :, :],
                                 x[:, c, :, :],
                                 reduction="sum")
    recon_loss /= 3.0

    # -----------------------------
    # 2. KL divergence loss
    # -----------------------------
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # -----------------------------
    # 3. Physics-driven loss Ψ
    # -----------------------------
    phys_loss = 0.0
    with torch.no_grad():  # 不反传梯度，仅作正则项
        batch_np = x.detach().cpu().numpy()
        for b in range(batch_np.shape[0]):
            for c in range(3):
                img = batch_np[b, c]
                # 归一化
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                # 去噪
                img_dn = denoise_bilateral(img, sigma_color=None, sigma_spatial=5)
                if not use_fft_loss:
                    # ---------- SL₁：Scharr 边缘总和 ----------
                    emag = filters.scharr(img_dn)
                    phys_loss += np.sum(emag)
                else:
                    # ---------- SL₂：FFT 中心峰外强度比 ----------
                    Fimg = fftpack.fft2(img_dn)
                    Fimg_shift = fftpack.fftshift(Fimg)
                    Flog = np.log(np.abs(Fimg_shift) + 1)
                    total_int = np.sum(Flog)
                    h, w = Flog.shape
                    cx, cy = h // 2, w // 2
                    a = 1
                    central_peak = np.sum(
                        Flog[cx-a:cx+a, cy-a:cy+a])
                    outer_int = total_int - central_peak
                    phys_loss += outer_int / (total_int + 1e-8)
        phys_loss /= (batch_np.shape[0] * 3)

    # -----------------------------
    # 4. 组合总损失  L_phy-VAE = (φ + D_KL) × (w + Ψ)
    # -----------------------------
    total_loss = (recon_loss + kl_loss) * (1.0 + w_phys * phys_loss)

    return total_loss, recon_loss + kl_loss, torch.tensor(phys_loss)

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    tdatasets = []
    vdatasets = []
    for dpath in os.listdir("data"):
        td = DendritePFMDataset(args.image_size, os.path.join("data", dpath, "dataset_split.json"), split="train",
                                 meta_path=os.path.join("data", dpath, f"{dpath}.json"))
        vd = DendritePFMDataset(args.image_size, os.path.join("data", dpath, "dataset_split.json"), split="val",
                                meta_path=os.path.join("data", dpath, f"{dpath}.json"))
        tdatasets.append(td)
        vdatasets.append(vd)
    train_dataset = ConcatDataset(tdatasets)
    valid_dataset = ConcatDataset(vdatasets)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)

    vae = VAE(
        image_size=args.image_size,
        latent_size=args.latent_size,
        hidden_dimension=args.hidden_dimension,
        conditional=args.conditional,
        num_params=args.num_params if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        vae.train()
        for iteration, (x, y, did) in enumerate(train_dataloader):

            # image and control variables
            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z.detach().cpu().numpy())
            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z_2d[i, 0].item()
                tracker_epoch[id]['y'] = z_2d[i, 1].item()
                tracker_epoch[id]['did'] = did[i]
                tracker_epoch[id]['label'] = f"t={yi[0].item()} did={did[i]}"   # label for each sample

            loss = loss_fn(recon_x.view(x.shape), x, mean, log_var)

            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()

            logs['train_loss'].append(loss[0].item())

            if iteration % args.print_every == 0 or iteration == len(train_dataloader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Train Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(train_dataloader)-1, loss[0].item()))

        # evaluate
        vae.eval()
        for iteration, (x, y, did) in enumerate(valid_dataloader):

            # image and control variables
            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            loss = loss_fn(recon_x.view(x.shape), x, mean, log_var)
            logs['valid_total_loss'].append(loss[0].item())
            logs['valid_vae_loss'].append(loss[1].item())
            logs['valid_phy_loss'].append(loss[2].item())

            if iteration % args.print_every == 0 or iteration == len(train_dataloader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Valid Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(valid_dataset)-1, loss[0].item()))

            plt.figure()
            for p in range(min(9, recon_x.shape[0])):
                plt.subplot(3, 3, p+1)
                plt.text(
                    0, 0, f"t={y[p][0].item()}_did={did[p]}", color='black',
                    backgroundcolor='white', fontsize=8)
                plt.imshow(recon_x[p].view(args.image_size).cpu().data.numpy().transpose(1, 2, 0))
                plt.axis('off')

            if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                if not(os.path.exists(os.path.join(args.fig_root))):
                    os.mkdir(os.path.join(args.fig_root))
                os.mkdir(os.path.join(args.fig_root, str(ts)))

            plt.savefig(
                os.path.join(args.fig_root, str(ts),
                             "E{:d}I{:d}.png".format(epoch, iteration)),
                dpi=300)
            plt.clf()
            plt.close('all')

        # select 100 samples for each simulation
        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        # build platte
        df['group'] = df['label'].str.split(' ').str[1]
        main_colors = sns.color_palette('tab10', n_colors=df['group'].nunique())
        palette = {}
        for color, g in zip(main_colors, df['group'].unique()):
            # 给每个小类分配同色系的不同亮度
            shades = sns.light_palette(color, n_colors=sum(df['group'] == g) + 1)[1:]
            sublabels = df[df['group'] == g]['label'].unique()
            for s, c in zip(sublabels, shades):
                palette[s] = c
        # plot result
        g = sns.lmplot(
            x='x', y='y', hue='label', data=df,
            fit_reg=False, legend=True, palette=palette
        )
        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)

    # show loss
    plt.figure()
    plt.plot(list(range(1, len(logs["train_loss"])+1)), logs["train_loss"], label='total loss')
    plt.title("train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(
            args.fig_root, str(ts), "train_loss.png"), dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(list(range(1, len(logs["valid_total_loss"]) + 1)), logs["valid_total_loss"], label='total loss')
    plt.plot(list(range(1, len(logs["valid_vae_loss"]) + 1)), logs["valid_vae_loss"], label='vae loss')
    plt.plot(list(range(1, len(logs["valid_phy_loss"]) + 1)), logs["valid_phy_loss"], label='phy loss')
    plt.legend(loc='upper right')
    plt.title("valid_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(
        args.fig_root, str(ts), "valid_loss.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # save model
    if args.conditional:
        torch.save(vae, "ckpt/CVAE.ckpt")
    else:
        torch.save(vae, "ckpt/VAE.ckpt")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--image_size", type=tuple, default=(3, 51, 51))
    parser.add_argument("--hidden_dimension", type=int, default=1024)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--num_params", type=int, default=14)    # another param is t
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", type=bool, default=True)

    args = parser.parse_args()

    main(args)
