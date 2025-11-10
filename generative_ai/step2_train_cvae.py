import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.ndimage import filters
from skimage.restoration import denoise_bilateral
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict

from src.dataloader import DendritePFMDataset
from src.models import VAE


class Loss(nn.Module):

    def __init__(self, device="cuda"):
        super(Loss, self).__init__()

        self.kernel_grady = torch.tensor([[[[1.], [-1.]]], [[[1.], [-1.]]]] * 3).to(device)
        self.kernel_gradx = torch.tensor([[[[1., -1.]]], [[[1., -1.]]]] * 3).to(device)

    def grad_loss(self, input, target):

        input_rectangles_h = F.conv2d(input, self.kernel_grady, padding=0, groups=3)
        target_rectangles_h = F.conv2d(target, self.kernel_grady, padding=0, groups=3)
        input_arget_rectangles_h = F.l1_loss(input_rectangles_h, target_rectangles_h)
        input_rectangles_o = F.conv2d(input, self.kernel_gradx, padding=0, groups=3)
        target_rectangles_o = F.conv2d(target, self.kernel_gradx, padding=0, groups=3)
        input_arget_rectangles_o = F.l1_loss(input_rectangles_o, target_rectangles_o)
        loss_rectangles = input_arget_rectangles_h + input_arget_rectangles_o

        return loss_rectangles

    def forward(self, recon_x, x, mean, log_var):

        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        g_loss = self.grad_loss(recon_x, x)

        return recon_loss + kl_loss + g_loss, kl_loss+recon_loss, g_loss

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

    loss_fn = Loss()
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
    parser.add_argument("--image_size", type=tuple, default=(3, 128, 128))
    parser.add_argument("--hidden_dimension", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=4)
    parser.add_argument("--num_params", type=int, default=14)    # another param is t
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", type=bool, default=True)

    args = parser.parse_args()

    main(args)
