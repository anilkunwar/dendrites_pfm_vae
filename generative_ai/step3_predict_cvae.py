import os
import time

import numpy as np
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict

from src.dataloader import DendritePFMDataset

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tdatasets = []
    for dpath in os.listdir("data"):
        td = DendritePFMDataset(args.image_size, os.path.join("data", dpath, "dataset_split.json"), split="test",
                                 meta_path=os.path.join("data", dpath, f"{dpath}.json"))
        tdatasets.append(td)
    test_dataset = ConcatDataset(tdatasets)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    vae = torch.load(os.path.join("ckpt", "CVAE.ckpt" if args.conditional else "VAE.ckpt"))

    if not os.path.exists(args.save_fig_path):
        os.makedirs(args.save_fig_path)

    zs = []
    ys = []
    dids = []
    # evaluate
    vae.eval()
    for iteration, (x, y, did) in enumerate(test_dataloader):

        ys.append(y[0])
        dids.append(did[0])

        # image and control variables
        x, y = x.to(device), y.to(device)

        if args.conditional:
            recon_x, mean, log_var, z = vae(x, y)
        else:
            recon_x, mean, log_var, z = vae(x)

        zs.append(z.detach().cpu()[0].numpy())

        plt.figure()
        plt.title(f"t={y[0][0].item()}_did={did[0]}")
        plt.axis('off')
        plt.subplot(3, 2, 1)
        plt.imshow(x[0].view(args.image_size).cpu().data.numpy()[0])
        plt.subplot(3, 2, 2)
        plt.imshow(recon_x[0].view(args.image_size).cpu().data.numpy()[0])
        plt.subplot(3, 2, 3)
        plt.imshow(x[0].view(args.image_size).cpu().data.numpy()[1])
        plt.subplot(3, 2, 4)
        plt.imshow(recon_x[0].view(args.image_size).cpu().data.numpy()[1])
        plt.subplot(3, 2, 5)
        plt.imshow(x[0].view(args.image_size).cpu().data.numpy()[2])
        plt.subplot(3, 2, 6)
        plt.imshow(recon_x[0].view(args.image_size).cpu().data.numpy()[2])
        plt.savefig(
            os.path.join(args.save_fig_path, str(f"t={y[0][0].item()}_did={did[0]}.png")),
            dpi=300)
        plt.show()

    tracker_epoch = defaultdict(lambda: defaultdict(dict))
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(np.array(zs))
    for i, yi in enumerate(ys):
        id = len(tracker_epoch)
        tracker_epoch[id]['x'] = z_2d[i, 0].item()
        tracker_epoch[id]['y'] = z_2d[i, 1].item()
        tracker_epoch[id]['did'] = dids[i]
        tracker_epoch[id]['label'] = f"t={yi[0].item()} did={dids[i]}"  # label for each sample
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
    g.savefig(os.path.join(args.save_fig_path, "Dist.png"), dpi=300)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=tuple, default=(3, 128, 128))
    parser.add_argument("--conditional", type=bool, default=True)
    parser.add_argument("--save_fig_path", type=str, default='test_figs')

    args = parser.parse_args()

    main(args)
