import os
import time

import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict

from src.dataloader import DendritePFMDataset
from src.models import VAE

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

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy_with_logits(
            recon_x.view(recon_x.shape[0], -1), x.view(x.shape[0], -1), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

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

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['did'] = did[i]
                tracker_epoch[id]['label'] = f"t={yi[0].item()} did={did[i]}"   # label for each sample

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['train_loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(train_dataloader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Train Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(train_dataloader)-1, loss.item()))

        # evaluate
        vae.eval()
        for iteration, (x, y, did) in enumerate(valid_dataloader):

            # image and control variables
            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            loss = loss_fn(recon_x, x, mean, log_var)
            logs['valid_loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(train_dataloader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Valid Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(valid_dataset)-1, loss.item()))

            plt.figure(figsize=(5, 10))
            for p in range(min(9, recon_x.shape[0])):
                plt.subplot(3, 3, p+1)
                if args.conditional:
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
    plt.plot(list(range(1, len(logs["train_loss"])+1)), logs["train_loss"], marker='o', color='b', label='y = x^2')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(
            args.fig_root, str(ts), "train_loss.png"), dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(list(range(1, len(logs["valid_loss"]) + 1)), logs["valid_loss"], marker='o', color='b', label='y = x^2')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(
        args.fig_root, str(ts), "valid_loss.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # save model
    if args.conditional:
        torch.save(vae.state_dict(), "ckpt/CVAE.pth")
    else:
        torch.save(vae.state_dict(), "ckpt/VAE.pth")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--image_size", type=tuple, default=(3, 51, 51))
    parser.add_argument("--hidden_dimension", type=tuple, default=1024)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--num_params", type=int, default=14)    # another param is t
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", type=bool, default=True)

    args = parser.parse_args()

    main(args)
