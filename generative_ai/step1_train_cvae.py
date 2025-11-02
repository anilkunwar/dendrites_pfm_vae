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

    dataset1 = DendritePFMDataset(args.image_size, "Data/sim_0/dataset_split.json", split="train",
                                 meta_path="Data/sim_0_meta.csv")
    dataset2 = DendritePFMDataset(args.image_size, "Data/sim_1/dataset_split.json", split="train",
                                 meta_path="Data/sim_1_meta.csv")
    dataset3 = DendritePFMDataset(args.image_size, "Data/sim_2/dataset_split.json", split="train",
                                 meta_path="Data/sim_2_meta.csv")
    dataset = ConcatDataset([dataset1, dataset2, dataset3])
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

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

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['params'] = yi
                tracker_epoch[id]['label'] = f"c={yi[1:].sum().item():.2f}_{yi[0].item():.2f}"

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if args.conditional:
                    c = torch.randn(6, args.num_params).to(device) # g
                    z = torch.randn([c.size(0), args.latent_size]).to(device)
                    x = vae.inference(z, c=c)
                else:
                    z = torch.randn([6, args.latent_size]).to(device)
                    x = vae.inference(z)

                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(6):
                    plt.subplot(3, 2, p+1)
                    if args.conditional:
                        plt.text(
                            0, 0, f"c={c[p][1:].sum().item():.2f}_{c[p][0].item():.2f}", color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].view(args.image_size).cpu().data.numpy().transpose(1, 2, 0))
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

        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        g = sns.lmplot(
            x='x', y='y', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)
        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)

    # show loss
    plt.plot(list(range(1, len(logs["loss"])+1)), logs["loss"], marker='o', color='b', label='y = x^2')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(
            args.fig_root, str(ts), "loss.png"), dpi=300, bbox_inches='tight')
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--image_size", type=tuple, default=(3, 28, 28))
    parser.add_argument("--hidden_dimension", type=tuple, default=256)
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--num_params", type=int, default=4)    # another param is t
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", type=bool, default=True)

    args = parser.parse_args()

    main(args)
