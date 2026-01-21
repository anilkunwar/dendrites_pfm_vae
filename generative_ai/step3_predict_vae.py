import os
import time

import numpy as np
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

from generative_ai.src.modelv8 import VAE
from src.dataloader import DendritePFMDataset

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = DendritePFMDataset(args.image_size, os.path.join("data", "dataset_split.json"), split="test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    # vae = torch.load(os.path.join(
    #     args.model_root,
    #     "ckpt", "best.pt"
    # ), weights_only=False).to(device)

    vae = VAE(
        image_size=args.image_size,
        latent_size=8,
        hidden_dimension=128,
        n_components=32,
        num_params=15
    ).to(device)
    vae.load_state_dict(torch.load(os.path.join(args.model_root, "ckpt", "best.pt"), map_location=device))
    vae.eval()

    save_fig_path = os.path.join(args.model_root, "figures")
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    ys = []
    dids = []
    # evaluate
    with torch.no_grad():
        for iteration, (x, y, did, _) in enumerate(test_dataloader):

            x = x.to(device)
            y = y.to(device)

            ys.append(y[0])
            dids.append(did[0])

            # image and control variables
            recon_x = vae(x)

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
                os.path.join(save_fig_path, str(f"t={y[0][0].item()}_did={did[0]}.png")),
                dpi=300)
            plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=tuple, default=(3, 64, 64))
    parser.add_argument("--model_root", type=str, default=r'E:\PhDProject\dendrites_pfm_vae\generative_ai\results\V8_lat=8_K=32_beta=0.25_warm=0.2_ctr=2.0_smooth=0.1_time=20260103_161539')

    args = parser.parse_args()

    main(args)
