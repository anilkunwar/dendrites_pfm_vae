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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = DendritePFMDataset(args.image_size, os.path.join("data", "dataset_split.json"), split="test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    vae = torch.load(os.path.join(args.model_root, "ckpt", "CVAE.ckpt")).to(device)

    save_fig_path = os.path.join(args.model_root, "figures")
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    ys = []
    dids = []
    # evaluate
    with torch.no_grad():
        for iteration, (x, y, did, _) in enumerate(test_dataloader):

            ys.append(y[0])
            dids.append(did[0])

            # image and control variables
            recon_x = vae.inference(y)

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
    parser.add_argument("--model_root", type=str, default='results/V4_noise0.1_edge0.001_fft0.001_tv0.0005_smooth0.005_grad0.05_20251114_115424')

    args = parser.parse_args()

    main(args)
