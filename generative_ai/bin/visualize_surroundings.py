import glob
import os
import random

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.dataloader import smooth_scale, inv_smooth_scale
from src.modelv11 import mdn_point_and_confidence, postprocess_image


def get_init_tensor(candidates, image_size):
    path = random.choice(candidates)
    arr = np.load(path)  # shape (H, W, 3)

    # assert arr.max() <= 5 and arr.min() >= -5, "Inappropriate values occured"
    arr = cv2.resize(arr, image_size)

    tensor = torch.from_numpy(arr).float().permute(2, 0, 1)  # -> (3, H, W)

    tensor = smooth_scale(tensor)

    return tensor

if __name__ == '__main__':

    # ====== CONFIG ======
    MODEL_ROOT = "results/final_model/"
    CKPT_PATH = os.path.join(MODEL_ROOT, "ckpt", "best.pt")
    IMAGE_SIZE = (48, 48)
    NUM_CAND = 6
    RW_SIGMA = 0.25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir_or_pattern = "data"
    recursive = True

    model = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.eval()

    # Build candidate list
    if isinstance(data_dir_or_pattern, (list, tuple)):
        candidates = list(data_dir_or_pattern)
    else:
        s = str(data_dir_or_pattern)
        if os.path.isdir(s):
            pattern = os.path.join(s, "**", "*.npy") if recursive else os.path.join(s, "*.npy")
            candidates = glob.glob(pattern, recursive=recursive)
        else:
            candidates = glob.glob(s, recursive=recursive)

    candidates = [p for p in candidates if os.path.isfile(p)]

    z = None
    dz = None
    for _ in range(NUM_CAND):

        if z is not None:
            dz = torch.randn(*z.shape) * RW_SIGMA
            z = z + dz.to(device)
            recon, _ = model.inference(z)
        else:
            with torch.no_grad():
                recon, mu_q, logvar_q, (pi_s, mu_s, log_sigma_s), z = model(get_init_tensor(candidates, IMAGE_SIZE).unsqueeze(0).to(device))

        recon = inv_smooth_scale(recon)
        recon = recon.cpu().detach().numpy()[0, 0]
        recon = postprocess_image(recon)

        plt.figure(figsize=(6, 5))
        plt.imshow(recon, cmap="coolwarm")
        # plt.colorbar(fraction=0.046)
        if dz is None:
            plt.title("dz=0")
        else:
            plt.title(f"dz={torch.norm(dz).item():.2f}")
        plt.axis("off")
        plt.margins(0, 0)
        plt.tight_layout()
        plt.show()