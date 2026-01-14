import os

import numpy as np
import torch
import tqdm
from skimage import metrics
from torch.utils.data import DataLoader

from generative_ai.src.dataloader import DendritePFMDataset
from generative_ai.src.tools import fractal_dimension_boxcount

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = DendritePFMDataset(
    (3, 64, 64),
    os.path.join("data", "dataset_split.json"),
    split="test"
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True
)

print(f"Test set size: {len(test_dataset)} samples")

ssimss_dict = {}
rmsss_dict = {}
for f in tqdm.tqdm(os.listdir("results")):
    vae = torch.load(os.path.join(
        "results", f,
        "ckpt", "best.pt"
    ), weights_only=False).to(device)
    vae.eval()

    rmses = []
    ssims = []
    all_ids = []

    with torch.no_grad():
        for i, (x, y, did, xo) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)
            # recon_x = vae.inference()
            recon_x = vae(x)[0]
            x_np = x.detach().cpu().numpy()[0]
            r_np = recon_x.detach().cpu().numpy()[0]

            rmse = metrics.normalized_root_mse(
                x_np,
                r_np
            )
            ssim = metrics.structural_similarity(
                x_np[0],
                r_np[0],
                win_size=3,
                data_range=(x_np.max() - x_np.min())
            )
            rmses.append(rmse)
            ssims.append(ssim)

            img_gt = x_np[0, :, :]
            img_pd = r_np[0, :, :]

            all_ids.append(did[0] if isinstance(did[0], str) else str(did[0]))

    rmses = np.array(rmses)
    ssims = np.array(ssims)

    print(f"{f}")
    print(f"Metrics computed for {len(rmses)} samples.")

    ssim_avg = np.nanmean(ssims)
    rmse_avg = np.nanmean(rmses)

    print(f"Average SSIM: {ssim_avg}")
    print(f"Average RMSE: {rmse_avg}")

    ssimss_dict[f] = ssim_avg
    rmsss_dict[f] = rmse_avg

print("\n===== Sorted by SSIM (High → Low) =====")

combined_sorted = sorted(
    ssimss_dict.keys(),
    key=lambda k: ssimss_dict[k],
    reverse=True
)

for i, model in enumerate(combined_sorted, 1):
    print(
        f"{i:02d}. {model:20s} "
        f"SSIM = {ssimss_dict[model]:.6f} | "
        f"RMSE = {rmsss_dict[model]:.6f}"
    )
