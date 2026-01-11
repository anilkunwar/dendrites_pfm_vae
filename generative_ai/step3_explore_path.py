import os

import sklearn
import torch
from matplotlib import pyplot as plt

from src.tools import *

def heuristic(z):
    pass

image_size = (3, 64, 64)
model_root = 'results/V9_lat=8_beta=0.1_warm=0.3_ctr=2.0_smooth=1.0_time=20260105_114337'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae =  torch.load(os.path.join(model_root, "ckpt", "best.pt"), map_location=device, weights_only=False)
vae.eval()

current_list = []   # (ctr_pred, im, z)

# random explore latent space
while True:

    im, ctr_pred, z = vae.inference(full_output=True)
    im, ctr_pred, z = im[0].detach().cpu().numpy(), ctr_pred[0].detach().cpu().numpy(), z[0].detach().cpu().numpy()
    if len(current_list) == 0:
        if ctr_pred[0] < 0.01:
            current_list.append((ctr_pred, im, z))
        else:
            continue
    else:
        if ctr_pred[0] > current_list[-1][0][0]:
            # calculate vector distance
            dis = sklearn.metrics.pairwise.cosine_distances(z.reshape(1, -1) , current_list[-1][-1].reshape(1, -1) )
            if dis > 0.8:
                current_list.append((ctr_pred, im, z))
            else:
                continue
        else:
            continue

    print(f"find time step: {ctr_pred[0]}")
    print(f"full output: {ctr_pred}")

    im = im[0]

    v, _, _ = fractal_dimension_boxcount(im)
    v = np.nanmean(v)

    print(ctr_pred)
    plt.title("fractal_dimension_boxcount: {}".format(v))
    plt.imshow(im)
    plt.show()