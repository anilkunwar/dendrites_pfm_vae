import os

import torch
from matplotlib import pyplot as plt

from src.tools import *

image_size = (3, 64, 64)
model_root = r'E:\PhDProject\dendrites_pfm_vae\generative_ai\results\V9_lat=8_beta=0.1_warm=0.2_ctr=0.5_smooth=0.05_time=20260104_001828'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae =  torch.load(os.path.join(model_root, "ckpt", "best.pt"), map_location=device, weights_only=False)
vae.eval()

# random explore latent space
for i in range(30):
    im = vae.inference()
    im = im.detach().cpu().numpy()[0, 0]

    v, _, _ = fractal_dimension_boxcount(im)
    v = np.nanmean(v)

    plt.title("fractal_dimension_boxcount: {}".format(v))
    plt.imshow(im)
    plt.show()