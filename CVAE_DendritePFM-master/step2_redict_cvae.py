import torch
from src.models import VAE
import matplotlib.pyplot as plt

vae = VAE((3, 28, 28), 2, 256, conditional=True, num_params=4)
vae.load_state_dict(torch.load('ckpt/CVAE.pth'))
vae.eval()

params = torch.tensor([[10, 0.3, 0.3, 0.5]])
z = torch.randn(1, 2)
x_recon = vae.inference(z, c=params)

plt.imshow(x_recon.view(3, 28, 28).detach().numpy().transpose(1, 2, 0))
plt.show()
