import torch
from src.models import VAE

vae = VAE((3, 28, 28), 2, 256)
vae.load_state_dict(torch.load('ckpt/VAE.pth'))
vae.eval()

# 随机采样一个潜变量 z
z = torch.randn(1, 2)
x_recon = vae.inference(z)

# 将输出还原为图片
import matplotlib.pyplot as plt
plt.imshow(x_recon.view(3, 28, 28).detach().numpy().transpose(1, 2, 0))
plt.show()
