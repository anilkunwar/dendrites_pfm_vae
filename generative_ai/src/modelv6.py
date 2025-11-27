# src/model_with_ctr_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.dataloader import smooth_scale


# =========================================
# 基础模块：你之前模型的结构
# =========================================
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MultiKernelResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.conv5 = nn.Conv2d(in_ch, out_ch, 5, stride, 2, bias=False)
        self.conv7 = nn.Conv2d(in_ch, out_ch, 7, stride, 3, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if downsample or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = (self.conv3(x) + self.conv5(x) + self.conv7(x)) / 3.0
        out = self.bn(out)
        out += self.shortcut(x)
        return self.relu(out)


class ResEncoder(nn.Module):
    def __init__(self, img_size, in_ch, base_ch, latent_dim):
        super().__init__()
        H, W = img_size

        self.layers = nn.Sequential(
            MultiKernelResBlock(in_ch, base_ch, downsample=True),
            MultiKernelResBlock(base_ch, base_ch*2, downsample=True),
            MultiKernelResBlock(base_ch*2, base_ch*4, downsample=True),
        )

        conv_dim = int(H * W * base_ch * 4 / 64)
        self.fc_mu = nn.Linear(conv_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_dim, latent_dim)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return self.fc_mu(x), self.fc_logvar(x)


class ResUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.short = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        a = self.relu(self.bn1(self.deconv(x)))
        a = self.bn2(self.conv(a))
        return self.relu(a + self.short(x))


class ResDecoder(nn.Module):
    def __init__(self, out_ch, base_ch, latent_dim, H, W):
        super().__init__()
        self.H, self.W = H, W

        self.fc = nn.Linear(latent_dim, base_ch * 4 * (H // 8) * (W // 8))
        self.up = nn.Sequential(
            ResUpBlock(base_ch * 4, base_ch * 2),
            ResUpBlock(base_ch * 2, base_ch),
            ResUpBlock(base_ch, base_ch),
        )
        self.final = nn.Conv2d(base_ch, out_ch, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), -1, self.H//8, self.W//8)
        x = self.up(x)
        x = self.final(x)
        return smooth_scale(x)


# =========================================
# ⭐ 新增 z → ctr predictor
# =========================================
class CtrPredictor(nn.Module):
    def __init__(self, latent_dim, num_params, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_params),
        )

    def forward(self, z):
        return self.net(z)


# =========================================
# ⭐ 完整模型：VAE + ctr predictor
# =========================================
class VAEWithCtrPredictor(nn.Module):
    def __init__(self, image_size=(3,64,64), latent_dim=32,
                 hidden=256, num_params=15):
        super().__init__()
        C, H, W = image_size

        self.encoder = ResEncoder((H, W), C, hidden, latent_dim)
        self.decoder = ResDecoder(C, hidden, latent_dim, H, W)

        self.ctr_predictor = CtrPredictor(latent_dim, num_params)
        init_weights(self)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        recon = self.decoder(z)
        ctr_pred = self.ctr_predictor(z)

        return recon, ctr_pred, mu, logvar, z

    @torch.no_grad()
    def predict_ctr(self, x):
        mu, logvar = self.encoder(x)
        z = mu  # 用 mu 更稳定
        return self.ctr_predictor(z)
