# src/modelv11.py
"""
VAE for image reconstruction / prediction + Linear Regression head for simulation
parameter prediction.

This version keeps the VAE image branch:
    x -> encoder -> (mu_q, logvar_q) -> z -> decoder -> recon

and replaces the MDN parameter head with a simple linear regression head:
    mu_q or z -> Linear(num_params) -> theta_hat

Recommended default:
    regression_source="mu"
because using mu_q avoids sampling noise when probing the latent representation
for simulation parameter prediction.
"""

import math
import numpy as np
from scipy import ndimage as ndi

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataloader import smooth_scale, inv_smooth_scale


def init_weights(model):
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ==========================================================
# Multi-kernel Residual Block (Encoder)
# ==========================================================
class MultiKernelResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, stride, 3, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)

        out = (out3 + out5 + out7) / 3.0
        out = self.bn(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class ResEncoder(nn.Module):
    """Downsampling 3 times: H,W -> H/8,W/8."""

    def __init__(self, img_hw, in_channels, base_channels, latent_size):
        super().__init__()

        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),
        )

        conv_dim = int(img_hw[0] * img_hw[1] * base_channels * 4 / (8 * 8))
        self.fc_mu = nn.Linear(conv_dim, latent_size)
        self.fc_logvar = nn.Linear(conv_dim, latent_size)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.bn2(out)

        sc = self.shortcut(x)
        out = out + sc
        return self.relu(out)


class ChannelWiseBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GroupResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.short = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.deconv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.bn2(out)

        sc = self.short(x)
        out = out + sc
        return self.relu(out)


class ResDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, latent_size, H, W):
        super().__init__()
        self.H, self.W = H, W
        self.out_channels = out_channels
        init_h, init_w = H // 8, W // 8

        self.fc = nn.Linear(latent_size, base_channels * 4 * init_h * init_w)

        self.shared_up = nn.Sequential(
            ResUpBlock(base_channels * 4, base_channels * 2),
            ResUpBlock(base_channels * 2, base_channels),
        )

        ch_per_group = math.ceil(base_channels / out_channels)
        self.group_channels = ch_per_group * out_channels
        self.align_proj = nn.Conv2d(base_channels, self.group_channels, kernel_size=1, bias=False)
        self.align_bn = nn.BatchNorm2d(self.group_channels)

        self.group_up = GroupResUpBlock(
            in_channels=self.group_channels,
            out_channels=self.group_channels,
            groups=out_channels,
        )

        self.refine = nn.Sequential(
            ChannelWiseBlock(self.group_channels, groups=out_channels),
            ChannelWiseBlock(self.group_channels, groups=out_channels),
        )

        self.head = nn.Conv2d(
            self.group_channels, out_channels, kernel_size=1, groups=out_channels, bias=True
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.H // 8, self.W // 8)

        x = self.shared_up(x)
        x = self.align_bn(self.align_proj(x))
        x = self.group_up(x)
        x = self.refine(x)
        x = self.head(x)

        return smooth_scale(x)


# ==========================================================
# Linear Regression Head for simulation parameters
# ==========================================================
class LinearRegressionHead(nn.Module):
    """
    A deliberately simple baseline/probe:
        latent -> linear projection -> simulation parameters

    No hidden layers are used, so the parameter prediction performance mainly
    reflects the information encoded in the latent representation.
    """

    def __init__(self, in_dim: int, num_params: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_params)

    def forward(self, h):
        return self.fc(h)


# ==========================================================
# VAE + Linear Regression Head
# ==========================================================
class VAE_LinearRegression(nn.Module):
    """
    forward output:
        recon, mu_q, logvar_q, theta_hat, z

    - recon is produced by the VAE decoder from sampled z.
    - theta_hat is predicted by a linear regression head from either mu_q or z.
    """

    def __init__(
        self,
        image_size=(3, 128, 128),
        latent_size=64,
        hidden_dimension=64,
        num_params=15,
        regression_source: str = "mu",
    ):
        super().__init__()
        if regression_source not in {"mu", "z"}:
            raise ValueError("regression_source must be either 'mu' or 'z'.")

        self.C, self.H, self.W = image_size
        self.latent_size = latent_size
        self.num_params = num_params
        self.regression_source = regression_source

        self.encoder = ResEncoder(
            (image_size[1], image_size[2]),
            self.C,
            hidden_dimension,
            latent_size,
        )

        self.decoder = ResDecoder(
            out_channels=self.C,
            base_channels=hidden_dimension,
            latent_size=latent_size,
            H=self.H,
            W=self.W,
        )

        self.regression_head = LinearRegressionHead(
            in_dim=latent_size,
            num_params=num_params,
        )

        init_weights(self)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def predict_params_from_latent(self, latent):
        return self.regression_head(latent)

    def forward(self, x):
        mu_q, logvar_q = self.encoder(x)
        z = self.reparameterize(mu_q, logvar_q)

        recon = self.decoder(z)

        if self.regression_source == "mu":
            reg_feature = mu_q
        else:
            reg_feature = z

        theta_hat = self.regression_head(reg_feature)

        return recon, mu_q, logvar_q, theta_hat, z

    @torch.no_grad()
    def inference(self, x=None, z=None, use_mu_for_recon: bool = True):
        """
        Inference modes:
        1. Given x:
            encode x, reconstruct image, and predict parameters.
        2. Given z:
            decode z only. Parameter prediction is also returned from z.

        Returns:
            recon, theta_hat, latent_info
        """
        if x is None and z is None:
            raise ValueError("Either x or z must be provided.")

        if x is not None:
            mu_q, logvar_q = self.encoder(x)
            z_sample = self.reparameterize(mu_q, logvar_q)
            z_recon = mu_q if use_mu_for_recon else z_sample
            recon = self.decoder(z_recon)

            if self.regression_source == "mu":
                theta_hat = self.regression_head(mu_q)
            else:
                theta_hat = self.regression_head(z_sample)

            latent_info = {
                "mu_q": mu_q,
                "logvar_q": logvar_q,
                "z": z_sample,
            }
            return recon, theta_hat, latent_info

        recon = self.decoder(z)
        theta_hat = self.regression_head(z)
        latent_info = {"z": z}
        return recon, theta_hat, latent_info


def vae_linear_regression_loss(
    recon,
    target_img,
    mu_q,
    logvar_q,
    theta_hat,
    theta,
    recon_weight: float = 1.0,
    kl_weight: float = 1.0,
    param_weight: float = 1.0,
    prior_var: float = 1.0,
    reduction: str = "mean",
):
    """
    Convenience loss for VAE image reconstruction + linear parameter regression.

    target_img:
        Usually the clean target image xo from the dataloader.
    theta:
        Simulation parameter vector, shape [B, num_params].
    """
    recon_loss = F.mse_loss(recon, target_img, reduction=reduction)
    param_loss = F.mse_loss(theta_hat, theta, reduction=reduction)

    var_q = torch.exp(logvar_q)
    kl_per_sample = 0.5 * torch.sum(
        var_q / prior_var
        + mu_q ** 2 / prior_var
        - 1.0
        + math.log(prior_var)
        - logvar_q,
        dim=1,
    )
    kl_loss = torch.mean(kl_per_sample) / mu_q.shape[1]

    total = (
        recon_weight * recon_loss
        + kl_weight * kl_loss
        + param_weight * param_loss
    )

    return {
        "loss": total,
        "recon": recon_loss,
        "kl": kl_loss,
        "param_mse": param_loss,
    }


def postprocess_image(
    img: np.ndarray,
    grad_thresh: float | None = None,
    grad_percentile: float = 90.0,
    boundary_width: int = 3,
    connectivity: int = 2,   # 1 = 4-connectivity, 2 = 8-connectivity
) -> np.ndarray:
    """
    Split the image into connected regions using high-gradient interfaces,
    and normalize each connected region to its mean value.
    """
    if img.ndim != 2:
        raise ValueError("img should be 2D")

    img_f = img.astype(np.float32)

    gx = ndi.sobel(img_f, axis=1, mode="reflect")
    gy = ndi.sobel(img_f, axis=0, mode="reflect")
    grad_mag = np.hypot(gx, gy)

    if grad_thresh is None:
        grad_thresh = float(np.percentile(grad_mag, grad_percentile))

    boundary = grad_mag >= grad_thresh

    if boundary_width > 0:
        boundary = ndi.binary_dilation(boundary, iterations=int(boundary_width))

    region_mask = ~boundary

    if connectivity == 1:
        structure = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            dtype=bool,
        )
    else:
        structure = np.ones((3, 3), dtype=bool)

    labels, num = ndi.label(region_mask, structure=structure)

    out = img_f.copy()

    for lab in range(1, num + 1):
        mask = labels == lab
        if mask.any():
            out[mask] = img_f[mask].mean()

    return out


# Optional backward-compatible alias.
# Prefer importing VAE_LinearRegression explicitly in new training scripts.
VAE_MDN = VAE_LinearRegression