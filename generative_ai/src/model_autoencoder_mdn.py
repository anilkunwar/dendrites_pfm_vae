# src/model_autoencoder_mdn.py
"""
AutoEncoder + MDN regression version converted from the original VAE_MDN model.

Main changes vs VAE_MDN:
1. Encoder outputs a deterministic latent vector z instead of (mu, logvar).
2. No reparameterization trick and no KL loss are needed.
3. The MDN head is kept, so parameter prediction and confidence estimation
   stay compatible with the original MDN workflow.

Typical training loss:
    loss = recon_loss(recon, x) + lambda_mdn * mdn_nll_loss(mdn_out, theta)
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


class ResEncoderAE(nn.Module):
    """Deterministic encoder: image -> z."""

    def __init__(self, img_hw, in_channels, base_channels, latent_size):
        super().__init__()

        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),
        )

        conv_dim = int(img_hw[0] * img_hw[1] * base_channels * 4 / (8 * 8))
        self.fc_z = nn.Linear(conv_dim, latent_size)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        z = self.fc_z(x)
        return z


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
# MDN Head: p(theta | z) = sum_k pi_k N(mu_k, sigma_k^2)
# ==========================================================
class MDNHead(nn.Module):
    def __init__(self, in_dim: int, num_params: int, num_components: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.num_params = num_params
        self.K = num_components

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.fc_pi = nn.Linear(hidden_dim, self.K)                    # [B, K]
        self.fc_mu = nn.Linear(hidden_dim, self.K * num_params)       # [B, K*P]
        self.fc_log_sigma = nn.Linear(hidden_dim, self.K * num_params)  # [B, K*P]

    def forward(self, z):
        h = self.backbone(z)
        pi_logits = self.fc_pi(h)
        pi = F.softmax(pi_logits, dim=-1)

        mu = self.fc_mu(h).view(-1, self.K, self.num_params)
        log_sigma = self.fc_log_sigma(h).view(-1, self.K, self.num_params)
        log_sigma = torch.clamp(log_sigma, min=-7.0, max=7.0)
        return pi, mu, log_sigma


@torch.no_grad()
def mdn_point_and_confidence(pi, mu, log_sigma, var_scale: float = 1.0, topk: int = 3):
    """
    Returns:
      theta_hat: [B, P]  mixture mean
      conf_param: [B, P] per-parameter confidence based on mixture variance
      conf_global: [B]   global confidence based on mixture-weight entropy
      modes: dict        top-k modes for analysis
    """
    eps = 1e-12
    B, K, P = mu.shape
    pi_ = pi / (pi.sum(dim=-1, keepdim=True) + eps)

    theta_hat = torch.sum(pi_.unsqueeze(-1) * mu, dim=1)  # [B, P]

    sigma2 = torch.exp(2.0 * log_sigma)
    e_mu2 = torch.sum(pi_.unsqueeze(-1) * (sigma2 + mu ** 2), dim=1)
    var = torch.clamp(e_mu2 - theta_hat ** 2, min=0.0)

    conf_param = torch.exp(-var / max(var_scale, eps))

    entropy = -torch.sum(pi_ * torch.log(pi_ + eps), dim=-1)
    conf_global = torch.exp(-entropy)

    k = min(topk, K)
    topv, topi = torch.topk(pi_, k=k, dim=-1)
    top_mu = torch.gather(mu, 1, topi.unsqueeze(-1).expand(B, k, P))

    modes = {"pi_topk": topv, "mu_topk": top_mu, "idx_topk": topi}
    return theta_hat, conf_param, conf_global, modes


def mdn_nll_loss(pi, mu, log_sigma, target, reduction: str = "mean"):
    """
    Negative log-likelihood loss for MDN regression.

    Args:
        pi: [B, K]
        mu: [B, K, P]
        log_sigma: [B, K, P]
        target: [B, P]
    """
    target = target.unsqueeze(1)  # [B, 1, P]
    inv_sigma = torch.exp(-log_sigma)
    log_prob = -0.5 * ((target - mu) * inv_sigma) ** 2 - log_sigma - 0.5 * math.log(2.0 * math.pi)
    log_prob = log_prob.sum(dim=-1)  # [B, K]
    log_mix_prob = torch.log(pi + 1e-12) + log_prob
    nll = -torch.logsumexp(log_mix_prob, dim=-1)  # [B]

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError(f"Unsupported reduction: {reduction}")


# ==========================================================
# AutoEncoder + MDN Regression Head
# ==========================================================
class AE_MDN(nn.Module):
    """
    forward output:
      recon, z, (pi, mu, log_sigma)
    """

    def __init__(
        self,
        image_size=(3, 128, 128),
        latent_size=64,
        hidden_dimension=64,
        num_params=15,
        mdn_components=8,
        mdn_hidden=256,
    ):
        super().__init__()
        self.C, self.H, self.W = image_size
        self.latent_size = latent_size
        self.num_params = num_params

        self.encoder = ResEncoderAE((image_size[1], image_size[2]), self.C, hidden_dimension, latent_size)

        self.decoder = ResDecoder(
            out_channels=self.C,
            base_channels=hidden_dimension,
            latent_size=latent_size,
            H=self.H,
            W=self.W,
        )

        self.mdn_head = MDNHead(
            in_dim=latent_size,
            num_params=num_params,
            num_components=mdn_components,
            hidden_dim=mdn_hidden,
        )

        init_weights(self)

    def forward(self, x):
        z = self.encoder(x)
        pi, mu, log_sigma = self.mdn_head(z)
        recon = self.decoder(z)
        return recon, z, (pi, mu, log_sigma)

    @torch.no_grad()
    def inference_from_image(self, x, var_scale: float = 1.0, topk: int = 3):
        z = self.encoder(x)
        return self.inference_from_z(z, var_scale=var_scale, topk=topk)

    @torch.no_grad()
    def inference_from_z(self, z, var_scale: float = 1.0, topk: int = 3):
        recon = self.decoder(z)
        pi, mu, log_sigma = self.mdn_head(z)
        theta_hat, conf_param, conf_global, modes = mdn_point_and_confidence(
            pi, mu, log_sigma, var_scale=var_scale, topk=topk
        )
        return recon, (theta_hat, conf_param, conf_global, modes)


# Backward-compatible alias if your training script expects a model class name.
AutoEncoder_MDN = AE_MDN


def postprocess_image(
    img: np.ndarray,
    grad_thresh: float | None = None,
    grad_percentile: float = 90.0,
    boundary_width: int = 3,
    connectivity: int = 2,
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
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
    else:
        structure = np.ones((3, 3), dtype=bool)

    labels, num = ndi.label(region_mask, structure=structure)

    out = img_f.copy()
    for lab in range(1, num + 1):
        mask = labels == lab
        if mask.any():
            out[mask] = img_f[mask].mean()

    return out
