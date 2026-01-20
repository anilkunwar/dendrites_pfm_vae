# src/modelvae_mdn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataloader import smooth_scale


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
    """ Downsampling 3 times: 64 -> 32 -> 16 """

    def __init__(self, img_hw, in_channels, base_channels, latent_size):
        super().__init__()

        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),
        )

        # 与你原代码一致的 flatten 维度计算方式 :contentReference[oaicite:6]{index=6}
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

        self.fc_pi = nn.Linear(hidden_dim, self.K)               # [B, K]
        self.fc_mu = nn.Linear(hidden_dim, self.K * num_params)  # [B, K*P]
        self.fc_log_sigma = nn.Linear(hidden_dim, self.K * num_params)  # [B, K*P]

    def forward(self, z):
        h = self.backbone(z)
        pi_logits = self.fc_pi(h)
        pi = F.softmax(pi_logits, dim=-1)

        mu = self.fc_mu(h).view(-1, self.K, self.num_params)
        log_sigma = self.fc_log_sigma(h).view(-1, self.K, self.num_params)

        # 稳定性：限制 sigma 下界，避免数值爆炸/塌陷
        log_sigma = torch.clamp(log_sigma, min=-7.0, max=7.0)
        return pi, mu, log_sigma


def mdn_nll_loss(pi, mu, log_sigma, y, eps: float = 1e-9):
    """
    y: [B, P]
    pi: [B, K]
    mu/log_sigma: [B, K, P]
    return: scalar (mean over batch)
    """
    B, K, P = mu.shape
    y = y.unsqueeze(1).expand(B, K, P)  # [B, K, P]

    # log N(y | mu, sigma^2) for diagonal Gaussian
    # = -0.5 * [sum((y-mu)^2/sigma^2 + 2log(sigma) + log(2pi))]
    sigma = torch.exp(log_sigma) + eps
    log_prob = -0.5 * (
        torch.sum(((y - mu) / sigma) ** 2, dim=-1)
        + 2.0 * torch.sum(torch.log(sigma), dim=-1)
        + P * math.log(2.0 * math.pi)
    )  # [B, K]

    log_pi = torch.log(pi + eps)  # [B, K]
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # [B]
    nll = -torch.mean(log_mix)
    return nll


@torch.no_grad()
def mdn_point_and_confidence(pi, mu, log_sigma, var_scale: float = 1.0, topk: int = 3):
    """
    返回：
      theta_hat: [B, P]  (混合均值)
      conf_param: [B, P] (逐参数置信度，基于混合方差映射到 (0,1])
      conf_global: [B]   (基于权重熵的全局置信度)
      modes: dict: top-k 模式 (pi_k, mu_k) 便于分析
    """
    eps = 1e-12
    B, K, P = mu.shape
    pi_ = pi / (pi.sum(dim=-1, keepdim=True) + eps)

    # 混合均值
    theta_hat = torch.sum(pi_.unsqueeze(-1) * mu, dim=1)  # [B, P]

    # 混合方差：Var = E[sigma^2 + mu^2] - (E[mu])^2
    sigma2 = torch.exp(2.0 * log_sigma)
    e_mu2 = torch.sum(pi_.unsqueeze(-1) * (sigma2 + mu ** 2), dim=1)  # [B, P]
    var = torch.clamp(e_mu2 - theta_hat ** 2, min=0.0)

    # 逐参数置信度：exp(-Var / s)
    conf_param = torch.exp(-var / max(var_scale, eps))  # [B, P]

    # 全局置信度：exp(-H(pi))
    entropy = -torch.sum(pi_ * torch.log(pi_ + eps), dim=-1)  # [B]
    conf_global = torch.exp(-entropy)

    # top-k 模式
    k = min(topk, K)
    topv, topi = torch.topk(pi_, k=k, dim=-1)  # [B, k]
    top_mu = torch.gather(
        mu, 1, topi.unsqueeze(-1).expand(B, k, P)
    )  # [B, k, P]

    modes = {"pi_topk": topv, "mu_topk": top_mu, "idx_topk": topi}
    return theta_hat, conf_param, conf_global, modes


# ==========================================================
# VAE + MDN Regression Head
# ==========================================================
class VAE_MDN(nn.Module):
    """
    forward 输出：
      recon, mu_q, logvar_q, (pi, mu, log_sigma), z
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

        self.encoder = ResEncoder((image_size[1], image_size[2]), self.C, hidden_dimension, latent_size)

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_q, logvar_q = self.encoder(x)
        z = self.reparameterize(mu_q, logvar_q)

        pi, mu, log_sigma = self.mdn_head(z)
        recon = self.decoder(z)

        return recon, mu_q, logvar_q, (pi, mu, log_sigma), z

    @torch.no_grad()
    def inference(self, num_samples=1, full_output=False, var_scale: float = 1.0, topk: int = 3):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_size, device=device)
        imgs = self.decoder(z)

        pi, mu, log_sigma = self.mdn_head(z)
        theta_hat, conf_param, conf_global, modes = mdn_point_and_confidence(
            pi, mu, log_sigma, var_scale=var_scale, topk=topk
        )

        if full_output:
            return imgs, theta_hat, conf_param, conf_global, modes, z, (pi, mu, log_sigma)
        return imgs, theta_hat, conf_param, conf_global
