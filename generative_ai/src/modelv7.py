import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataloader import smooth_scale


# ==========================================================
# 初始化函数
# ==========================================================
def init_weights(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ==========================================================
# Multi-kernel Residual Block (Encoder)
# ==========================================================
class MultiKernelResBlock(nn.Module):
    """
    多尺度卷积残差块：3x3, 5x5, 7x7 并联卷积
    """
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
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)

        out = (out3 + out5 + out7) / 3.0
        out = self.bn(out)
        out = out + self.shortcut(x)
        return self.relu(out)


# ==========================================================
# Encoder (Conv-only) producing q(s|x) and q(z|x,s)
# ==========================================================
class ResEncoderGMM(nn.Module):
    """
    全卷积后验编码器：
      - backbone 输出 feature map: [B, Cfeat, H/8, W/8]
      - 1x1 conv 头输出:
          logits_s: [B, K]  -> q(s|x)
          mu_q:     [B, K, Z]
          logvar_q: [B, K, Z]
    """
    def __init__(self, in_channels, base_channels, latent_size, n_components=3):
        super().__init__()
        self.latent_size = latent_size
        self.n_components = n_components

        self.backbone = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),             # 128 -> 64
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),      # 64 -> 32
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),  # 32 -> 16
        )

        feat_ch = base_channels * 4

        # 1x1 conv heads (no FC)
        self.conv_logits = nn.Conv2d(feat_ch, n_components, kernel_size=1, bias=True)
        self.conv_mu = nn.Conv2d(feat_ch, n_components * latent_size, kernel_size=1, bias=True)
        self.conv_logvar = nn.Conv2d(feat_ch, n_components * latent_size, kernel_size=1, bias=True)

        # global pooling to turn feature maps into vectors
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        f = self.backbone(x)  # [B, feat_ch, H/8, W/8]

        logits_map = self.conv_logits(f)   # [B, K, h, w]
        mu_map = self.conv_mu(f)           # [B, K*Z, h, w]
        logvar_map = self.conv_logvar(f)   # [B, K*Z, h, w]

        # pool to [B, *, 1, 1] then squeeze
        logits = self.pool(logits_map).squeeze(-1).squeeze(-1)  # [B, K]
        mu = self.pool(mu_map).squeeze(-1).squeeze(-1)          # [B, K*Z]
        logvar = self.pool(logvar_map).squeeze(-1).squeeze(-1)  # [B, K*Z]

        mu = mu.view(-1, self.n_components, self.latent_size)         # [B, K, Z]
        logvar = logvar.view(-1, self.n_components, self.latent_size) # [B, K, Z]

        return logits, mu, logvar


# ==========================================================
# Decoder 部分：ResUpBlock + GroupResUpBlock + ChannelWiseBlock
# ==========================================================
class ResUpBlock(nn.Module):
    """
    残差上采样块：ConvTranspose2d + Conv2d
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels)
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
    """
    通道独立卷积块（group conv）
    """
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
    """
    分组残差上采样块
    """
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1,
            groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.short = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
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
    """
    解码器：latent -> 16 -> 32 -> 64 -> 128
    """
    def __init__(self, out_channels, base_channels, latent_size, H, W):
        super().__init__()
        self.H, self.W = H, W
        self.out_channels = out_channels
        init_h, init_w = H // 8, W // 8

        self.fc = nn.Linear(latent_size, base_channels * 4 * init_h * init_w)

        self.shared_up = nn.Sequential(
            ResUpBlock(base_channels * 4, base_channels * 2),  # 16->32
            ResUpBlock(base_channels * 2, base_channels),      # 32->64
        )

        ch_per_group = math.ceil(base_channels / out_channels)
        self.group_channels = ch_per_group * out_channels
        self.align_proj = nn.Conv2d(base_channels, self.group_channels, kernel_size=1, bias=False)
        self.align_bn = nn.BatchNorm2d(self.group_channels)

        self.group_up = GroupResUpBlock(
            in_channels=self.group_channels,
            out_channels=self.group_channels,
            groups=out_channels
        )

        self.refine = nn.Sequential(
            ChannelWiseBlock(self.group_channels, groups=out_channels),
            ChannelWiseBlock(self.group_channels, groups=out_channels),
        )

        self.head = nn.Conv2d(
            self.group_channels, out_channels, kernel_size=1,
            groups=out_channels, bias=True
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
# 条件高斯混合先验：p(s|c)=pi(c), p(z|s,c)=N(mu_s(c), var_s(c))
# ==========================================================
class ConditionalGMM(nn.Module):
    def __init__(self, cond_dim, latent_dim, n_components=3, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components

        self.fc = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components * latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, n_components * latent_dim)

    def forward(self, c):
        h = self.fc(c)
        pis = torch.softmax(self.fc_pi(h), dim=-1)  # [B,K]
        mus = self.fc_mu(h).view(-1, self.n_components, self.latent_dim)
        logvars = self.fc_logvar(h).view(-1, self.n_components, self.latent_dim)
        return pis, mus, logvars


# ==========================================================
# KL 辅助函数
# ==========================================================
def kl_gaussian_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL( N(mu_q, logvar_q) || N(mu_p, logvar_p) )
    mu_*: [B, Z] or [B, K, Z]
    logvar_* same shape
    return: [B] or [B, K]
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * torch.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-12) - 1.0,
        dim=-1
    )


def kl_categorical(alpha, pi, eps=1e-9):
    """
    KL( Cat(alpha) || Cat(pi) )
    alpha, pi: [B, K]
    return: [B]
    """
    alpha = torch.clamp(alpha, min=eps, max=1.0)
    pi = torch.clamp(pi, min=eps, max=1.0)
    return torch.sum(alpha * (torch.log(alpha) - torch.log(pi)), dim=-1)


# ==========================================================
# 完整 VAE：q(s,z|x) + p(s,z|c) + decoder p(x|z)
# ==========================================================
class VAE(nn.Module):
    """
    改进版（联合离散变量分解）：
      - Encoder 输出 q(s|x) 和 q(z|x,s) 的每个组分参数
      - Prior 输出 p(s|c) 和 p(z|c,s)
      - KL 解析分解：
          KL(q(s|x)||p(s|c)) + E_{q(s|x)} KL(q(z|x,s)||p(z|c,s))
      - 为保持可微，forward 中用 soft 混合得到 z：
          z = sum_s alpha_s * z_s
    """
    def __init__(self, image_size=(3, 128, 128), latent_size=64,
                 hidden_dimension=64, num_params=0, n_components=3):
        super().__init__()
        self.C, self.H, self.W = image_size
        self.latent_size = latent_size
        self.num_params = num_params
        self.n_components = n_components

        # Encoder: q(s|x), q(z|x,s)
        self.encoder = ResEncoderGMM(
            in_channels=self.C,
            base_channels=hidden_dimension,
            latent_size=latent_size,
            n_components=n_components
        )

        # Prior: p(s|c), p(z|c,s)
        self.prior = ConditionalGMM(
            cond_dim=num_params,
            latent_dim=latent_size,
            n_components=n_components,
            hidden_dim=hidden_dimension
        )

        # Decoder: p(x|z)
        self.decoder = ResDecoder(
            out_channels=self.C,
            base_channels=hidden_dimension,
            latent_size=latent_size,
            H=self.H,
            W=self.W
        )

        init_weights(self)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, ctr):
        """
        x:   [B,C,H,W]
        ctr: [B,num_params]
        return:
          recon: [B,C,H,W]
          alpha: [B,K]          (q(s|x))
          mu_q, logvar_q: [B,K,Z]
          kl: [B]
        """
        # 1) q(s|x), q(z|x,s)
        logits_s, mu_q, logvar_q = self.encoder(x)     # logits: [B,K], mu/logvar: [B,K,Z]
        alpha = torch.softmax(logits_s, dim=-1)        # [B,K]

        # 2) p(s|c), p(z|c,s)
        pi, mu_p, logvar_p = self.prior(ctr)          # pi: [B,K], mu_p/logvar_p: [B,K,Z]

        # 3) KL 分解
        kl_s = kl_categorical(alpha, pi)              # [B]
        kl_z_each = kl_gaussian_gaussian(mu_q, logvar_q, mu_p, logvar_p)  # [B,K]
        kl_z = torch.sum(alpha * kl_z_each, dim=-1)   # [B]
        kl = kl_s + kl_z                              # [B]

        # 4) 采样 z（soft mixture，保持可微）
        z_each = self.reparameterize(mu_q, logvar_q)  # [B,K,Z]
        z = torch.sum(alpha.unsqueeze(-1) * z_each, dim=1)  # [B,Z]

        # 5) 解码
        recon = self.decoder(z)
        return recon, alpha, mu_q, logvar_q, kl

    @torch.no_grad()
    def inference(self, ctr, num_samples_per_cond=1, use_component_mean=False):
        """
        条件生成（从 prior 采样）：
          ctr: [B,num_params]
        返回:
          imgs: [B * num_samples_per_cond, C, H, W]
        """
        self.eval()
        device = next(self.parameters()).device
        ctr = ctr.to(device)
        B = ctr.size(0)

        pi, mu_p, logvar_p = self.prior(ctr)  # [B,K,Z]

        z_list = []
        for i in range(B):
            for _ in range(num_samples_per_cond):
                k = torch.multinomial(pi[i], 1).item()
                if use_component_mean:
                    z_i = mu_p[i, k]
                else:
                    std_k = torch.exp(0.5 * logvar_p[i, k])
                    eps = torch.randn_like(std_k)
                    z_i = mu_p[i, k] + eps * std_k
                z_list.append(z_i)

        z_all = torch.stack(z_list, dim=0)  # [B*num_samples_per_cond,Z]
        imgs = self.decoder(z_all)
        return imgs


# ==========================================================
# Quick Test
# ==========================================================
if __name__ == "__main__":
    model = VAE(
        image_size=(3, 128, 128),
        latent_size=32,
        hidden_dimension=64,
        num_params=15,
        n_components=3
    )

    x = torch.randn(4, 3, 128, 128)
    c = torch.randn(4, 15)

    model(x, c)
