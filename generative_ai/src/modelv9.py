import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataloader import smooth_scale


# ==========================================================
# 初始化函数
# ==========================================================
def init_weights(model):
    for _, m in model.named_modules():
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
# Encoder (NO conditioning on c)
# Downsampling 3 times: 128 -> 64 -> 32 -> 16
# ==========================================================
class ResEncoder(nn.Module):
    def __init__(self, img_size, in_channels, base_channels, latent_size):
        super().__init__()

        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),   # 64x64
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),  # 32x32
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),  # 16x16
        )

        conv_dim = int(img_size[0] * img_size[1] * base_channels * 4 / (8 * 8))
        self.fc_mu = nn.Linear(conv_dim, latent_size)
        self.fc_logvar = nn.Linear(conv_dim, latent_size)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# ==========================================================
# Decoder: ResUpBlock + GroupResUpBlock + ChannelWiseBlock
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
# VAE with:
#  - GMM posterior q(s,z|x)
#  - decoder p(x|z)
#  - regressor head to predict control params
#  (KL 不在模型内部计算，交给训练脚本)
# ==========================================================
class VAE(nn.Module):
    """
    forward 输出：
      recon, alpha, mu_q, logvar_q, ctr_pred, z
    """
    def __init__(
        self,
        image_size=(3, 128, 128),
        latent_size=64,
        hidden_dimension=64,
        num_params=15,
        ctr_head_hidden=256
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
            W=self.W
        )

        # 控制参数预测头：从 encoder pooled feature 回归到 num_params
        self.ctr_head = nn.Sequential(
            nn.Linear(latent_size, ctr_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ctr_head_hidden, ctr_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ctr_head_hidden, num_params),
        )

        init_weights(self)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # q(s|x), q(z|x,s)
        mu_q, logvar_q = self.encoder(x)  # feat_vec: [B, Cfeat]
        z = self.reparameterize(mu_q, logvar_q)         # [B,K,Z]

        # 控制参数预测
        ctr_pred = self.ctr_head(z)  # [B, num_params]
        recon = self.decoder(z)  # [B,C,H,W]

        return recon, mu_q, logvar_q, ctr_pred, z

    @torch.no_grad()
    def inference(self, num_samples=1, full_output=False):

        device = next(self.parameters()).device

        z = torch.randn(num_samples, self.latent_size, device=device)
        imgs = self.decoder(z)

        if full_output:
            ctr_pred = self.ctr_head(z)
            return imgs, ctr_pred, z
        else:
            return imgs