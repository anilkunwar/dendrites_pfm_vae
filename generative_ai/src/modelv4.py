import math

import torch
import torch.nn as nn

from generative_ai.src.dataloader import smooth_scale

# 定义初始化函数
def init_weights(model):
    for name, m in model.named_modules():  # 递归遍历所有模块
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# ==========================================================
# Multi-kernel Residual Block
# ==========================================================
class MultiKernelResBlock(nn.Module):
    """
    Residual block with multiple convolutional kernel sizes (3x3, 5x5, 7x7)
    for multi-scale feature extraction.
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        # Parallel convolutions with different receptive fields
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, stride, 3, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Multi-kernel feature extraction
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)

        # Aggregate + residual connection
        out = (out3 + out5 + out7) / 3.0
        out = self.bn(out)
        out += self.shortcut(x)
        return self.relu(out)


# ==========================================================
# Encoder (NO conditioning on c)
# Downsampling 3 times: 128 -> 64 -> 32 -> 16
# ==========================================================
class ResEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, latent_size):
        super().__init__()

        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),   # 64x64
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),  # 32x32
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),  # 16x16
        )

        # Infer flattened dimension
        dummy = torch.zeros(1, in_channels, 128, 128)
        conv_out = self.layers(dummy)
        conv_dim = conv_out.view(1, -1).shape[1]

        self.fc_mu = nn.Linear(conv_dim, latent_size)
        self.fc_logvar = nn.Linear(conv_dim, latent_size)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# ==========================================================
# Residual Upsampling Block (for decoder)
# ==========================================================
class ResUpBlock(nn.Module):
    """
    Residual block for upsampling in decoder using ConvTranspose2d.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 主路径：上采样 + 卷积
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut（调整通道 + 上采样）
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

        shortcut = self.shortcut(x)
        out += shortcut
        return self.relu(out)


class ChannelWiseBlock(nn.Module):
    """
    通道独立卷积块（group conv），每个组独立处理自己的特征，不发生跨通道串扰。
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
    分组残差上采样块：ConvTranspose2d + group conv，使上采样阶段就实现通道独立。
    需要 in/out 通道数都能被 groups 整除。
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

        # 残差支路：分组反卷积保持通道独立
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
    改进解码器（满足 16→32→64→128 三次上采样 & 通道独立）：
      1) 共享 up: 16→32→64
      2) 通道对齐投影（把通道数调到 groups 的整数倍）
      3) 分组上采样: 64→128（GroupResUpBlock）
      4) 通道独立细化: ChannelWiseBlock × 2
      5) 分组 1×1 输出到各通道
    """
    def __init__(self, out_channels, base_channels, latent_size, H, W, num_params=0):
        super().__init__()
        self.H, self.W = H, W
        self.out_channels = out_channels
        init_h, init_w = H // 8, W // 8  # 128→16

        # 1) latent → feature map
        self.fc = nn.Linear(latent_size, base_channels * 4 * init_h * init_w)

        # 2) 共享上采样骨干：16→32→64
        self.shared_up = nn.Sequential(
            ResUpBlock(base_channels * 4, base_channels * 2),  # 16→32
            ResUpBlock(base_channels * 2, base_channels),      # 32→64
        )

        # 2.5) 通道数对齐到 groups 的整数倍（为后续分组层做准备）
        #     例如 base_channels=256, out_channels=3，则对齐到 ceil(256/3)*3 = 258
        #     这样就不会因为不可整除而报错
        ch_per_group = math.ceil(base_channels / out_channels)
        self.group_channels = ch_per_group * out_channels
        self.align_proj = nn.Conv2d(base_channels, self.group_channels, kernel_size=1, bias=False)
        self.align_bn = nn.BatchNorm2d(self.group_channels)

        # 3) 分组上采样：64→128（在上采样阶段就分离通道）
        self.group_up = GroupResUpBlock(
            in_channels=self.group_channels,
            out_channels=self.group_channels,
            groups=out_channels
        )  # 64→128

        # 4) 通道独立细化
        self.refine = nn.Sequential(
            ChannelWiseBlock(self.group_channels, groups=out_channels),
            ChannelWiseBlock(self.group_channels, groups=out_channels),
        )

        # 5) 分组 1×1 输出头：grouped 1×1 把每个组的特征聚合到各自的 1 个输出通道
        self.head = nn.Conv2d(self.group_channels, out_channels, kernel_size=1, groups=out_channels, bias=True)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.H // 8, self.W // 8)  # [B, 4C, 16, 16]

        x = self.shared_up(x)                                # [B, C, 64, 64]
        x = self.align_bn(self.align_proj(x))                # [B, G*CpG, 64, 64]
        x = self.group_up(x)                                 # [B, G*CpG, 128, 128]
        x = self.refine(x)                                   # [B, G*CpG, 128, 128]
        x = self.head(x)                                     # [B, out_channels, 128, 128]

        return smooth_scale(x)

# ==========================================================
# Full Model
# ==========================================================
class VAE(nn.Module):
    """
    VAE with:
      - Multi-kernel residual encoder
      - Additive coupling between z and c in the decoder
      - Coordinate channels concatenated to input
      - Downsampling 8x (128 -> 16)
    """
    def __init__(self, image_size=(3, 128, 128), latent_size=64, hidden_dimension=64, num_params=0):
        super().__init__()
        self.C, self.H, self.W = image_size
        self.latent_size = latent_size

        # Encoder does NOT take c
        self.encoder = ResEncoder(self.C, hidden_dimension, latent_size)

        # Project c to same dimension as z
        self.cMLP = nn.Sequential(
            nn.Linear(num_params, latent_size*2),
            nn.ReLU(inplace=True)
        )
        self.zMLP = nn.Sequential(
            nn.Linear(num_params, latent_size),
            nn.ReLU()
        )
        self.zAttn = nn.Linear(latent_size*2, latent_size)

        # Decoder takes z (+ c additively)
        self.decoder = ResDecoder(
            self.C, hidden_dimension, latent_size, self.H, self.W
        )

        init_weights(self)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, ctr=None):
        B, _, H, W = x.shape

        mu, logvar = self.encoder(x)

        # Additive coupling: z' = z + f(c)
        if ctr is not None:
            c = self.cMLP(ctr).view(B, self.latent_size, 2)
            mu_ = mu + c[..., 0]
            logvar_ = logvar + c[..., 1]
        else:
            mu_ = mu
            logvar_ = logvar
        z = self.reparameterize(mu_, logvar_)
        if ctr is not None:
            z = self.zAttn(torch.cat([z, self.zMLP(ctr)], dim=-1))

        recon = self.decoder(z)
        return recon, mu, logvar, z

    @torch.no_grad()
    def inference(self, ctr: torch.Tensor = None, num_samples: int = 1, device=None):
        """
        推理（生成）函数：
          - 若提供 ctr（条件向量），则生成条件图像
          - 若不提供 ctr，则随机采样生成图像

        参数:
            ctr: torch.Tensor [B, num_params]，可选的控制向量
            num_samples: int，若无条件输入时生成的样本数
            device: torch.device，可选

        返回:
            recon: [B, C, H, W] 生成图像
            mu, logvar, z: 推理时的隐变量统计（主要便于一致接口）
        """
        device = device or next(self.parameters()).device
        self.eval()

        if ctr is not None:
            # 条件生成
            B = ctr.size(0)
            ctr = ctr.to(device)

            # 条件映射
            c = self.cMLP(ctr).view(B, self.latent_size, 2)
            mu = c[..., 0]
            logvar = c[..., 1]

            # 采样
            z = self.reparameterize(mu, logvar)
            if ctr is not None:
                z = self.zAttn(torch.cat([z, self.zMLP(ctr)], dim=-1))

        else:
            # 无条件生成（随机采样）
            B = num_samples
            z = torch.randn(B, self.latent_size, device=device)

        # 解码生成图像
        recon = self.decoder(z)
        return recon

# ==========================================================
# Quick Test
# ==========================================================
if __name__ == "__main__":
    model = VAE(
        image_size=(3, 128, 128),
        latent_size=32,
        hidden_dimension=256,
        num_params=15
    )

    x = torch.randn(2, 3, 128, 128)
    c = torch.randn(2, 15)
    recon, mu, logvar, z = model(x, c)

    print("Input:", x.shape)
    print("Reconstruction:", recon.shape)
    print("Latent:", z.shape)
    print("Total parameters: %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
