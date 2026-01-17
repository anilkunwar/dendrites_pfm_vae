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
# Encoder (3 次下采样: 128 -> 64 -> 32 -> 16)
# ==========================================================
class ResEncoder(nn.Module):
    def __init__(self, img_size, in_channels, base_channels, latent_size):
        """
        img_size: (H, W)
        in_channels: 图像通道数
        base_channels: 基础通道数
        latent_size: 潜空间维度
        """
        super().__init__()

        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),        # 128 -> 64
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True), # 64 -> 32
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),  # 32 -> 16
        )

        # 最后 feature map 尺度: H/8, W/8, 通道 base_channels*4
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
    通道独立卷积块（group conv）：每个组独立处理自己的特征
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
    需要 in/out 通道数都能被 groups 整除
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
    改进解码器：
      latent -> 16x16 -> 32 -> 64 -> 128
      支持通道独立 refinement
    """
    def __init__(self, out_channels, base_channels, latent_size, H, W):
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

        # 2.5) 通道数对齐到 groups 的整数倍
        ch_per_group = math.ceil(base_channels / out_channels)
        self.group_channels = ch_per_group * out_channels
        self.align_proj = nn.Conv2d(base_channels, self.group_channels, kernel_size=1, bias=False)
        self.align_bn = nn.BatchNorm2d(self.group_channels)

        # 3) 分组上采样：64→128
        self.group_up = GroupResUpBlock(
            in_channels=self.group_channels,
            out_channels=self.group_channels,
            groups=out_channels
        )

        # 4) 通道独立细化
        self.refine = nn.Sequential(
            ChannelWiseBlock(self.group_channels, groups=out_channels),
            ChannelWiseBlock(self.group_channels, groups=out_channels),
        )

        # 5) 输出头（分组1x1 conv）
        self.head = nn.Conv2d(
            self.group_channels, out_channels, kernel_size=1,
            groups=out_channels, bias=True
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.H // 8, self.W // 8)  # [B,4C,16,16]

        x = self.shared_up(x)                                # [B,C,64,64]
        x = self.align_bn(self.align_proj(x))                # [B,G*CpG,64,64]
        x = self.group_up(x)                                 # [B,G*CpG,128,128]
        x = self.refine(x)                                   # [B,G*CpG,128,128]
        x = self.head(x)                                     # [B,out_channels,128,128]

        return smooth_scale(x)


# ==========================================================
# 条件高斯混合先验：p(z|c) = Σ π_k(c) N(μ_k(c), σ_k²(c)I)
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
        """
        c: [B, cond_dim]
        返回:
            pis: [B, K]
            mus: [B, K, latent_dim]
            logvars: [B, K, latent_dim]
        """
        h = self.fc(c)
        pis = torch.softmax(self.fc_pi(h), dim=-1)  # [B,K]

        mus = self.fc_mu(h).view(-1, self.n_components, self.latent_dim)
        logvars = self.fc_logvar(h).view(-1, self.n_components, self.latent_dim)

        return pis, mus, logvars


# ==========================================================
# KL 相关辅助函数
# ==========================================================
def kl_gaussian_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL( N(mu_q, logvar_q) || N(mu_p, logvar_p) )
    支持 [B,1,Z] vs [B,K,Z]
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * torch.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0,
        dim=-1
    )


def kl_q_vs_gmm(mu_q, logvar_q, pis, mus, logvars):
    """
    KL(q || GMM) ≈ -log( sum_k pi_k * exp( -KL(q||N_k) ) )
    mu_q:    [B, Z]
    logvar_q:[B, Z]
    pis:     [B, K]
    mus:     [B, K, Z]
    logvars: [B, K, Z]
    返回:    [B]
    """
    # 扩展 q： [B,Z] -> [B,1,Z]
    mu_q_expand = mu_q.unsqueeze(1)
    logvar_q_expand = logvar_q.unsqueeze(1)

    # 对每个 component 计算 KL(q||N_k): [B,K]
    kl_comp = kl_gaussian_gaussian(mu_q_expand, logvar_q_expand, mus, logvars)

    # log( sum_k pi_k * exp( - KL(q||N_k) ) )
    log_pis = torch.log(pis + 1e-9)
    log_sum = torch.logsumexp(log_pis - kl_comp, dim=1)  # [B]

    return -log_sum  # 近似的 KL(q||GMM)


# ==========================================================
# 完整 VAE 模型：Encoder + Conditional GMM Prior + Decoder
# ==========================================================
class VAE(nn.Module):
    """
    条件高斯混合先验 VAE:
      - q(z|x) 由 encoder 得到
      - p(z|c) = Σ π_k(c) N(μ_k(c), σ_k²(c)I)
      - 解码器只看 z，不直接看 c
      - forward 返回 recon, mu_q, logvar_q, kl_qp
    """
    def __init__(self, image_size=(3, 128, 128), latent_size=64,
                 hidden_dimension=64, num_params=0, n_components=3):
        super().__init__()
        self.C, self.H, self.W = image_size
        self.latent_size = latent_size
        self.num_params = num_params
        self.n_components = n_components

        # Encoder: q(z|x)
        self.encoder = ResEncoder(
            (image_size[1], image_size[2]),
            in_channels=self.C,
            base_channels=hidden_dimension,
            latent_size=latent_size
        )

        # Conditional prior: p(z|c)
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

    def forward(self, x, ctr=None):
        """
        x:   [B,C,H,W]
        ctr: [B,num_params]，条件（比如材料成分/工艺参数/目标属性等）
        返回:
            recon: [B,C,H,W]
            mu_q, logvar_q: 后验参数
            kl: [B]，每个样本的 KL(q||p)
        """
        B = x.size(0)

        # 1) 编码 q(z|x)
        mu_q, logvar_q = self.encoder(x)
        z = self.reparameterize(mu_q, logvar_q)

        if ctr is not None:
            # 2) 条件先验 p(z|c)
            pis, mus, logvars = self.prior(ctr)

            # 3) KL(q||p)
            kl = kl_q_vs_gmm(mu_q, logvar_q, pis, mus, logvars)  # [B]
        else:
            kl = 0.

        # 4) 解码
        recon = self.decoder(z)
        return recon, mu_q, logvar_q, kl, z

    @torch.no_grad()
    def inference(self, ctr, num_samples_per_cond=1, use_component_mean=False):
        """
        条件生成：
          ctr: [B,num_params]
        返回:
          imgs: [B * num_samples_per_cond, C, H, W]
        """
        self.eval()
        device = next(self.parameters()).device
        ctr = ctr.to(device)
        B = ctr.size(0)

        pis, mus, logvars = self.prior(ctr)  # [B,K,Z]

        z_list = []

        for i in range(B):
            for _ in range(num_samples_per_cond):
                if use_component_mean:
                    # 直接取每个 component 的均值（可以改成循环 K 分量）
                    k = torch.multinomial(pis[i], 1).item()
                    z_i = mus[i, k]
                else:
                    # GMM 采样：先选 component，再从该成分采样
                    k = torch.multinomial(pis[i], 1).item()
                    mu_k = mus[i, k]
                    std_k = torch.exp(0.5 * logvars[i, k])
                    eps = torch.randn_like(std_k)
                    z_i = mu_k + eps * std_k

                z_list.append(z_i)

        z_all = torch.stack(z_list, dim=0)  # [B*num_samples_per_cond,Z]
        imgs = self.decoder(z_all)
        return imgs


# ==========================================================
# 一个简单的 train_step 示例
# ==========================================================
def train_step(model, optimizer, x, ctr, beta_kl=1.0, recon_mode="mse"):
    """
    model: VAE
    x:     [B,C,H,W]
    ctr:   [B,num_params]
    beta_kl: KL 权重
    recon_mode: "mse" 或 "bce"
    """
    model.train()
    optimizer.zero_grad()

    recon, mu_q, logvar_q, kl = model(x, ctr)

    if recon_mode == "mse":
        recon_loss = F.mse_loss(recon, x, reduction="mean")
    else:
        # 假设 x 归一化到 [0,1]
        recon_loss = F.binary_cross_entropy(
            recon, x, reduction="mean"
        )

    kl_loss = kl.mean()

    loss = recon_loss + beta_kl * kl_loss
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
    }


# ==========================================================
# Quick Test
# ==========================================================
if __name__ == "__main__":
    # 假设图像为 3x128x128，条件向量 dim=15
    model = VAE(
        image_size=(3, 128, 128),
        latent_size=32,
        hidden_dimension=64,
        num_params=15,
        n_components=3
    )

    x = torch.randn(4, 3, 128, 128)
    c = torch.randn(4, 15)

    out = train_step(model, torch.optim.Adam(model.parameters(), lr=1e-3), x, c)
    print("train_step:", out)

    with torch.no_grad():
        imgs = model.inference(c, num_samples_per_cond=2)
        print("Generated imgs:", imgs.shape)
