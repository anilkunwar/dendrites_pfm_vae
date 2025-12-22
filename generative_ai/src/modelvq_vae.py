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

# ==========================================================
# Vector Quantizer (EMA) for VQ-VAE
# ==========================================================
class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE quantizer with optional EMA updates.

    Inputs:
        z_e: [B, D]
    Outputs:
        z_q_st: [B, D]  (straight-through)
        vq_loss: scalar tensor
        perplexity: scalar tensor
        encoding_indices: [B]
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        use_ema: bool = True
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.eps = float(eps)
        self.use_ema = bool(use_ema)

        init_embed = torch.randn(self.num_embeddings, self.embedding_dim)

        if self.use_ema:
            self.register_buffer("embedding", init_embed)
            self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
            self.register_buffer("ema_w", init_embed.clone())
        else:
            self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
            self.codebook.weight.data.copy_(init_embed)

    def _codebook(self):
        return self.embedding if self.use_ema else self.codebook.weight

    def forward(self, z_e: torch.Tensor):
        if z_e.dim() != 2 or z_e.size(-1) != self.embedding_dim:
            raise ValueError(f"Expected z_e [B,{self.embedding_dim}], got {tuple(z_e.shape)}")

        codebook = self._codebook()  # [K, D]
        z = z_e

        # distances: [B, K] = ||z||^2 + ||e||^2 - 2 z·e
        z_sq = torch.sum(z ** 2, dim=1, keepdim=True)          # [B, 1]
        e_sq = torch.sum(codebook ** 2, dim=1).unsqueeze(0)    # [1, K]
        ze = torch.matmul(z, codebook.t())                     # [B, K]
        distances = z_sq + e_sq - 2.0 * ze

        encoding_indices = torch.argmin(distances, dim=1)      # [B]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type_as(z)  # [B, K]

        if self.use_ema:
            z_q = torch.matmul(encodings, self.embedding)      # [B, D]
        else:
            z_q = self.codebook(encoding_indices)              # [B, D]

        # straight-through
        z_q_st = z + (z_q - z).detach()

        # VQ loss
        if self.use_ema:
            # only commitment loss; codebook updated via EMA
            vq_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        else:
            codebook_loss = F.mse_loss(z_q, z.detach())
            commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
            vq_loss = codebook_loss + commitment_loss

        # EMA updates
        if self.training and self.use_ema:
            with torch.no_grad():
                cluster_size = torch.sum(encodings, dim=0)  # [K]
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=(1.0 - self.decay))

                dw = torch.matmul(encodings.t(), z)         # [K, D]
                self.ema_w.mul_(self.decay).add_(dw, alpha=(1.0 - self.decay))

                n = torch.sum(self.ema_cluster_size)
                # Laplace smoothing
                cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
                new_embed = self.ema_w / cluster_size.unsqueeze(1)
                self.embedding.copy_(new_embed)

        # perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, vq_loss, perplexity, encoding_indices


# ==========================================================
# Encoder for VQ-VAE (vector latent)
# ==========================================================
class ResEncoderVQ(nn.Module):
    """
    Same backbone as ResEncoderGMM (downsample 3 times), but output a single latent vector z: [B, Z].
    """
    def __init__(self, in_channels, base_channels, latent_size):
        super().__init__()
        self.latent_size = latent_size

        self.backbone = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),             # -> /2
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),      # -> /4
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),  # -> /8
        )

        feat_ch = base_channels * 4
        self.conv_z = nn.Conv2d(feat_ch, latent_size, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.backbone(x)
        z_map = self.conv_z(f)
        z = self.pool(z_map).squeeze(-1).squeeze(-1)
        return z


# ==========================================================
# Condition encoder: y -> latent vector (shared space with image encoder)
# ==========================================================
class CondEncoder(nn.Module):
    """
    Map condition vector y to the SAME latent space as the image encoder.
    Output: z_y [B, Z]
    """
    def __init__(self, num_params: int, latent_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_params, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_size),
        )

    def forward(self, y):
        return self.net(y)


# ==========================================================
# VQ-VAE with explicit latent alignment objective
# ==========================================================
class VQVAE(nn.Module):
    """
    z_x = Enc_img(x) and z_y = Enc_cond(y) are trained to be consistent.

    Returns:
      recon, vq_loss, perplexity, indices, z_x, z_y
    """
    def __init__(
        self,
        image_size=(3, 128, 128),
        latent_size=64,
        hidden_dimension=64,
        num_params=0,
        codebook_size=512,
        commitment_cost=0.25,
        ema_decay=0.99,
        use_ema=True,
        cond_hidden=128,
        l2_normalize_latent=False,
    ):
        super().__init__()
        self.C, self.H, self.W = image_size
        self.latent_size = latent_size
        self.num_params = num_params
        self.l2_normalize_latent = bool(l2_normalize_latent)

        self.img_encoder = ResEncoderVQ(
            in_channels=self.C,
            base_channels=hidden_dimension,
            latent_size=latent_size
        )

        if num_params <= 0:
            raise ValueError("VQVAE(alignment) requires num_params > 0.")
        self.cond_encoder = CondEncoder(num_params=num_params, latent_size=latent_size, hidden=cond_hidden)

        self.quantizer = VectorQuantizerEMA(
            num_embeddings=codebook_size,
            embedding_dim=latent_size,
            commitment_cost=commitment_cost,
            decay=ema_decay,
            use_ema=use_ema
        )

        self.decoder = ResDecoder(
            out_channels=self.C,
            base_channels=hidden_dimension,
            latent_size=latent_size,
            H=self.H,
            W=self.W
        )

        init_weights(self)

    def _maybe_norm(self, z: torch.Tensor):
        if not self.l2_normalize_latent:
            return z
        return F.normalize(z, p=2, dim=1, eps=1e-8)

    def forward(self, x, y):
        z_x = self._maybe_norm(self.img_encoder(x))
        z_y = self._maybe_norm(self.cond_encoder(y))

        z_q, vq_loss, perplexity, indices = self.quantizer(z_x)
        recon = self.decoder(z_q)
        return recon, vq_loss, perplexity, indices, z_x, z_y

    @torch.no_grad()
    def inference(self, y):
        """
        Args:
            model: 训练好的 VQVAE (alignment version)
            y:     condition tensor, shape [B, num_params]
            device: torch.device

        Returns:
            recon_x: generated images, [B, C, H, W]
            indices: codebook indices used
        """

        # 1. 条件编码 → latent
        z_y = self.cond_encoder(y)  # [B, Z]

        if self.l2_normalize_latent:
            z_y = torch.nn.functional.normalize(z_y, dim=1)

        # 2. 量化（直接用 VQ）
        z_q, _, _, indices = self.quantizer(z_y)

        # 3. 解码
        recon_x = self.decoder(z_q)

        return recon_x