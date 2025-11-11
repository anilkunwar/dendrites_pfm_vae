import torch
import torch.nn as nn

from generative_ai.src.dataloader import smooth_scale


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
# Decoder (with additive coupling between z and c)
# ==========================================================
class ResDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, latent_size, H, W,
                 conditional=True, num_params=0):
        super().__init__()
        self.conditional = conditional
        self.H, self.W = H, W
        init_h, init_w = H // 8, W // 8  # 16x16 for 128x128 input

        if self.conditional:
            # Project c to same dimension as z
            self.cMLP = nn.Sequential(
                nn.Linear(num_params, latent_size),
                nn.ReLU(inplace=True)
            )

        # Fully connected layer expands latent vector to spatial feature map
        self.fc = nn.Linear(latent_size, base_channels * 4 * init_h * init_w)

        # Transposed convolutions for upsampling (16 → 128)
        self.deconv = nn.Sequential(
            self._block(base_channels * 4, base_channels * 2),
            self._block(base_channels * 2, base_channels),
            nn.ConvTranspose2d(base_channels, out_channels, 4, 2, 1)
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, z, c=None):
        # Additive coupling: z' = z + f(c)
        if self.conditional and c is not None:
            c_proj = self.cMLP(c)
            z = z + c_proj

        x = self.fc(z)
        x = x.view(x.size(0), -1, self.H // 8, self.W // 8)
        x = self.deconv(x)
        x = smooth_scale(x)
        return x


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
    def __init__(self, image_size=(3, 128, 128), latent_size=64, hidden_dimension=64,
                 conditional=True, num_params=0):
        super().__init__()
        self.C, self.H, self.W = image_size
        self.latent_size = latent_size
        self.conditional = conditional

        # Encoder does NOT take c
        self.encoder = ResEncoder(self.C + 2, hidden_dimension, latent_size)

        # Decoder takes z (+ c additively)
        self.decoder = ResDecoder(
            self.C, hidden_dimension, latent_size, self.H, self.W,
            conditional=conditional, num_params=num_params
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c=None):
        B, _, H, W = x.shape
        device = x.device

        # Add coordinate channels to x
        y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = torch.cat([x, x_coords, y_coords], dim=1)

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, c)
        return recon, mu, logvar, z


# ==========================================================
# Quick Test
# ==========================================================
if __name__ == "__main__":
    model = VAE(
        image_size=(3, 128, 128),
        latent_size=32,
        hidden_dimension=256,
        conditional=True,
        num_params=15
    )

    x = torch.randn(2, 3, 128, 128)
    c = torch.randn(2, 15)
    recon, mu, logvar, z = model(x, c)

    print("Input:", x.shape)
    print("Reconstruction:", recon.shape)
    print("Latent:", z.shape)
    print("Total parameters: %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
