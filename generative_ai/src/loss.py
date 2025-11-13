import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import fftpack
from skimage import filters
from skimage.restoration import denoise_bilateral


class PhysicsConstrainedVAELoss(nn.Module):
    """
    Physics-constrained VAE loss that enforces physical plausibility on reconstructed images.
    Applies smoothness, frequency, and total variation constraints to each output image.
    """

    def __init__(self,
                 w_edge=0.01,
                 w_fft=0.01,
                 w_tv=0.001,
                 w_smoothness=0.01,
                 use_edge_loss=True,
                 use_fft_loss=True,
                 use_tv_loss=True,
                 use_smoothness_loss=True,
                 device="cuda"):
        """
        Args:
            w_edge: Weight for edge magnitude loss (penalize excessive edges)
            w_fft: Weight for FFT-based frequency loss (penalize high-frequency noise)
            w_tv: Weight for total variation loss (encourage smoothness)
            w_smoothness: Weight for local smoothness loss
            use_edge_loss: Apply edge magnitude constraint
            use_fft_loss: Apply frequency domain constraint
            use_tv_loss: Apply total variation constraint
            use_smoothness_loss: Apply local smoothness constraint
        """
        super(PhysicsConstrainedVAELoss, self).__init__()
        self.w_edge = w_edge
        self.w_fft = w_fft
        self.w_tv = w_tv
        self.w_smoothness = w_smoothness
        self.use_edge_loss = use_edge_loss
        self.use_fft_loss = use_fft_loss
        self.use_tv_loss = use_tv_loss
        self.use_smoothness_loss = use_smoothness_loss

        # 梯度卷积核
        self.kernel_grady = torch.tensor(
            [[[[1.], [-1.]]]] * 3, device=device
        )
        self.kernel_gradx = torch.tensor(
            [[[[1., -1.]]]] * 3, device=device
        )

    # ===============================
    # 梯度损失
    # ===============================
    def grad_loss(self, input, target):
        input_rectangles_h = F.conv2d(input, self.kernel_grady, padding=0, groups=3)
        target_rectangles_h = F.conv2d(target, self.kernel_grady, padding=0, groups=3)
        loss_h = torch.sum(torch.abs(input_rectangles_h - target_rectangles_h) * (target_rectangles_h.abs().exp()))

        input_rectangles_o = F.conv2d(input, self.kernel_gradx, padding=0, groups=3)
        target_rectangles_o = F.conv2d(target, self.kernel_gradx, padding=0, groups=3)
        loss_o = torch.sum(torch.abs(input_rectangles_o - target_rectangles_o) * (target_rectangles_o.abs().exp()))

        return loss_h + loss_o

    def forward(self, recon_x, x, mean, log_var):
        """
        Compute physics-constrained VAE loss.

        Args:
            recon_x: Reconstructed images [batch_size, channels, H, W]
            x: Original images [batch_size, channels, H, W]
            mean: Latent mean [batch_size, latent_dim]
            log_var: Latent log variance [batch_size, latent_dim]

        Returns:
            total_loss: Total loss including ELBO and physics constraints
            loss_dict: Dictionary with individual loss components
        """
        batch_size = x.size(0)

        # 1. Standard ELBO components
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size
        elbo_loss = recon_loss + kl_loss

        # 2. Physics constraints on reconstructed images
        physics_loss = 0.0
        edge_loss_val = 0.0
        fft_loss_val = 0.0
        tv_loss_val = 0.0
        smoothness_loss_val = 0.0

        if self.use_edge_loss:
            edge_loss_val = self.edge_magnitude_loss(recon_x)
            physics_loss += self.w_edge * edge_loss_val

        if self.use_fft_loss:
            fft_loss_val = self.fft_frequency_loss(recon_x)
            physics_loss += self.w_fft * fft_loss_val

        if self.use_tv_loss:
            tv_loss_val = self.total_variation_loss(recon_x)
            physics_loss += self.w_tv * tv_loss_val

        if self.use_smoothness_loss:
            smoothness_loss_val = self.local_smoothness_loss(recon_x)
            physics_loss += self.w_smoothness * smoothness_loss_val

        # 3. Total loss
        total_loss = elbo_loss + physics_loss + (self.grad_loss(recon_x, x) / batch_size) * 0.1

        # Return detailed loss breakdown
        loss_dict = {
            'total': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'elbo': elbo_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'physics': physics_loss.item() if torch.is_tensor(physics_loss) else physics_loss,
            'edge': edge_loss_val if isinstance(edge_loss_val, float) else edge_loss_val.item(),
            'fft': fft_loss_val if isinstance(fft_loss_val, float) else fft_loss_val.item(),
            'tv': tv_loss_val if isinstance(tv_loss_val, float) else tv_loss_val.item(),
            'smoothness': smoothness_loss_val if isinstance(smoothness_loss_val, float) else smoothness_loss_val.item()
        }

        return total_loss, loss_dict

    def edge_magnitude_loss(self, images):
        """
        可微分的 Scharr edge magnitude loss。
        使用 PyTorch 卷积实现，不再使用 numpy / skimage。
        """

        # Scharr kernels
        scharr_x = torch.tensor([
            [3, 0, -3],
            [10, 0, -10],
            [3, 0, -3]
        ], dtype=images.dtype, device=images.device).unsqueeze(0).unsqueeze(0) / 16.0

        scharr_y = torch.tensor([
            [3, 10, 3],
            [0, 0, 0],
            [-3, -10, -3]
        ], dtype=images.dtype, device=images.device).unsqueeze(0).unsqueeze(0) / 16.0

        # Apply to each channel independently
        channels = images.size(1)
        scharr_x = scharr_x.repeat(channels, 1, 1, 1)
        scharr_y = scharr_y.repeat(channels, 1, 1, 1)

        grad_x = F.conv2d(images, scharr_x, padding=1, groups=channels)
        grad_y = F.conv2d(images, scharr_y, padding=1, groups=channels)

        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # 返回平均边缘强度
        return edge_mag.mean()

    def fft_frequency_loss(self, images):
        """
        可微分的 FFT high-frequency ratio loss。
        使用 torch.fft，不使用 numpy / scipy。
        """
        # images: [B, C, H, W]
        B, C, H, W = images.shape

        # 对每个 channel 做 FFT
        fft_img = torch.fft.fft2(images)  # complex tensor
        fft_shifted = torch.fft.fftshift(fft_img)

        # Power spectrum
        power = (fft_shifted.real ** 2 + fft_shifted.imag ** 2)

        # 构造 low-frequency mask（中心圆）
        yy, xx = torch.meshgrid(
            torch.arange(H, device=images.device),
            torch.arange(W, device=images.device),
            indexing="ij"
        )
        cy, cx = H // 2, W // 2
        radius = min(H, W) // 4
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2).float()  # [H,W]

        # 扩展为 batch 和 channel
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        center_energy = (power * mask).sum(dim=[1, 2, 3])
        total_energy = power.sum(dim=[1, 2, 3])
        hf_energy = total_energy - center_energy

        # high-frequency 比例
        hf_ratio = hf_energy / (total_energy + 1e-8)

        return hf_ratio.mean()

    def total_variation_loss(self, images):
        """
        Total Variation loss encourages spatial smoothness.
        Physical reasoning: Penalizes pixel-to-pixel variations,
        encouraging piecewise smooth images typical of physical systems.

        Args:
            images: [batch_size, channels, H, W]
        Returns:
            Total variation loss
        """
        # Compute differences between adjacent pixels
        diff_h = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        diff_w = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])

        # Sum of absolute differences
        tv_loss = torch.sum(diff_h) + torch.sum(diff_w)

        return tv_loss / images.size(0)

    def local_smoothness_loss(self, images):
        """
        Penalize large variations between neighboring pixels using Laplacian.
        Physical reasoning: Physical fields typically vary smoothly in space.

        Args:
            images: [batch_size, channels, H, W]
        Returns:
            Local smoothness loss
        """
        # Laplacian kernel (discrete approximation of second derivative)
        # Detects rapid changes in intensity
        batch_size, channels, h, w = images.shape

        # Compute Laplacian using convolution
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=images.dtype, device=images.device).view(1, 1, 3, 3)

        # Replicate kernel for all channels
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)

        # Apply Laplacian
        laplacian = F.conv2d(images, laplacian_kernel, padding=1, groups=channels)

        # L2 norm of Laplacian (penalize large second derivatives)
        smoothness_loss = torch.sum(laplacian ** 2)

        return smoothness_loss / batch_size


class AdaptivePhysicsVAELoss(nn.Module):
    """
    Adaptive version that adjusts physics constraint strength based on training progress.
    Gradually increases physics constraints as reconstruction quality improves.
    """

    def __init__(self,
                 w_edge_max=0.01,
                 w_fft_max=0.01,
                 w_tv_max=0.001,
                 warmup_epochs=10):
        """
        Args:
            w_edge_max: Maximum weight for edge loss
            w_fft_max: Maximum weight for FFT loss
            w_tv_max: Maximum weight for TV loss
            warmup_epochs: Number of epochs to gradually increase physics weights
        """
        super(AdaptivePhysicsVAELoss, self).__init__()
        self.w_edge_max = w_edge_max
        self.w_fft_max = w_fft_max
        self.w_tv_max = w_tv_max
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.base_loss = PhysicsConstrainedVAELoss(
            w_edge=0, w_fft=0, w_tv=0,
            use_edge_loss=True, use_fft_loss=True, use_tv_loss=True
        )

    def set_epoch(self, epoch):
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch

        # Linear warmup
        alpha = min(1.0, epoch / self.warmup_epochs)
        self.base_loss.w_edge = alpha * self.w_edge_max
        self.base_loss.w_fft = alpha * self.w_fft_max
        self.base_loss.w_tv = alpha * self.w_tv_max

    def forward(self, recon_x, x, mean, log_var):
        return self.base_loss(recon_x, x, mean, log_var)