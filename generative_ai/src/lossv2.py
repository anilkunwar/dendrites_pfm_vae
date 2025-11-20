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
                 w_kl = 0.1,
                 w_tv=0.001,
                 w_smoothness=0.01,
                 w_grad=0.01,
                 device="cuda"):
        """
        Args:
            w_tv: Weight for total variation loss (encourage smoothness)
            w_smoothness: Weight for local smoothness loss
        """
        super(PhysicsConstrainedVAELoss, self).__init__()
        self.w_kl = w_kl
        self.w_tv = w_tv
        self.w_smoothness = w_smoothness
        self.w_grad = w_grad

        # 梯度卷积核
        self.kernel_grady = torch.tensor(
            [[[[1.], [-1.]]]] * 3, device=device
        )
        self.kernel_gradx = torch.tensor(
            [[[[1., -1.]]]] * 3, device=device
        )

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
        recon_loss = F.l1_loss(recon_x, x, reduction='sum') / batch_size
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size
        elbo_loss = recon_loss + kl_loss * self.w_kl

        # 2. Physics constraints on reconstructed images
        smoothness_loss_val = self.local_smoothness_loss(recon_x) * self.w_smoothness
        tv_loss_val = self.total_variation_loss(x) * self.w_tv
        grad_loss = (self.grad_loss(recon_x, x) / batch_size) * self.w_grad

        # 3. Total loss
        total_loss = elbo_loss + smoothness_loss_val + tv_loss_val + grad_loss

        # Return detailed loss breakdown
        loss_dict = {
            'total': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'elbo': elbo_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'tv': tv_loss_val if isinstance(tv_loss_val, float) else tv_loss_val.item(),
            'smoothness': smoothness_loss_val if isinstance(smoothness_loss_val, float) else smoothness_loss_val.item()
        }

        return total_loss, loss_dict

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

    def grad_loss(self, input, target):
        input_rectangles_h = F.conv2d(input, self.kernel_grady, padding=0, groups=3)
        target_rectangles_h = F.conv2d(target, self.kernel_grady, padding=0, groups=3)
        loss_h = torch.sum(torch.abs(input_rectangles_h - target_rectangles_h) * (target_rectangles_h.abs().exp()))

        input_rectangles_o = F.conv2d(input, self.kernel_gradx, padding=0, groups=3)
        target_rectangles_o = F.conv2d(target, self.kernel_gradx, padding=0, groups=3)
        loss_o = torch.sum(torch.abs(input_rectangles_o - target_rectangles_o) * (target_rectangles_o.abs().exp()))

        return loss_h + loss_o

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

        self.base_loss = PhysicsConstrainedVAELoss()

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