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
                 w_grad=0.01,
                 device="cuda"):
        """
        Args:
            w_tv: Weight for total variation loss (encourage smoothness)
            w_smoothness: Weight for local smoothness loss
        """
        super(PhysicsConstrainedVAELoss, self).__init__()
        self.w_kl = w_kl
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
        grad_loss = (self.grad_loss(recon_x, x) / batch_size) * self.w_grad

        # 3. Total loss
        total_loss = elbo_loss + grad_loss

        # Return detailed loss breakdown
        loss_dict = {
            'total': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'elbo': elbo_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
        }

        return total_loss, loss_dict

    def grad_loss(self, input, target):
        input_rectangles_h = F.conv2d(input, self.kernel_grady, padding=0, groups=3)
        target_rectangles_h = F.conv2d(target, self.kernel_grady, padding=0, groups=3)
        loss_h = torch.sum(torch.abs(input_rectangles_h - target_rectangles_h) * (target_rectangles_h.abs().exp()))

        input_rectangles_o = F.conv2d(input, self.kernel_gradx, padding=0, groups=3)
        target_rectangles_o = F.conv2d(target, self.kernel_gradx, padding=0, groups=3)
        loss_o = torch.sum(torch.abs(input_rectangles_o - target_rectangles_o) * (target_rectangles_o.abs().exp()))

        return loss_h + loss_o