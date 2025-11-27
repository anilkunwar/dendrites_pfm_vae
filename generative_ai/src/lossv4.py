import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstrainedVAELoss(nn.Module):
    """
    Physics-constrained loss with annealed beta-VAE KL term.
    """

    def __init__(self,
                 beta_start=0.0,         # 初始 β
                 beta_end=4.0,           # 最终 β
                 anneal_steps=1000,     # β 从 start 增到 end 的步数
                 w_grad=0.01,
                 device="cuda"):
        super().__init__()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_steps = anneal_steps
        self.current_step = 0

        self.w_grad = w_grad

        # gradient kernels
        self.kernel_grady = torch.tensor(
            [[[[1.], [-1.]]]] * 3, device=device
        )
        self.kernel_gradx = torch.tensor(
            [[[[1., -1.]]]] * 3, device=device
        )

    # ----------------------------
    #  计算当前 β (线性 Annealing)
    # ----------------------------
    def compute_beta(self):
        t = min(self.current_step / self.anneal_steps, 1)
        beta = self.beta_start + t * (self.beta_end - self.beta_start)
        return beta

    def forward(self, recon_x, x, mean, log_var, freeze=False):
        batch_size = x.size(0)
        if not freeze:
            self.current_step += 1

        # 获取 annealed β
        beta = self.compute_beta()

        # 重构损失
        recon_loss = F.l1_loss(recon_x, x, reduction='sum') / batch_size

        # KL 散度
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size

        kl_loss = beta * kl

        # ELBO
        elbo_loss = recon_loss + kl_loss

        # 物理梯度约束
        grad_loss = (self.grad_loss(recon_x, x) / batch_size) * self.w_grad

        total_loss = elbo_loss + grad_loss

        return total_loss, {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "kl": kl.item(),
            "beta": beta,
            "grad": grad_loss.item(),
        }

    def grad_loss(self, input, target):
        input_rectangles_h = F.conv2d(input, self.kernel_grady, padding=0, groups=3)
        target_rectangles_h = F.conv2d(target, self.kernel_grady, padding=0, groups=3)
        loss_h = torch.sum(torch.abs(input_rectangles_h - target_rectangles_h) *
                           (target_rectangles_h.abs().exp()))

        input_rectangles_o = F.conv2d(input, self.kernel_gradx, padding=0, groups=3)
        target_rectangles_o = F.conv2d(target, self.kernel_gradx, padding=0, groups=3)
        loss_o = torch.sum(torch.abs(input_rectangles_o - target_rectangles_o) *
                           (target_rectangles_o.abs().exp()))

        return loss_h + loss_o
