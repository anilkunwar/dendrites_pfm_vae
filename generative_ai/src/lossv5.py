import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstrainedVAELoss(nn.Module):
    """
    Physics-constrained loss with annealed beta-VAE KL term.
    现在总损失包含：
      - 重构损失 (L1)
      - KL(x-encoder 对标准正态的 KL, 带 beta anneal)
      - 物理梯度约束
      - ctr 分布一致性损失：让 N(mu_x, logvar_x) ~ N(mu_c, logvar_c)
    """

    def __init__(self,
                 beta_start=0.0,         # 初始 β
                 beta_end=4.0,           # 最终 β
                 anneal_steps=10000,      # β 从 start 增到 end 的步数
                 w_grad=0.01,            # 物理梯度权重
                 w_kl=1.0,          # 分布一致性损失权重
                 device="cuda"):
        super().__init__()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_steps = anneal_steps
        self.current_step = 0

        self.w_grad = w_grad
        self.w_kl = w_kl

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
        t = min(self.current_step / self.anneal_steps, 1.0)
        beta = self.beta_start + t * (self.beta_end - self.beta_start)
        return beta

    def forward(self,
                recon_x,
                x,
                mean,
                log_var,
                ctr_mean=None,
                ctr_logvar=None,
                freeze=False):
        """
        参数：
            recon_x: 重建图像 [B, C, H, W]
            x:       目标图像   [B, C, H, W]
            mean, log_var:       图像 encoder 输出的 mu_x, logvar_x
            ctr_mean, ctr_logvar: ctr 分支生成的 mu_c, logvar_c（如果有）
            freeze: 评估时不再更新 beta 的 step
        """
        batch_size = x.size(0)
        if not freeze:
            self.current_step += 1

        # 获取 annealed β
        beta = self.compute_beta()

        # ----------------------------
        # 1) 重构损失
        # ----------------------------
        recon_loss = F.l1_loss(recon_x, x, reduction='sum') / batch_size

        # ----------------------------
        # 3) 物理梯度约束
        # ----------------------------
        grad_loss = (self.grad_loss(recon_x, x) / batch_size) * self.w_grad

        # ----------------------------
        # 4) 分布一致性损失：
        #    KL( N(mu_x, logvar_x) || N(mu_c, logvar_c) )
        # ----------------------------
        if ctr_mean is not None and ctr_logvar is not None:
            # KL(p||q) = 0.5 * [ log|Σ_q| - log|Σ_p| - k + tr(Σ_q^-1 Σ_p)
            #                    + (μ_q - μ_p)^T Σ_q^-1 (μ_q - μ_p) ]
            # 这里 Σ_p = diag(exp(log_var)), Σ_q = diag(exp(ctr_logvar))
            var_x = torch.exp(log_var)
            var_c = torch.exp(ctr_logvar)

            consist_mat = (
                ctr_logvar - log_var  # log|Σ_q| - log|Σ_p|
                + (var_x + (mean - ctr_mean) ** 2) / var_c
                - 1
            )  # shape [B, latent_dim]

            consist_loss = 0.5 * torch.sum(consist_mat) / batch_size
        else:
            consist_loss = torch.tensor(0., device=x.device)

        # ----------------------------
        # 总损失
        # ----------------------------
        total_loss = recon_loss + grad_loss + self.w_kl * consist_loss

        return total_loss, {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "kl": consist_loss.item(),
            "beta": float(beta),
            "grad": grad_loss.item()
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
