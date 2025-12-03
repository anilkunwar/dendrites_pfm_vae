import torch
import torch.nn as nn
import torch.nn.functional as F


def multiscale_recon_loss(
    x_recon_logits,
    x_true,
    num_scales=4,
    scale_weight=0.5
):
    """
    多尺度重建损失
    - 每往下采样一倍，分辨率变为上一层的一半
    - 低分辨率 -> 强调形状 / 连通性
    - 高分辨率 -> 保留局部细节

    参数：
        x_recon_logits: B x 1 x H x W  (你的模型输出 logits)
        x_true:         B x 1 x H x W  (ground truth)
        num_scales:     一共使用多少尺度
        loss_type:      "bce" 或 "mse"
        scale_weight:   (0~1) coarse 层权重随尺度递减的倍数
                        e.g. 0.5 → coarse 层权重要比细层大
    """

    # 初始损失
    total_loss = 0.0
    weight_sum = 0.0

    # 当前尺度输入
    cur_pred = x_recon_logits
    cur_true = x_true

    # 权重：粗尺度更重要（形态），细尺度稍弱（像素对齐）
    # e.g. scale_weight = 0.5 → 权重依次：1, 0.5, 0.25, 0.125 ...
    cur_weight = 1.0

    for scale in range(num_scales):

        loss = F.mse_loss(cur_pred, cur_true, reduction="mean")

        total_loss += cur_weight * loss
        weight_sum += cur_weight

        # 下一个尺度：尺寸减半
        # 注意：这里我们用 avg_pool 下采样
        cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2)
        cur_true = F.avg_pool2d(cur_true, kernel_size=2, stride=2)

        # 更新权重
        cur_weight *= scale_weight

    # 平均化权重
    return total_loss / weight_sum

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

    def grad_loss(self, input, target):
        """
        对 dx 和 dy 的误差做 0–1 逐通道归一化。
        归一化区间来自 target 的每个通道的 min/max。
        """

        eps = 1e-6
        C = input.shape[1]

        # dx: (B, C, H, W-1)
        input_dx = F.conv2d(input, self.kernel_gradx, padding=0, groups=C)
        target_dx = F.conv2d(target, self.kernel_gradx, padding=0, groups=C)

        # dy: (B, C, H-1, W)
        input_dy = F.conv2d(input, self.kernel_grady, padding=0, groups=C)
        target_dy = F.conv2d(target, self.kernel_grady, padding=0, groups=C)

        # --------------------------------------------------
        # 每通道: 用 target_dx/target_dy 的 min/max 做 0-1 归一化
        # --------------------------------------------------
        # dx 通道的 min/max
        dx_min = target_dx.amin(dim=(2, 3), keepdim=True)
        dx_max = target_dx.amax(dim=(2, 3), keepdim=True)
        dx_scale = (dx_max - dx_min).clamp_min(1e-6)

        # dy 通道的 min/max
        dy_min = target_dy.amin(dim=(2, 3), keepdim=True)
        dy_max = target_dy.amax(dim=(2, 3), keepdim=True)
        dy_scale = (dy_max - dy_min).clamp_min(1e-6)

        # 归一化
        input_dx_norm = (input_dx - dx_min) / dx_scale
        target_dx_norm = (target_dx - dx_min) / dx_scale

        input_dy_norm = (input_dy - dy_min) / dy_scale
        target_dy_norm = (target_dy - dy_min) / dy_scale

        # 归一化后做 L1
        dx_loss = torch.abs(input_dx_norm - target_dx_norm).mean()
        dy_loss = torch.abs(input_dy_norm - target_dy_norm).mean()

        return dx_loss + dy_loss

    def forward(self, recon_x, x, mean, log_var, freeze=False):
        batch_size = x.size(0)
        if not freeze:
            self.current_step += 1

        beta = self.compute_beta()

        # ---------------------------------------
        # 逐通道 0-1 归一化（参考 ground truth）
        # x_norm, recon_norm : [B,C,H,W]
        # ---------------------------------------
        x_min = x.amin(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        x_max = x.amax(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        scale = (x_max - x_min).clamp_min(1e-6)

        x_norm = (x - x_min) / scale
        recon_norm = (recon_x - x_min) / scale

        # 重建损失（在归一化空间中算 L1）
        recon_loss = multiscale_recon_loss(recon_norm, x_norm)

        # KL 损失
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size
        kl_loss = beta * kl

        # 梯度损失（逐通道 normalization）
        grad_loss = self.grad_loss(recon_x, x) * self.w_grad

        total_loss = recon_loss + kl_loss + grad_loss

        return total_loss, {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "kl": kl.item(),
            "beta": beta,
            "grad": grad_loss.item()
        }
