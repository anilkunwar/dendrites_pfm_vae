import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)
    适合你的场景：同一相场模拟的样本作为正样本，其余为负样本。
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        参数:
            features: (N, D)  每个样本的隐变量 / embedding
            labels:   (N,)    每个样本所属的相场模拟 id（可为 int/long）

        返回:
            标量 loss
        """
        device = features.device

        # L2 归一化，得到单位向量（用余弦相似度）
        features = F.normalize(features, p=2, dim=1)

        # 相似度矩阵 (N, N)，s_ij = z_i^T z_j
        logits = torch.matmul(features, features.T)  # (N, N)
        logits = logits / self.temperature

        # 构造标签相等矩阵 (N, N)：同一个相场模拟为 1，否则为 0
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).to(device)  # bool

        # 去掉对角线（样本自己不作为自己的正样本）
        logits_mask = torch.ones_like(mask, dtype=torch.bool, device=device)
        logits_mask.fill_diagonal_(False)

        # 正样本 mask（排除自身）
        positives_mask = mask & logits_mask

        # 对所有样本，softmax 分母要排除自己，所以用 logits_mask 做 mask
        # 为了数值稳定性，先减去按行的最大值
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 把被 mask 掉的位置设成非常小的数，相当于不参与 softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 对每个样本 i，只对它的正样本 j∈P(i) 的 log_prob_ij 求平均
        positives_per_row = positives_mask.sum(dim=1)  # 每行的正样本个数
        # 避免除 0：只对有正样本的样本计算 loss
        valid_rows = positives_per_row > 0

        # (N, N) -> (N,)：对每一行，只保留正样本位置
        mean_log_prob_pos = (positives_mask * log_prob).sum(dim=1) / (positives_per_row + 1e-12)

        # 最终 loss：对 valid_rows 求负号平均
        loss = -mean_log_prob_pos[valid_rows].mean()

        return loss

class PhysicsConstrainedVAELoss(nn.Module):
    """
    Physics-constrained VAE loss that enforces physical plausibility on reconstructed images.
    Applies smoothness, frequency, and total variation constraints to each output image.
    """

    def __init__(self,
                 w_kl = 0.1,
                 w_grad=0.01,
                 w_con=0.1,
                 device="cuda"):
        """
        Args:
            w_tv: Weight for total variation loss (encourage smoothness)
            w_smoothness: Weight for local smoothness loss
        """
        super(PhysicsConstrainedVAELoss, self).__init__()
        self.w_kl = w_kl
        self.w_grad = w_grad
        self.w_con = w_con

        # 梯度卷积核
        self.kernel_grady = torch.tensor(
            [[[[1.], [-1.]]]] * 3, device=device
        )
        self.kernel_gradx = torch.tensor(
            [[[[1., -1.]]]] * 3, device=device
        )

        self.cLoss = SupervisedContrastiveLoss()

    def grad_loss(self, input, target):
        input_rectangles_h = F.conv2d(input, self.kernel_grady, padding=0, groups=3)
        target_rectangles_h = F.conv2d(target, self.kernel_grady, padding=0, groups=3)
        loss_h = torch.sum(torch.abs(input_rectangles_h - target_rectangles_h) * (target_rectangles_h.abs().exp()))

        input_rectangles_o = F.conv2d(input, self.kernel_gradx, padding=0, groups=3)
        target_rectangles_o = F.conv2d(target, self.kernel_gradx, padding=0, groups=3)
        loss_o = torch.sum(torch.abs(input_rectangles_o - target_rectangles_o) * (target_rectangles_o.abs().exp()))

        return loss_h + loss_o

    def forward(self, recon_x, x, mean, log_var, n_mean, n_log_var, dids):
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
        elbo_loss = recon_loss + kl_loss * self.w_kl

        grad_loss = self.grad_loss(recon_x, x) * self.w_grad
        con_mean_loss = self.cLoss(n_mean, dids)
        con_log_var_loss = self.cLoss(n_log_var, dids)
        c_loss = (con_mean_loss + con_log_var_loss) * self.w_con

        # 3. Total loss
        total_loss = elbo_loss + c_loss + grad_loss

        # Return detailed loss breakdown
        loss_dict = {
            'total': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'elbo': elbo_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'contrasive': c_loss.item()
        }

        return total_loss, loss_dict