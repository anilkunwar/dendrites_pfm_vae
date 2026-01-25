import os
import textwrap

import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

from src.dataloader import DendritePFMDataset
from src.modelv11 import mdn_point_and_confidence, postprocess_image


def main(args):

    plt.rcParams["image.cmap"] = "coolwarm"
    VAR_SCALE = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = DendritePFMDataset(args.image_size, os.path.join("data", "dataset_split.json"), split="test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    vae = torch.load(os.path.join(
        args.model_root,
        "ckpt", "best.pt"
    ), weights_only=False).to(device)
    vae.eval()

    save_fig_path = os.path.join(args.model_root, "figures")
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    # evaluate
    with torch.no_grad():
        for iteration, (x, y, did, _) in enumerate(test_dataloader):

            x = x.to(device)

            # image and control variables
            recon_x, mu_q, logvar_q, (pi_s, mu_s, log_sigma_s), z = vae(x)

            # ---- stochastic prediction + confidence (from sampled z) ----
            theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
                pi_s, mu_s, log_sigma_s, var_scale=VAR_SCALE
            )
            y_pred_s = theta_hat_s.detach().cpu().numpy()[0].tolist()
            conf_s = conf_param_s.detach().cpu().numpy()[0].tolist()
            conf_global_s = conf_global_s.detach().cpu().numpy()[0].tolist()

            # determinstic
            pi_d, mu_d, log_sigma_d = vae.mdn_head(mu_q)
            theta_hat_d, conf_param_d, conf_global_d, modes_d = mdn_point_and_confidence(
                pi_d, mu_d, log_sigma_d, var_scale=VAR_SCALE
            )
            y_pred_d = theta_hat_d.detach().cpu().numpy()[0].tolist()
            conf_d = conf_param_d.detach().cpu().numpy()[0].tolist()
            conf_global_d = conf_global_d.detach().cpu().numpy()[0].tolist()

            fig = plt.figure(figsize=(26, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.30, wspace=0.25, width_ratios=[1, 1, 3])

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(x[0].view(args.image_size).cpu().data.numpy()[0])
            ax1.set_title("eta gt", fontsize=12, fontweight="bold")
            ax1.axis("off")

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(recon_x[0].view(args.image_size).cpu().data.numpy()[0])
            ax2.set_title("eta pd", fontsize=12, fontweight="bold")
            ax2.axis("off")

            ax3 = fig.add_subplot(gs[1, 0])
            ax3.imshow(x[0].view(args.image_size).cpu().data.numpy()[1])
            ax3.set_title("c gt", fontsize=12, fontweight="bold")
            ax3.axis("off")

            ax4 = fig.add_subplot(gs[1, 1])
            ax4.imshow(recon_x[0].view(args.image_size).cpu().data.numpy()[1])
            ax4.set_title("c pd", fontsize=12, fontweight="bold")
            ax4.axis("off")

            ax5 = fig.add_subplot(gs[2, 0])
            ax5.imshow(x[0].view(args.image_size).cpu().data.numpy()[2])
            ax5.set_title("φ gt", fontsize=12, fontweight="bold")
            ax5.axis("off")

            ax6 = fig.add_subplot(gs[2, 1])
            ax6.imshow(recon_x[0].view(args.image_size).cpu().data.numpy()[2])
            ax6.set_title("φ pd", fontsize=12, fontweight="bold")
            ax6.axis("off")

            plt.title(f"t={y[0][0].item()}_did={did[0]}")
            plt.axis('off')

            ax_text = fig.add_subplot(gs[:, 2])
            ax_text.axis("off")

            t = f"""
            y={y[0].numpy().tolist()}
            y_pred_s={y_pred_s}
            conf_s={conf_s}
            conf_global_s={conf_global_s}
            y_pred_d={y_pred_d}
            conf_d={conf_d}
            conf_global_d={conf_global_d}
            """
            # 设置每行最大字符数
            max_line_length = 40  # 根据需要调整
            wrapped_text = textwrap.fill(t, width=max_line_length)

            ax_text.text(
                0.0, 1.0, wrapped_text,
                va="top", ha="left",
                fontsize=12,
                family="monospace",
                linespacing=1.25,
            )

            plt.savefig(
                os.path.join(save_fig_path, str(f"t={y[0][0].item()}_did={did[0]}.png")),
                dpi=300)
            plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=tuple, default=(3, 48, 48))
    parser.add_argument("--model_root", type=str, default='results/final_model')

    args = parser.parse_args()

    main(args)
