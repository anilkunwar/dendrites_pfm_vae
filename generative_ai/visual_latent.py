import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import seaborn as sns

from src.dataloader import DendritePFMDataset
from src.modelv4 import VAE


# ===== y 维度名称（含 did）=====
Y_NAMES = [
    "t",
    "POT_LEFT",
    "fo",
    "Al",
    "Bl",
    "Cl",
    "As",
    "Bs",
    "Cs",
    "cleq",
    "cseq",
    "L1o",
    "L2o",
    "ko",
    "Noise",
    "did"
]


@torch.no_grad()
def collect_latents(model, loader, device, latent_source="mu", use_ctr=False):
    model.eval()
    feats, ys, dids = [], [], []

    for batch in loader:
        if len(batch) == 4:
            x, y, did, xo = batch
        else:
            x, y = batch
            did = np.arange(len(x))

        x = x.to(device)
        y_t = y.to(device)

        # if use_ctr:
        _, mu, _, z = model(x, y_t)
        # else:
        #     _, mu, _, z = model(x)

        latent = mu if latent_source == "mu" else z
        feats.append(latent.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        dids.append(np.asarray(did, dtype=np.float32))

    feats = np.concatenate(feats, axis=0)
    ys = np.concatenate(ys, axis=0)
    dids = np.concatenate(dids, axis=0)[:, None]

    y_all = np.concatenate([ys, dids], axis=1)
    return feats, y_all


def smooth_on_knn_graph(y, knn_idx):
    return np.array([y[n].mean() for n in knn_idx])


def main():
    parser = argparse.ArgumentParser()

    # ===== basic =====
    parser.add_argument("--ckpt", type=str,
        default="results/V7_latent_size4_noise0.8_beta_start0.1_beta_end2.0_phy1.0_ncomp32_scale_weight0.25_20251221_104412/ckpt/CVAE.ckpt")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--split_json", type=str, default="data/dataset_split.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # ===== latent =====
    parser.add_argument("--latent_source", choices=["mu", "z"], default="mu")
    parser.add_argument("--use_ctr", action="store_true")

    # ===== t-SNE =====
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--tsne_iter", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)

    # ===== similarity =====
    parser.add_argument("--knn_k", type=int, default=15)

    # ===== output =====
    parser.add_argument("--out_dir", type=str, default="latent_viz_all_train_v2")
    parser.add_argument("--cmap", type=str, default="viridis")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== load model =====
    model = torch.load(args.ckpt, weights_only=False, map_location=device).to(device)
    model.eval()

    # ===== dataset =====
    dataset = DendritePFMDataset(
        (3, args.image_size, args.image_size),
        args.split_json,
        split=args.split
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # ===== collect =====
    feats, y_all = collect_latents(
        model, loader, device,
        latent_source=args.latent_source,
        use_ctr=args.use_ctr
    )

    if y_all.shape[1] != len(Y_NAMES):
        raise ValueError("y dimension mismatch")

    # ===== PCA =====
    if args.pca_dim > 0 and feats.shape[1] > args.pca_dim:
        feats = PCA(args.pca_dim, random_state=args.seed).fit_transform(feats)

    # ===== t-SNE =====
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, (feats.shape[0] - 1) / 3),
        init="pca",
        learning_rate="auto",
        random_state=args.seed
    )
    emb = tsne.fit_transform(feats)

    # ===== per-y visualization =====
    for d, name in enumerate(Y_NAMES):
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(
            emb[:, 0], emb[:, 1],
            c=y_all[:, d],
            cmap=args.cmap,
            s=10, alpha=0.8
        )
        plt.colorbar(sc, label=name)
        plt.title(f"t-SNE colored by {name}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()

        out = os.path.join(
            args.out_dir, f"tsne_{d:02d}_{name}_{args.latent_source}.png"
        )
        plt.savefig(out, dpi=300)
        plt.close()

    # ===== similarity analysis =====
    knn = NearestNeighbors(n_neighbors=args.knn_k + 1).fit(feats)
    _, idx = knn.kneighbors(feats)
    knn_idx = idx[:, 1:]

    num_y = y_all.shape[1]
    y_smooth = np.zeros_like(y_all)

    for d in range(num_y):
        y_smooth[:, d] = smooth_on_knn_graph(y_all[:, d], knn_idx)

    sim = np.zeros((num_y, num_y))
    for i in range(num_y):
        for j in range(num_y):
            sim[i, j] = spearmanr(y_smooth[:, i], y_smooth[:, j])[0]

    df = pd.DataFrame(sim, index=Y_NAMES, columns=Y_NAMES)
    csv_path = os.path.join(args.out_dir, "y_similarity_spearman.csv")
    df.to_csv(csv_path)

    # ===== heatmap =====
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="coolwarm", center=0, square=True)
    plt.title("Similarity of y dimensions in latent space")
    plt.tight_layout()

    fig_path = os.path.join(args.out_dir, "y_similarity_heatmap.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("All results saved to:", args.out_dir)


if __name__ == "__main__":
    main()
