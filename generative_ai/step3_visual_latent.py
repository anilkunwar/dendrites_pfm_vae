import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import seaborn as sns

from src.dataloader import DendritePFMDataset


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
def collect_latents(model, loader, device, latent_source="mu"):
    feats, ys, dids = [], [], []

    for batch in loader:
        if len(batch) == 4:
            x, y, did, xo = batch
        else:
            x, y = batch
            did = np.arange(len(x))

        x = x.to(device)
        y_t = y.to(device)

        recon_x, mu_q, logvar_q, ctr_pred, z = model(x)

        latent = mu_q if latent_source == "mu" else z
        feats.append(latent.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        dids.append(np.asarray(did, dtype=np.float32))

    feats = np.concatenate(feats, axis=0)
    ys = np.concatenate(ys, axis=0)
    dids = np.concatenate(dids, axis=0)[:, None]

    all_ys = np.concatenate([ys, dids], axis=1)
    return feats, all_ys


def smooth_on_knn_graph(y, knn_idx):
    return np.array([y[n].mean() for n in knn_idx])


# ===== 判断是否为“离散整数类别” =====
def is_discrete_integer_values(arr, max_unique_for_discrete=30):
    a = np.asarray(arr).reshape(-1)
    a = a[~np.isnan(a)]

    if a.size == 0:
        return False

    # 是否接近整数
    if not np.allclose(a, np.round(a)):
        return False

    # unique 数量是否合理
    uniq = np.unique(a.astype(np.int64))
    return len(uniq) <= max_unique_for_discrete


# ===== 构造离散 colormap + norm =====
def get_discrete_cmap_and_norm(values, cmap_name="tab20"):
    values = np.asarray(values)
    uniq = np.unique(np.round(values).astype(np.int64))
    uniq = np.sort(uniq)

    class_to_idx = {c: i for i, c in enumerate(uniq)}
    mapped = np.vectorize(lambda x: class_to_idx[int(np.round(x))])(values)

    n_cls = len(uniq)
    cmap = plt.get_cmap(cmap_name, n_cls)

    boundaries = np.arange(-0.5, n_cls + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=n_cls)

    return mapped, uniq, cmap, norm




def main():
    parser = argparse.ArgumentParser()

    # ===== basic =====
    parser.add_argument("--model_root", type=str,
                        default='results/V10_ADV_lat=16_beta=0.1_warm=0.3_ctr=1.0_smooth=0.05_adv=0.1_d=0.0001_g=0.0001_time=20260116_123955')

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--split_json", type=str, default="data/dataset_split.json")
    parser.add_argument("--split", type=str, default="val")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    # ===== latent =====
    parser.add_argument("--latent_source", choices=["mu", "z"], default="z")

    # ===== t-SNE =====
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--tsne_iter", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)

    # ===== similarity =====
    parser.add_argument("--knn_k", type=int, default=15)

    # ===== output =====
    parser.add_argument("--out_dir", type=str, default="latent_viz_test")
    parser.add_argument("--cmap", type=str, default="viridis")

    # ===== did visualization optimization =====
    parser.add_argument("--num_did_vis", type=int, default=-1,
                        help="number of did groups to visualize in did plot")
    parser.add_argument("--did_discrete_cmap", type=str, default="tab20",
                        help="discrete colormap name for did/categories")
    parser.add_argument("--did_show_centroids", action="store_true",
                        help="draw centroid marker for each selected did group")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== load model =====
    vae = torch.load(
        os.path.join(args.model_root, "ckpt", "best.pt"),
        map_location=device,
        weights_only=False
    )
    vae.eval()

    # ===== dataset =====
    dataset = DendritePFMDataset(
        (3, args.image_size, args.image_size),
        args.split_json,
        split=args.split
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    out_dir = os.path.join(args.model_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ===== collect =====
    feats, y_all = collect_latents(
        vae, loader, device,
        latent_source=args.latent_source
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

    # ===== 随机选择 did 子集（仅用于 did 可视化）=====
    did_idx = Y_NAMES.index("did")
    all_dids = y_all[:, did_idx].astype(int)
    uniq_dids = np.unique(all_dids)

    rng = np.random.RandomState(args.seed)
    num_vis = min(args.num_did_vis, len(uniq_dids)) if args.num_did_vis > 0 else len(uniq_dids)
    selected_dids = rng.choice(uniq_dids, size=num_vis, replace=False)
    did_vis_mask = np.isin(all_dids, selected_dids)

    print(f"[INFO] Visualizing {num_vis} did groups (seed={args.seed}): {np.sort(selected_dids)}")

    # ===== per-y visualization =====
    for d, name in enumerate(Y_NAMES):
        y = y_all[:, d].astype(float)

        plt.figure(figsize=(7, 6))

        # ---------- 只对 did 进行子集可视化 ----------
        if name == "did":
            emb_plot = emb[did_vis_mask]
            y_plot = y[did_vis_mask]
        else:
            emb_plot = emb
            y_plot = y
        # --------------------------------------------

        if is_discrete_integer_values(y_plot):
            cmap_name = args.did_discrete_cmap if name == "did" else "tab20"

            mapped, classes, cmap, norm = get_discrete_cmap_and_norm(
                y_plot, cmap_name=cmap_name
            )

            sc = plt.scatter(
                emb_plot[:, 0], emb_plot[:, 1],
                c=mapped,
                cmap=cmap,
                norm=norm,
                s=12, alpha=0.85
            )

            cbar = plt.colorbar(
                sc,
                ticks=np.arange(len(classes)),
                label=name
            )
            cbar.ax.set_yticklabels([str(int(c)) for c in classes])

            # ===== did: 可选绘制每个组的 centroid（论文图常用）=====
            if name == "did" and args.did_show_centroids:
                # mapped 是 0..C-1 的类别索引
                for ci in range(len(classes)):
                    mask_ci = (mapped == ci)
                    if np.any(mask_ci):
                        cx = emb_plot[mask_ci, 0].mean()
                        cy = emb_plot[mask_ci, 1].mean()
                        # centroid marker
                        plt.scatter([cx], [cy], s=80, marker="X", edgecolors="k")

        else:
            sc = plt.scatter(
                emb_plot[:, 0], emb_plot[:, 1],
                c=y_plot,
                cmap=args.cmap,
                s=10, alpha=0.8
            )
            plt.colorbar(sc, label=name)

        title_extra = ""
        if name == "did":
            title_extra = f" (random {len(np.unique(y_plot.astype(int)))} groups)"

        plt.title(f"t-SNE colored by {name}{title_extra}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()

        out = os.path.join(
            out_dir, f"tsne_{d:02d}_{name}_{args.latent_source}.png"
        )
        plt.savefig(out, dpi=300)
        plt.close()

    # ===== similarity analysis =====
    knn = NearestNeighbors(n_neighbors=args.knn_k + 1).fit(feats)
    _, idx = knn.kneighbors(feats)
    knn_idx = idx[:, 1:]

    num_y = y_all.shape[1]
    y_smooth = np.zeros_like(y_all, dtype=float)

    for d in range(num_y):
        y_smooth[:, d] = smooth_on_knn_graph(y_all[:, d], knn_idx)

    sim = np.zeros((num_y, num_y))
    for i in range(num_y):
        for j in range(num_y):
            sim[i, j] = spearmanr(y_smooth[:, i], y_smooth[:, j])[0]

    df = pd.DataFrame(sim, index=Y_NAMES, columns=Y_NAMES)
    df.to_csv(os.path.join(out_dir, "y_similarity_spearman.csv"))

    # ===== heatmap =====
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="coolwarm", center=0, square=True)
    plt.title("Similarity of y dimensions in latent space")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "y_similarity_heatmap.png"), dpi=300)
    plt.close()

    print("All results saved to:", out_dir)


if __name__ == "__main__":
    main()
