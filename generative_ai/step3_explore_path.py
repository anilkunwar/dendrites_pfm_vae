import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from src.tools import *

# =========================
# 配置
# =========================
image_size = (3, 64, 64)
model_root = 'results/V9_lat=8_beta=0.1_warm=0.3_ctr=2.0_smooth=1.0_time=20260105_114337'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

POOL_SIZE = 8000         # latent 候选库
KNN_K = 80               # 每步近邻候选数
DIST_MIN = 0.03          # cosine distance 最小移动量（防抖）
CTR_START_MAX = 0.01     # 起点 ctr 阈值
MAX_STEPS = 300          # 单条路径最大步数（防无限）
REPEATS = 60             # 搜索次数：越大越容易找到更长路
PICK_MODE = "stochastic" # "greedy" or "stochastic"
TEMPERATURE = 0.10       # stochastic 采样温度，越小越接近 greedy

# =========================
# 模型
# =========================
vae = torch.load(os.path.join(model_root, "ckpt", "best.pt"), map_location=device, weights_only=False)
vae.eval()

def heuristic(ctr_pred):
    # 越大越好：你也可以加入 fractal / novelty / distance 等
    return float(ctr_pred[0])

def show_im(im, ctr_pred):
    _im = im
    if _im.ndim == 4:
        _im = _im[0]
    if _im.shape[0] in (1, 3):
        _im_vis = np.transpose(_im, (1, 2, 0))
    else:
        _im_vis = _im

    v, _, _ = fractal_dimension_boxcount(_im_vis)
    v = np.nanmean(v)

    print(f"ctr_pred[0]={ctr_pred[0]}")
    print(f"full ctr_pred={ctr_pred}")
    plt.title(f"fractal_dimension_boxcount: {v:.4f}")
    plt.imshow(_im_vis)
    plt.axis("off")
    plt.show()

# =========================
# 1) 建候选库
# =========================
Z_pool, ctr_pool, im_pool = [], [], []
with torch.no_grad():
    for _ in range(POOL_SIZE):
        im, ctr_pred, z = vae.inference(full_output=True)
        im = im[0].detach().cpu().numpy()
        ctr_pred = ctr_pred[0].detach().cpu().numpy()
        z = z[0].detach().cpu().numpy()
        Z_pool.append(z)
        ctr_pool.append(ctr_pred)
        im_pool.append(im)

Z_pool = np.stack(Z_pool, axis=0)        # (N, latent_dim)
ctr_pool = np.stack(ctr_pool, axis=0)    # (N, ?)
im_pool = np.stack(im_pool, axis=0)      # (N, C,H,W)

knn = NearestNeighbors(n_neighbors=min(KNN_K, len(Z_pool)), metric="cosine")
knn.fit(Z_pool)

# 起点集合：满足 ctr_start_max
start_candidates = np.where(ctr_pool[:, 0] < CTR_START_MAX)[0]
if len(start_candidates) == 0:
    raise RuntimeError("候选库中没有找到 ctr_pred[0] < CTR_START_MAX 的起点；调大 POOL_SIZE 或放宽 CTR_START_MAX")

# =========================
# 2) 单次搜索：从某个起点走到不能走为止
# =========================
def run_one_path(start_idx, rng: np.random.Generator):
    path_idx = [int(start_idx)]
    z_cur = Z_pool[start_idx]
    ctr_cur = float(ctr_pool[start_idx][0])

    # 防止回头：记录已经走过的点
    visited = set(path_idx)

    for _ in range(MAX_STEPS):
        dists, idxs = knn.kneighbors(z_cur.reshape(1, -1), n_neighbors=min(KNN_K, len(Z_pool)), return_distance=True)
        dists = dists[0]
        idxs = idxs[0]

        # 候选：ctr 单调增加 + 距离足够大 + 没访问过
        cand = []
        for dis, j in zip(dists, idxs):
            j = int(j)
            if j in visited:
                continue
            ctr_j = float(ctr_pool[j][0])
            if ctr_j > ctr_cur and dis >= DIST_MIN:
                cand.append((j, dis, heuristic(ctr_pool[j])))

        if not cand:
            # 尝试扩大检索范围（兜底）
            K_try = min(len(Z_pool), KNN_K * 4)
            d2, i2 = knn.kneighbors(z_cur.reshape(1, -1), n_neighbors=K_try, return_distance=True)
            d2 = d2[0]
            i2 = i2[0]
            for dis, j in zip(d2, i2):
                j = int(j)
                if j in visited:
                    continue
                ctr_j = float(ctr_pool[j][0])
                if ctr_j > ctr_cur and dis >= DIST_MIN:
                    cand.append((j, dis, heuristic(ctr_pool[j])))

        if not cand:
            break  # 走不动了

        # 选点策略：greedy 或 stochastic（更容易找到更长路径）
        if PICK_MODE == "greedy":
            # 以 heuristic 最大优先；可加 dis 次级排序
            cand.sort(key=lambda x: x[2], reverse=True)
            j, dis, _h = cand[0]
        else:
            # softmax(heuristic/T) 采样
            hs = np.array([c[2] for c in cand], dtype=np.float64)
            hs = hs - hs.max()
            probs = np.exp(hs / max(TEMPERATURE, 1e-6))
            probs = probs / probs.sum()
            pick = rng.choice(len(cand), p=probs)
            j, dis, _h = cand[pick]

        path_idx.append(j)
        visited.add(j)

        z_cur = Z_pool[j]
        ctr_cur = float(ctr_pool[j][0])

    return path_idx

# =========================
# 3) 多次搜索，挑最长
# =========================
rng = np.random.default_rng(0)

best_path = None
best_len = -1
best_end_ctr = -np.inf

all_paths = []
for r in range(REPEATS):
    start_idx = int(rng.choice(start_candidates))
    path_idx = run_one_path(start_idx, rng)
    all_paths.append(path_idx)

    L = len(path_idx)
    end_ctr = float(ctr_pool[path_idx[-1]][0])
    # 先比长度，再比终点 ctr（可选）
    if (L > best_len) or (L == best_len and end_ctr > best_end_ctr):
        best_path = path_idx
        best_len = L
        best_end_ctr = end_ctr

print(f"Best path length = {best_len}, end ctr_pred[0] = {best_end_ctr:.6f}")

# =========================
# 4) 可视化：搜索空间 + 最长路径（PCA 2D）
# =========================
pca = PCA(n_components=2, random_state=0)
Z_2d = pca.fit_transform(Z_pool)                     # (N,2)
path_2d = Z_2d[np.array(best_path, dtype=int)]       # (L,2)

# 用 ctr 作为颜色（不指定具体颜色，matplotlib 默认 colormap）
plt.figure(figsize=(9, 7))
sc = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], s=6, c=ctr_pool[:, 0], alpha=0.25)
plt.colorbar(sc, label="ctr_pred[0]")

# 画路径（点 + 线）
plt.plot(path_2d[:, 0], path_2d[:, 1], linewidth=2)
plt.scatter(path_2d[:, 0], path_2d[:, 1], s=40)

plt.title(f"Latent search space (PCA) + longest path (L={best_len})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# =========================
# 5) 展示最长路径上的若干关键帧（可选）
# =========================
# 例如：只看起点/中间/终点，避免太多图
if best_len >= 3:
    key_steps = [0, best_len // 2, best_len - 1]
else:
    key_steps = list(range(best_len))

for t in key_steps:
    idx = best_path[t]
    print(f"[path step {t}/{best_len-1}] idx={idx}, ctr_pred[0]={ctr_pool[idx][0]:.6f}")
    show_im(im_pool[idx], ctr_pool[idx])
