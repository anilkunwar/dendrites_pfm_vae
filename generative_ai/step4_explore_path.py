import json
import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

from src.dataloader import DendritePFMDataset, smooth_scale


# ====== Utils ======
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def get_init_tensor(image_size=(64, 64)):

    arr = np.load("data/case_000/npy_files/0.000000.npy")  # shape (H, W, 3)

    # assert arr.max() <= 5 and arr.min() >= -5, "Inappropriate values occured"
    arr = cv2.resize(arr, image_size)
    tensor_t = torch.from_numpy(arr).float().permute(2, 0, 1)
    tensor_t = smooth_scale(tensor_t)

    # build control variable
    c = [0.0]

    # find meta
    meta_path = "data/case_000/case_000.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    c += meta

    return tensor_t

def decode_single_channel(model, device, z_vec):
    z = torch.tensor(z_vec, device=device).unsqueeze(0)
    img = model.decoder(z)[0, 0]  # (H,W)
    img = img.detach().cpu().float()
    # 如果你的 decoder 输出是 [-1,1]，可以转到 [0,1]
    if img.min() < 0:
        img = (img + 1) / 2
    return img.numpy()

def heuristic_score(img):
    # return  W_TIME * t + W_FD * fd - (PRIOR_L2_W * (np.linalg.norm(z) ** 2) if MODE == "safe" else 0.0)
    return img.sum()

# ====== CONFIG ======
MODEL_ROOT = "/home/xtanghao/THPycharm/dendrites_pfm_vae/tmp/results_expV9/V9_lat=8_beta=0.1_warm=0.3_ctr=2.0_smooth=0.05_time=20260105_071004/"
CKPT_PATH  = os.path.join(MODEL_ROOT, "ckpt", "best.pt")
OUT_DIR    = os.path.join(MODEL_ROOT, "heuristic_search_continuous")

IMAGE_SIZE = 64
BATCH_SIZE = 64
NUM_WORKERS = 4
SPLIT_JSON = "data/dataset_split.json"
SPLIT = "train"

TIME_IDX = 0

STEPS = 200
SEED = 0

# objective weights
W_TIME = 1.0
W_FD   = 1.0

# --- naive random walk params ---
RW_SIGMA = 0.25          # 扰动幅度（大一点就更容易崩）
NUM_CAND = 32            # 每步试多少个候选

# --- safe guards ---
PRIOR_L2_W = 0.03        # 惩罚 ||z||^2
MAX_NORM   = 3.5         # 限制 z 的范数（粗暴但有效）
PIX_CLIP_MIN = -0.2      # 用于检测异常 decode
PIX_CLIP_MAX =  1.2

# optional projection to dataset manifold
USE_PROJ = False
PROJ_EVERY = 5           # 每隔几步投影一次
PROJ_ALPHA = 0.5         # z <- alpha*z + (1-alpha)*z_nn
KNN_K = 1

MODE = "safe"           # "naive" or "safe"

def fractal_dimension(img):
    bw = img > img.mean()
    sizes = [2, 4, 8, 16, 32]
    counts = []
    for s in sizes:
        cnt = 0
        for i in range(0, bw.shape[0], s):
            for j in range(0, bw.shape[1], s):
                if bw[i:i+s, j:j+s].any():
                    cnt += 1
        counts.append(cnt)
    x = np.log(1 / np.array(sizes))
    y = np.log(np.array(counts) + 1e-6)
    return np.polyfit(x, y, 1)[0]

def predict_time(model, device, z_vec):
    # 用回归头预测 time（你模型里 ctr_head 接 latent）
    z = torch.tensor(z_vec, device=device).unsqueeze(0)
    with torch.no_grad():
        ctr = model.ctr_head(z)
    return float(ctr[0, TIME_IDX].detach().cpu().float())

# def is_decode_weird(img):
#     # 简单异常检测：像素范围/饱和
#     mn, mx = float(img.min()), float(img.max())
#     if mn < PIX_CLIP_MIN or mx > PIX_CLIP_MAX:
#         return True
#     # 太接近常数也算异常
#     if float(img.std()) < 1e-3:
#         return True
#     return False

def save_step(out_dir, step, img, z, t, fd, score):
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="coolwarm")
    plt.colorbar(fraction=0.046)
    plt.title(f"step={step}  score={score:.3f}\nt={t:.3f}, FD={fd:.3f}, ||z||={np.linalg.norm(z):.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"step_{step:03d}.png"), dpi=200)
    plt.close()


def main():
    ensure_dir(OUT_DIR)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.eval()

    # === （可选）加载 dataset latent，用于投影回流形 ===
    Z_data = None
    knn = None
    if USE_PROJ:
        dataset = DendritePFMDataset((3, IMAGE_SIZE, IMAGE_SIZE), SPLIT_JSON, split=SPLIT)
        loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        Z = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                out = model(x)
                # 兼容你之前的返回：(_, mu, _, ctr, z) 这种
                z = out[-1]
                Z.append(z.detach().cpu().numpy())
        Z_data = np.concatenate(Z, axis=0)
        knn = NearestNeighbors(n_neighbors=KNN_K, metric="euclidean").fit(Z_data)

    run_dir = ensure_dir(os.path.join(OUT_DIR, f"run_{MODE}"))

    # === 初始化：直接从 prior 采样 ===
    # 这一步本身就可能生成很差的图（取决于你的 VAE prior match 是否好）
    with torch.no_grad():
        out = model(get_init_tensor().unsqueeze(0).to(device))
    img = out[0][0, 0].detach().cpu().numpy()
    z = out[-1][0].detach().cpu()

    # 记录初始
    t = predict_time(model, device, z)
    fd = fractal_dimension(img)
    score = heuristic_score(img)
    save_step(run_dir, 0, img, z, t, fd, score)

    for step in range(1, STEPS + 1):
        best = None
        best_score = -1e18
        best_img = None
        best_t = None
        best_fd = None

        # 生成候选
        for _ in range(NUM_CAND):

            dz = np.random.randn(*z.shape).astype(np.float32) * RW_SIGMA
            z_cand = z + dz

            # safe: 限 norm，避免飞太远
            if MODE == "safe":
                nrm = np.linalg.norm(z_cand)
                if nrm > MAX_NORM:
                    z_cand = z_cand / (nrm + 1e-12) * MAX_NORM

            img_cand = decode_single_channel(model, device, z_cand)

            # # safe: 拒绝明显崩坏的 decode
            # if MODE == "safe" and is_decode_weird(img_cand):
            #     continue

            t_cand = predict_time(model, device, z_cand)
            fd_cand = fractal_dimension(img_cand)

            s = heuristic_score(img_cand)
            if MODE == "safe":
                s = s - PRIOR_L2_W * (np.linalg.norm(z_cand) ** 2)

            if s > best_score:
                best_score = s
                best = z_cand
                best_img = img_cand
                best_t = t_cand
                best_fd = fd_cand

        if best is None:
            print("[Stop] no valid candidate (all rejected).")
            break

        z = best

        # safe: 定期投影回 dataset latent 的最近邻附近（非常有效）
        if MODE == "safe" and USE_PROJ and (step % PROJ_EVERY == 0):
            _, idx = knn.kneighbors(z.reshape(1, -1))
            z_nn = Z_data[idx[0, 0]]
            z = PROJ_ALPHA * z + (1 - PROJ_ALPHA) * z_nn

        save_step(run_dir, step, best_img, z, best_t, best_fd, best_score)

    print("Done. Saved to:", run_dir)


if __name__ == "__main__":
    main()
