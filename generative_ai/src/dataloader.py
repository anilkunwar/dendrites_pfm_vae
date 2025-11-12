import json, os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# param range -> (min, max)
PARAM_RANGES = {
    "POT_LEFT": (-0.5, -0.2),
    # "flo": (1e-3, 1e-1),
    # "fso": (1e-3, 1e-1),
    "fo": (1e-3, 1e-1),
    "Al": (0, 10),
    "Bl": (0, 10),
    "Cl": (0, 10),
    # "Dl": (-10, 10),
    "As": (0, 10),
    "Bs": (0, 10),
    "Cs": (0, 10),
    # "Ds": (-10, 10),
    "cleq": (0.01, 0.99),
    "cseq": (0.01, 0.99),
    "L1o": (0.01, 0.5),
    "L2o": (0.01, 0.5),
    "ko": (1e-11, 1e-9),
    "Noise": (5e-4, 5e-3)
}

def inverse_scale_params(params):
    normed = {}
    for key, val in params.items():
        lo, hi = PARAM_RANGES[key]
        normed[key] = (val - lo) / (hi - lo)
    return normed

def smooth_scale(x, k=0.3):
    return 0.5 + 0.5 * torch.tanh(k * x)

def inv_smooth_scale(y, k=0.3, eps=1e-6):
    y = torch.clamp(y, eps, 1 - eps)
    return 0.5 / k * torch.log(y / (1 - y))

class DendritePFMDataset(Dataset):
    def __init__(self, image_size, json_path, split="train", transform=None, device="cuda"):
        """
        参数:
            json_path: str, JSON 文件路径 (包含 train/val/test 列表)
            split: str, 指定要读取的数据集 ("train" / "val" / "test")
            transform: 可选，对每个样本的变换函数
            device: str, 默认 'cpu'
        """
        with open(json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        self.meta_dict = {}

        assert split in splits
        self.files = splits[split]

        image_size = (image_size[1], image_size[2])
        self.resize = transforms.Resize(image_size)
        self.transform = transform

        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]
        arr = np.load(path)  # shape (H, W, 3)

        # assert arr.max() <= 5 and arr.min() >= -5, "Inappropriate values occured"

        tensor = torch.from_numpy(arr).float().permute(2, 0, 1)  # -> (3, H, W)
        tensor = self.resize(tensor)

        if self.transform:
            tensor = self.transform(tensor)

        tensor = smooth_scale(tensor)

        # build control variable
        base = os.path.basename(path)
        name_no_ext = os.path.splitext(base)[0]
        name_id = float(name_no_ext) / 5e3  # scale
        c = [name_id]
        # find meta
        sub_path = os.path.dirname(os.path.dirname(path))
        meta_path = os.path.join(sub_path, os.path.basename(sub_path) + ".json")
        if meta_path not in self.meta_dict:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.meta_dict[meta_path] = inverse_scale_params(meta).values()
        c += self.meta_dict[meta_path]

        return tensor.to(self.device), torch.tensor(c, dtype=torch.float32), os.path.basename(sub_path)

    def showSample(self):
        import matplotlib.pyplot as plt
        import random
        path = random.choice(self.files)
        arr = np.load(path)  # shape (H, W, 3)

        plt.figure(figsize=(3, 3))
        plt.imshow(arr[..., 0], cmap='coolwarm')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(3, 3))
        plt.imshow(arr[..., 1], cmap='coolwarm')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(3, 3))
        plt.imshow(arr[..., 2], cmap='coolwarm')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')
        plt.show()