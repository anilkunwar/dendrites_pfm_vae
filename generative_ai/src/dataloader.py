import json, os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class DendritePFMDataset(Dataset):
    def __init__(self, image_size, json_path, split="train", meta_path="", transform=None, device="cuda"):
        """
        参数:
            json_path: str, JSON 文件路径 (包含 train/val/test 列表)
            split: str, 指定要读取的数据集 ("train" / "val" / "test")
            transform: 可选，对每个样本的变换函数
            device: str, 默认 'cpu'
        """
        with open(json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = f.readline()

        assert split in splits
        self.files = splits[split]

        image_size = (image_size[1], image_size[2])
        self.resize = transforms.Resize(image_size)
        self.transform = transform

        self.meta = [float(e) for e in meta.strip().split(" ")]
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]
        arr = np.load(path)  # shape (H, W, 3)

        tensor = torch.from_numpy(arr).float().permute(2, 0, 1)  # -> (3, H, W)
        tensor = self.resize(tensor)

        if self.transform:
            tensor = self.transform(tensor)

        # build control variable
        base = os.path.basename(path)
        name_no_ext = os.path.splitext(base)[0]
        name_id = float(name_no_ext)
        c = [name_id]
        c += self.meta

        return tensor.to(self.device), torch.tensor(c, dtype=torch.float32)