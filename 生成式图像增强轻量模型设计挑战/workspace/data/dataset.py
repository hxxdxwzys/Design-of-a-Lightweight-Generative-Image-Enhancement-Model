"""训练数据集：HQ 图像 + 在线同分辨率退化 -> (lq, hq) 张量对 [-1,1]"""
import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from data.degradation import degrade

IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.PNG', '*.JPG')


def to_tensor_norm(img_pil):
    arr = np.asarray(img_pil, dtype=np.float32) / 127.5 - 1.0  # [-1,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class HQDegradationDataset(Dataset):
    def __init__(self, data_dir, patch_size=256, augment=True):
        self.paths = []
        for ext in IMG_EXTS:
            self.paths += glob.glob(os.path.join(data_dir, '**', ext), recursive=True)
        self.paths = sorted(self.paths)
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _load(self, idx):
        try:
            return Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            return Image.open(random.choice(self.paths)).convert('RGB')

    def __getitem__(self, idx):
        img = self._load(idx)
        w, h = img.size
        ps = self.patch_size
        if w < ps or h < ps:
            img = img.resize((max(ps, w), max(ps, h)), Image.BICUBIC)
            w, h = img.size
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        hq = img.crop((x, y, x + ps, y + ps))
        if self.augment:
            if random.random() < 0.5:
                hq = hq.transpose(Image.FLIP_LEFT_RIGHT)
        lq = degrade(hq)
        return to_tensor_norm(lq), to_tensor_norm(hq)
