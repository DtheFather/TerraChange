import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.t1 = os.path.join(root_dir, "t1")
        self.t2 = os.path.join(root_dir, "t2")
        self.masks = os.path.join(root_dir, "masks")

        self.files = sorted(os.listdir(self.t1))
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def _load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img)

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = (mask > 0).astype(np.float32)
        return torch.tensor(mask).unsqueeze(0)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img1 = self._load_img(os.path.join(self.t1, fname))
        img2 = self._load_img(os.path.join(self.t2, fname))
        mask = self._load_mask(os.path.join(self.masks, fname))
        return img1, img2, mask
