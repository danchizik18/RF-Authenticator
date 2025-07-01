import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class SignalDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                self.samples.append(os.path.join(cls_dir, f))
                self.labels.append(self.class_to_idx[cls])

    def __getitem__(self, index):
        x = np.load(self.samples[index])
        x = torch.tensor(x).float().unsqueeze(0)  # (1, H, W)
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.samples)

class SiameseDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.class_indices = {cls: [] for cls in set(self.base.labels)}
        for idx, label in enumerate(self.base.labels):
            self.class_indices[label].append(idx)

    def __getitem__(self, index):
        x1, y1 = self.base[index]
        same_class = random.choice([True, False])
        if same_class:
            idx2 = random.choice(self.class_indices[y1])
        else:
            idx2 = random.choice([i for i in range(len(self.base)) if self.base.labels[i] != y1])
        x2, y2 = self.base[idx2]
        label = 1 if same_class else 0
        return x1, x2, torch.tensor(label).float()

    def __len__(self):
        return len(self.base)
