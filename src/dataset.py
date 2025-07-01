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