import torch.nn as nn
import torch.nn.functional as F

class FingerprintNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*64, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*64*64, 128)
        )

    def forward(self, x1, x2):
        out1 = self.embedding(x1)
        out2 = self.embedding(x2)
        return out1, out2

def contrastive_loss(out1, out2, label, margin=1.0):
    dist = F.pairwise_distance(out1, out2)
    loss = (label * dist**2 + (1-label) * F.relu(margin - dist)**2).mean()
    return loss
