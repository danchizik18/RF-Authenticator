import torch
import numpy as np
from model import FingerprintNet, SiameseNet

classifier = FingerprintNet(num_classes=3)
classifier.load_state_dict(torch.load('models/classifier.pt'))
classifier.eval()

siamese = SiameseNet()
siamese.load_state_dict(torch.load('models/siamese.pt'))
siamese.eval()

prototypes = {
    0: torch.randn(128),
    1: torch.randn(128),
    2: torch.randn(128)
}

def authenticate(sample, threshold=1.0):
    sample = torch.tensor(sample).float().unsqueeze(0)
    pred = classifier(sample).argmax().item()
    emb, _ = siamese(sample, sample)
    dist = torch.norm(emb - prototypes[pred])
    return pred, dist.item(), dist.item() < threshold