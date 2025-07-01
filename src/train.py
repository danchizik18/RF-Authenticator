import torch
from torch.utils.data import DataLoader
from dataset import SignalDataset, SiameseDataset
from model import FingerprintNet, SiameseNet, contrastive_loss
import torch.nn.functional as F
import torch.optim as optim
import os

def train_classifier(data_dir, out_path):
    ds = SignalDataset(data_dir)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = FingerprintNet(num_classes=len(ds.classes))
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        for xb, yb in dl:
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), out_path)

def train_siamese(data_dir, out_path):
    ds = SiameseDataset(SignalDataset(data_dir))
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    model = SiameseNet()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        for x1, x2, y in dl:
            emb1, emb2 = model(x1, x2)
            loss = contrastive_loss(emb1, emb2, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), out_path)

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    train_classifier('data/spectra', 'models/classifier.pt')
    train_siamese('data/spectra', 'models/siamese.pt')