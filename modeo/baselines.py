"""Hand-designed baselines: fixed CNN (reuses BlockCNN), LSTM classifier, and an
autoencoder anomaly detector for the zero-day comparison."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .encoding import ConvGene, Genome, SettingGene
from .train import get_device


def handcrafted_cnn_genome() -> Genome:
    """3 chained blocks of 64 ReLU 3-wide kernels with 3-wide max-pooling, Adam lr=1e-3, CE."""
    conv = ConvGene(knum=3, ksize=0, kact=6, pt=0, ps=0)
    return Genome(n=3, topo=[1, 0, 1], convs=[conv, conv, conv],
                  setting=SettingGene(bs=0, opt=1, lr=3, reg=0, loss=0))


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)
        self.embed_dim = hidden

    def features(self, x):
        _, (h, _) = self.lstm(x.unsqueeze(-1))
        return h[-1]

    def forward(self, x):
        return self.head(self.features(x))

    def forward_with_features(self, x):
        z = self.features(x)
        return self.head(z), z


class AutoEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, code: int = 16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, code))
        self.dec = nn.Sequential(nn.ReLU(), nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x):
        return self.dec(self.enc(x))


def _fit(model, X, y, epochs, batch, lr, loss_fn, device, seed):
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tx = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(tx, torch.tensor(y, dtype=torch.long)) if y is not None else TensorDataset(tx)
    for _ in range(epochs):
        for b in DataLoader(ds, batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed)):
            opt.zero_grad()
            xb = b[0].to(device)
            loss = loss_fn(model(xb), b[1].to(device) if y is not None else xb)
            loss.backward()
            opt.step()
    return model.eval()


def train_lstm(X, y, n_classes, epochs=30, batch=256, lr=1e-3, seed=0):
    return _fit(LSTMClassifier(X.shape[1], n_classes), X, y, epochs, batch, lr, nn.CrossEntropyLoss(),
                get_device(), seed)


def train_autoencoder(X, epochs=30, batch=256, lr=1e-3, seed=0):
    return _fit(AutoEncoder(X.shape[1]), X, None, epochs, batch, lr, nn.MSELoss(), get_device(), seed)


@torch.no_grad()
def autoencoder_score(ae: AutoEncoder, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    device = next(ae.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i : i + batch], dtype=torch.float32, device=device)
        out.append(((ae(xb) - xb) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out)
