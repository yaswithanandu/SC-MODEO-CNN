"""Training / inference helpers shared by the search and the final evaluation."""
from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .encoding import Genome
from .losses import make_loss
from .model import BlockCNN, build_model

REG_COEF = 1e-4  # L1/L2 penalty strength (not given in the paper; assumption)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(name: str, params, lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adadelta":
        return torch.optim.Adadelta(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(name)


def _penalty(model: nn.Module, kind: str) -> torch.Tensor:
    if kind == "none":
        return torch.zeros((), device=next(model.parameters()).device)
    weights = [p for n, p in model.named_parameters() if p.dim() > 1]
    l1 = sum(p.abs().sum() for p in weights)
    l2 = sum((p ** 2).sum() for p in weights)
    return REG_COEF * {"l1": l1, "l2": l2, "l1l2": l1 + l2}[kind]


def train_model(genome: Genome, X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int,
                base_batch: int, seed: int = 0, loss_override: str | None = None,
                device: torch.device | None = None, verbose: bool = False) -> Tuple[BlockCNN, dict]:
    """Train the CNN decoded from `genome`. Returns (model, info).

    Baseline MODEO-CNN always uses cross-entropy (loss gene absent); SC mode uses the
    loss chosen by the genome's 5th setting gene.
    """
    device = device or get_device()
    torch.manual_seed(seed)
    model = build_model(genome, X.shape[1], n_classes).to(device)
    s = genome.setting
    loss_name = loss_override or (s.loss_name if genome.use_loss_gene else "ce")
    counts = np.bincount(y, minlength=n_classes)
    criterion = make_loss(loss_name, counts, device)
    opt = make_optimizer(s.optimizer, model.parameters(), s.learning_rate)
    bs = s.batch_size(base_batch)
    # whole training set lives on the device; shuffled index batches avoid DataLoader overhead
    Xd = torch.tensor(X, dtype=torch.float32, device=device)
    yd = torch.tensor(y, dtype=torch.long, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0, diverged, last = time.time(), False, float("nan")
    model.train()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        perm = torch.randperm(len(Xd), generator=g).to(device)
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            xb, yb = Xd[idx], yd[idx]
            opt.zero_grad()
            loss = criterion(model(xb), yb) + _penalty(model, s.regularizer)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if diverged:
            break
        last = tot / max(nb, 1)
        if verbose:
            print(f"  epoch {ep + 1}/{epochs} loss={last:.4f}")
    model.eval()
    return model, {"loss": loss_name, "final_train_loss": last, "diverged": diverged,
                   "train_seconds": time.time() - t0, "batch_size": bs}


@torch.no_grad()
def predict(model: BlockCNN, X: np.ndarray, batch: int = 4096,
            device: torch.device | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (logits, penultimate features) as numpy arrays."""
    device = device or next(model.parameters()).device
    model.eval()
    logits, feats = [], []
    Xt = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(Xt), batch):
        lo, z = model.forward_with_features(Xt[i : i + batch].to(device))
        logits.append(lo.cpu())
        feats.append(z.cpu())
    L = torch.cat(logits).numpy()
    Z = torch.cat(feats).numpy()
    L[~np.isfinite(L)] = 0.0
    Z[~np.isfinite(Z)] = 0.0
    return L, Z
