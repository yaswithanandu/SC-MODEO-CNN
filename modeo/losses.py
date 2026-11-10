"""Loss functions selectable by the SC-MODEO-CNN `loss` gene (extension #5)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_frequency_weights(counts: np.ndarray) -> torch.Tensor:
    counts = np.maximum(counts.astype(float), 1.0)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    """Effective-number weights, Cui et al., CVPR 2019."""
    counts = np.maximum(counts.astype(float), 1.0)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / eff
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss, Lin et al., ICCV 2017 (no alpha term)."""

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        return (-(1 - p_t) ** self.gamma * logp_t).mean()


def make_loss(name: str, class_counts: np.ndarray, device) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=inverse_frequency_weights(class_counts).to(device))
    if name == "cb_ce":
        return nn.CrossEntropyLoss(weight=class_balanced_weights(class_counts).to(device))
    if name == "focal1":
        return FocalLoss(1.0)
    if name == "focal2":
        return FocalLoss(2.0)
    raise ValueError(name)
