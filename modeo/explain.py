"""Explainability for selected Pareto models (extension #15).

  * attributions: Integrated Gradients and GradientSHAP (Captum)
  * global importance: mean |attribution| per feature, per predicted attack class
  * faithfulness: deletion / insertion curves (+ area under each curve)
  * stability: Kendall tau between feature rankings (runs / Pareto models)
  * zero-day alert explanation: nearest known class (Mahalanobis) + top features
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.stats import kendalltau

from .model import BlockCNN
from .ood import Mahalanobis


def attributions(model: BlockCNN, X: np.ndarray, target: np.ndarray, method: str = "ig",
                 baseline: Optional[np.ndarray] = None, n_steps: int = 32,
                 n_samples: int = 20, batch: int = 256, seed: int = 0) -> np.ndarray:
    """Per-sample, per-feature attributions w.r.t. the logit of `target`."""
    from captum.attr import GradientShap, IntegratedGradients

    device = next(model.parameters()).device
    model.eval()
    torch.manual_seed(seed)
    base = torch.tensor(baseline if baseline is not None else np.zeros(X.shape[1]),
                        dtype=torch.float32, device=device).unsqueeze(0)
    out = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i : i + batch], dtype=torch.float32, device=device)
        tb = torch.tensor(target[i : i + batch], dtype=torch.long, device=device)
        if method == "ig":
            a = IntegratedGradients(model).attribute(xb, baselines=base.expand_as(xb), target=tb,
                                                     n_steps=n_steps)
        elif method == "gradshap":
            noise_base = base + 0.1 * torch.randn(n_samples, X.shape[1], device=device)
            a = GradientShap(model).attribute(xb, baselines=noise_base, target=tb, n_samples=n_samples,
                                              stdevs=0.01)
        else:
            raise ValueError(method)
        out.append(a.detach().cpu().numpy())
    return np.concatenate(out)


def global_importance(attr: np.ndarray, classes: np.ndarray, n_classes: int) -> Dict[str, np.ndarray]:
    """Mean |attribution| overall and per class."""
    res = {"overall": np.abs(attr).mean(0)}
    for c in range(n_classes):
        m = classes == c
        if m.any():
            res[str(c)] = np.abs(attr[m]).mean(0)
    return res


@torch.no_grad()
def _target_prob(model: BlockCNN, X: np.ndarray, target: np.ndarray, batch: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    ps = []
    for i in range(0, len(X), batch):
        lo = model(torch.tensor(X[i : i + batch], dtype=torch.float32, device=device))
        p = torch.softmax(lo, 1).gather(1, torch.tensor(target[i : i + batch], device=device).unsqueeze(1))
        ps.append(p.squeeze(1).cpu().numpy())
    return np.concatenate(ps)


def deletion_insertion(model: BlockCNN, X: np.ndarray, target: np.ndarray, attr: np.ndarray,
                       baseline: np.ndarray, steps: Optional[Sequence[int]] = None,
                       random_ranking: bool = False, seed: int = 0) -> Dict[str, np.ndarray]:
    """Deletion: replace the top-k features (by |attr|) with the baseline and track the
    target-class probability (faster drop = more faithful). Insertion: start from the
    baseline and restore the top-k features (faster rise = more faithful).
    `random_ranking=True` gives the random-order reference curve."""
    d = X.shape[1]
    steps = list(range(0, d + 1)) if steps is None else list(steps)
    if random_ranking:
        rng = np.random.default_rng(seed)
        order = np.stack([rng.permutation(d) for _ in range(len(X))])
    else:
        order = np.argsort(-np.abs(attr), axis=1)
    dele, ins = [], []
    rows = np.arange(len(X))[:, None]
    for k in steps:
        Xd = X.copy()
        Xi = np.tile(baseline, (len(X), 1)).astype(np.float32)
        if k > 0:
            top = order[:, :k]
            Xd[rows, top] = baseline[top]
            Xi[rows, top] = X[rows, top]
        dele.append(_target_prob(model, Xd, target).mean())
        ins.append(_target_prob(model, Xi, target).mean())
    frac = np.array(steps) / d
    dele, ins = np.array(dele), np.array(ins)
    return {"fraction": frac, "deletion": dele, "insertion": ins,
            "deletion_auc": float(np.trapezoid(dele, frac)), "insertion_auc": float(np.trapezoid(ins, frac))}


def ranking_stability(importances: List[np.ndarray], top_k: Optional[int] = None) -> Dict[str, float]:
    """Pairwise Kendall tau between global feature-importance vectors, plus top-k overlap."""
    taus, overlaps = [], []
    for i in range(len(importances)):
        for j in range(i + 1, len(importances)):
            a, b = importances[i], importances[j]
            taus.append(kendalltau(a, b).statistic)
            if top_k:
                ta, tb = set(np.argsort(-a)[:top_k]), set(np.argsort(-b)[:top_k])
                overlaps.append(len(ta & tb) / top_k)
    res = {"kendall_tau_mean": float(np.nanmean(taus)) if taus else float("nan"),
           "kendall_tau_std": float(np.nanstd(taus)) if taus else float("nan"), "n_pairs": len(taus)}
    if top_k:
        res[f"top{top_k}_overlap_mean"] = float(np.mean(overlaps)) if overlaps else float("nan")
    return res


def explain_zero_day_alerts(model: BlockCNN, X_alert: np.ndarray, maha: Mahalanobis,
                            feature_names: Sequence[str], class_names: Sequence[str],
                            baseline: np.ndarray, top_k: int = 5, method: str = "ig") -> List[dict]:
    """For samples flagged as unknown: report the nearest known class in embedding space
    and the top-k features pushing the model toward that class."""
    from .train import predict

    _, Z = predict(model, X_alert)
    nearest = maha.nearest_class(Z)
    dist = maha.distances(Z)
    attr = attributions(model, X_alert, nearest, method=method, baseline=baseline)
    reports = []
    for i in range(len(X_alert)):
        order = np.argsort(-np.abs(attr[i]))[:top_k]
        reports.append({
            "nearest_known_class": class_names[nearest[i]],
            "mahalanobis_distance": float(dist[i, nearest[i]]),
            "top_features": [{"feature": feature_names[j], "value": float(X_alert[i, j]),
                              "attribution": float(attr[i, j])} for j in order],
        })
    return reports
