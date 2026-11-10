"""Post-hoc zero-day (unknown-attack) scores and open-set metrics (extension #13).

Convention: every score is "higher = more likely unknown".
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import f1_score, roc_auc_score


def msp_score(logits: np.ndarray) -> np.ndarray:
    """Max-softmax-probability baseline (Hendrycks & Gimpel, 2017)."""
    return -softmax(logits, axis=1).max(axis=1)


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy score (Liu et al., NeurIPS 2020): E(x) = -T * logsumexp(f(x)/T)."""
    return -T * logsumexp(logits / T, axis=1)


class Mahalanobis:
    """Class-conditional Gaussians with a tied covariance on penultimate features
    (Lee et al., NeurIPS 2018, without input pre-processing)."""

    def __init__(self, shrinkage: float = 1e-3):
        self.shrinkage = shrinkage

    def fit(self, Z: np.ndarray, y: np.ndarray, n_classes: int) -> "Mahalanobis":
        Z = Z.astype(np.float64)
        self.means = np.stack([Z[y == c].mean(0) if np.any(y == c) else Z.mean(0)
                               for c in range(n_classes)])
        centred = Z - self.means[y]
        cov = centred.T @ centred / max(len(Z) - 1, 1)
        cov += self.shrinkage * (np.trace(cov) / cov.shape[0] + 1e-12) * np.eye(cov.shape[0])
        self.prec = np.linalg.pinv(cov)
        return self

    def distances(self, Z: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance to every class mean, shape (N, C)."""
        Z = Z.astype(np.float64)
        out = np.empty((len(Z), len(self.means)))
        for c, mu in enumerate(self.means):
            d = Z - mu
            out[:, c] = np.einsum("ij,jk,ik->i", d, self.prec, d)
        return out

    def score(self, Z: np.ndarray) -> np.ndarray:
        return self.distances(Z).min(axis=1)

    def nearest_class(self, Z: np.ndarray) -> np.ndarray:
        return self.distances(Z).argmin(axis=1)


def compute_scores(method: str, logits: np.ndarray, Z: np.ndarray,
                   maha: Optional[Mahalanobis] = None) -> np.ndarray:
    if method == "msp":
        return msp_score(logits)
    if method == "energy":
        return energy_score(logits)
    if method == "mahalanobis":
        assert maha is not None, "fit Mahalanobis on training features first"
        return maha.score(Z)
    raise ValueError(method)


def auroc(score_known: np.ndarray, score_unknown: np.ndarray) -> float:
    """AUROC with unknown = positive class."""
    y = np.r_[np.zeros(len(score_known)), np.ones(len(score_unknown))]
    s = np.r_[score_known, score_unknown]
    if len(np.unique(y)) < 2 or not np.all(np.isfinite(s)):
        return 0.5
    return float(roc_auc_score(y, s))


def fpr_at_95tpr(score_known: np.ndarray, score_unknown: np.ndarray) -> float:
    """Standard OOD metric: threshold accepting 95% of known samples; return the
    fraction of unknown samples that are (wrongly) accepted as known."""
    thr = np.percentile(score_known, 95)
    return float(np.mean(score_unknown <= thr))


def class_thresholds(score_cal: np.ndarray, pred_cal: np.ndarray, n_known: int, tpr: float = 95.0,
                     min_count: int = 20) -> np.ndarray:
    """Per-predicted-class thresholds: the `tpr` percentile of calibration scores among samples the
    classifier assigns to each class (global percentile for classes with < min_count samples).
    A single global threshold sacrifices rare known classes whose scores are naturally high."""
    glob_thr = np.percentile(score_cal, tpr)
    return np.array([np.percentile(score_cal[pred_cal == c], tpr) if (pred_cal == c).sum() >= min_count
                     else glob_thr for c in range(n_known)])


def open_set_report(score_known_val: np.ndarray, score_known_test: np.ndarray,
                    score_unknown_test: np.ndarray, y_known_test: np.ndarray,
                    pred_known_test: np.ndarray, pred_unknown_test: np.ndarray,
                    n_known: int, tpr: float = 95.0, pred_known_val: np.ndarray = None) -> Dict[str, float]:
    """Thresholds fixed on *calibration* known data (no test information used). With
    `pred_known_val` given, thresholds are per predicted class (see class_thresholds)."""
    if pred_known_val is None:
        thr = np.percentile(score_known_val, tpr)
        thr_k = thr_u = thr
    else:
        t = class_thresholds(score_known_val, pred_known_val, n_known, tpr)
        thr, thr_k, thr_u = t.tolist(), t[pred_known_test], t[pred_unknown_test]
    rej_known = score_known_test > thr_k
    rej_unknown = score_unknown_test > thr_u
    # open-set labels: known ids 0..K-1, unknown -> K
    y_true = np.r_[y_known_test, np.full(len(score_unknown_test), n_known)]
    y_pred = np.r_[np.where(rej_known, n_known, pred_known_test),
                   np.where(rej_unknown, n_known, pred_unknown_test)]
    return {
        "auroc": auroc(score_known_test, score_unknown_test),
        "fpr_at_95tpr": fpr_at_95tpr(score_known_test, score_unknown_test),
        "unknown_detection_rate": float(rej_unknown.mean()),
        "known_false_alarm_rate": float(rej_known.mean()),
        "closed_set_macro_f1": float(f1_score(y_known_test, pred_known_test, average="macro")),
        "known_macro_f1_with_rejection": float(
            f1_score(y_known_test, np.where(rej_known, n_known, pred_known_test),
                     labels=list(range(n_known)), average="macro", zero_division=0)),
        "open_set_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "threshold": thr if isinstance(thr, list) else float(thr),
    }


class ScoreFusion:
    """Fuse non-negative 'higher = more unknown' scores (Mahalanobis distance, reconstruction
    error). Each is log-transformed (both are heavy-tailed), z-normalised with statistics of the
    known-class *calibration* split, and the z-scores are averaged. No test/unknown data is used."""

    @staticmethod
    def _log(s):
        return np.log(np.maximum(np.asarray(s, dtype=np.float64), 0.0) + 1e-12)

    def fit(self, cal_scores):
        self.stats = [(self._log(s).mean(), self._log(s).std() + 1e-12) for s in cal_scores]
        return self

    def score(self, scores):
        return np.mean([(self._log(s) - mu) / sd for s, (mu, sd) in zip(scores, self.stats)], axis=0)
