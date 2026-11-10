"""Data preparation and final evaluation shared by the scripts.

Split protocol (seeded, so every script reproduces the same split):
  full data -> 80% train / 20% test (stratified)
  test-unknown class U (LOAO): removed from train entirely; its test samples are the
      'unknown' test set. U is never seen by the search or by final training.
  train -> 90% fit / 10% calib   (calib = threshold calibration for open-set metrics)
  search uses a (sub-sampled) 80/20 split of `fit`; pseudo-unknown class P (!= U) is
      removed from the search-train split and its search-val samples feed f3.
      Final training uses all known classes, P included.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .data import (DEFAULT_NORM, DEFAULT_TRANSFORM, Dataset, apply_transform, load_dataset, normalise,
                   open_set_view, stratified_split, subsample)
from .encoding import Genome
from .ood import Mahalanobis, ScoreFusion, auroc, compute_scores, open_set_report
from .search import SearchData
from .train import get_device, predict, train_model

OOD_METHODS = ["msp", "energy", "mahalanobis"]
AE_EPOCHS = 30
_AE_CACHE: dict = {}


def fitted_autoencoder(p: "Prepared", seed: int):
    """Autoencoder on the known training classes; cached per data split (it does not depend on
    the CNN), so the two picked solutions of a run - and all runs of one process - share it."""
    from .baselines import train_autoencoder
    key = (p.ds.name, p.unknown, seed, p.norm, p.transform, len(p.y_fit))
    if key not in _AE_CACHE:
        _AE_CACHE[key] = train_autoencoder(p.X_fit, epochs=AE_EPOCHS, batch=256, seed=seed)
    return _AE_CACHE[key]


def resolve_class(ds: Dataset, value) -> Optional[int]:
    if value is None or value == "" or str(value).lower() == "none":
        return None
    if str(value).lstrip("-").isdigit():
        return int(value)
    names = [c.lower() for c in ds.class_names]
    if str(value).lower() not in names:
        raise ValueError(f"unknown class '{value}', choose from {ds.class_names}")
    return names.index(str(value).lower())


def hardest_pseudo_unknown(X: np.ndarray, y: np.ndarray, ds: Dataset, unknown: Optional[int],
                           seed: int = 0, max_fit: int = 20000) -> int:
    """Pick the pseudo-unknown class P for f3 using *training data only* (the test-unknown U is
    already removed from X, y): the attack class that a raw-feature Mahalanobis detector finds
    hardest to flag when it is hidden. An easy P (e.g. Recon on Gas Pipeline, raw AUROC = 1.000)
    makes f3 = 0 for every architecture, so the objective carries no signal."""
    rng = np.random.default_rng(seed)
    attacks = [c for c in range(len(ds.class_names)) if c not in (ds.normal_class, unknown) and np.any(y == c)]
    best, best_auc = attacks[-1], 2.0
    for c in attacks:
        known = [k for k in np.unique(y) if k != c]
        m = np.isin(y, known)
        idx = np.where(m)[0]
        idx = rng.permutation(idx)[: max_fit]
        tr, va = idx[: int(0.7 * len(idx))], idx[int(0.7 * len(idx)):]
        lut = {k: i for i, k in enumerate(known)}
        maha = Mahalanobis().fit(X[tr], np.array([lut[v] for v in y[tr]]), len(known))
        a = auroc(maha.score(X[va]), maha.score(X[y == c][: max_fit]))
        if a < best_auc:
            best, best_auc = c, a
    return best


def default_pseudo_unknown(ds: Dataset, unknown: Optional[int]) -> int:
    """Next attack class after U (cyclic), skipping the normal class."""
    attacks = [c for c in range(len(ds.class_names)) if c != ds.normal_class]
    if unknown is None:
        return attacks[-1]
    i = attacks.index(unknown)
    return attacks[(i + 1) % len(attacks)]


@dataclass
class Prepared:
    ds: Dataset
    norm: str
    X_fit: np.ndarray
    y_fit: np.ndarray      # original ids
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    unknown: Optional[int]
    pseudo_unknown: Optional[int]
    load_kw: Optional[dict] = None
    transform: Optional[str] = None


def prepare(dataset: str, data_path: Optional[str], seed: int, norm: Optional[str] = None,
            unknown=None, pseudo_unknown=None, use_f3: bool = False, transform: Optional[str] = None,
            **load_kw) -> Prepared:
    ds = load_dataset(dataset, data_path, **load_kw)
    norm = norm or DEFAULT_NORM.get(dataset, "zscore")
    transform = transform or DEFAULT_TRANSFORM.get(dataset, "none")
    X_all = apply_transform(ds.X, transform)
    X_tr, X_te, y_tr, y_te = stratified_split(X_all, ds.y, 0.2, seed)
    U = resolve_class(ds, unknown)
    if U is not None:
        keep = y_tr != U
        X_tr, y_tr = X_tr[keep], y_tr[keep]
    X_fit, X_cal, y_fit, y_cal = stratified_split(X_tr, y_tr, 0.1, seed)
    X_fit, X_cal, X_te = normalise(X_fit, X_cal, X_te, method=norm)
    if not use_f3 and pseudo_unknown in (None, "auto", "next"):
        P = None
    elif pseudo_unknown in (None, "auto"):
        P = hardest_pseudo_unknown(X_fit, y_fit, ds, U, seed)
    elif pseudo_unknown == "next":
        P = default_pseudo_unknown(ds, U)
    else:
        P = resolve_class(ds, pseudo_unknown)
    if P is not None and P == U:
        raise ValueError("pseudo-unknown class must differ from the test-unknown class")
    return Prepared(ds, norm, X_fit, y_fit, X_cal, y_cal, X_te, y_te, U, P, dict(load_kw), transform)


def search_data(p: Prepared, fraction: float, seed: int) -> tuple[SearchData, list]:
    hidden = [c for c in (p.unknown, p.pseudo_unknown) if c is not None]
    view = open_set_view(range(len(p.ds.class_names)), hidden)
    X, y = subsample(p.X_fit, p.y_fit, fraction, seed)
    X_a, X_b, y_a, y_b = stratified_split(X, y, 0.2, seed)
    ya, yb = view.remap(y_a), view.remap(y_b)
    tr, va = ya >= 0, yb >= 0
    X_unk = X_b[y_b == p.pseudo_unknown] if p.pseudo_unknown is not None else None
    return SearchData(X_a[tr], ya[tr], X_b[va], yb[va], view.n_known, X_unk), view.known_classes


def classification_metrics(y, pred) -> Dict[str, float]:
    return {"accuracy": accuracy_score(y, pred),
            "precision_weighted": precision_score(y, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y, pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0)}


def final_evaluate(genome: Genome, p: Prepared, epochs: int, base_batch: int, seed: int,
                   save_path: Optional[str] = None, loss_override: Optional[str] = None) -> dict:
    """Fully train a genome on all known training classes and test it (closed + open set)."""
    hidden = [p.unknown] if p.unknown is not None else []
    view = open_set_view(range(len(p.ds.class_names)), hidden)
    y_fit = view.remap(p.y_fit)
    model, tinfo = train_model(genome, p.X_fit, y_fit, view.n_known, epochs, base_batch, seed=seed,
                               loss_override=loss_override)
    res = {"genome": genome.as_list(), "search_fitness": genome.fitness, "train": tinfo,
           "mflops": model.mflops()}
    res.update(evaluate_trained(model, p, seed))
    if save_path:
        torch.save({"genome": genome.as_list(), "use_loss_gene": genome.use_loss_gene,
                    "state_dict": model.state_dict(), "n_features": p.X_fit.shape[1],
                    "n_classes": view.n_known, "known_classes": view.known_classes,
                    "feature_names": p.ds.feature_names, "class_names": p.ds.class_names,
                    "dataset": p.ds.name, "unknown": p.unknown, "seed": seed, "norm": p.norm, "load_kw": p.load_kw or {},
                    "transform": p.transform}, save_path)
    return res


def evaluate_trained(model, p: Prepared, seed: int) -> dict:
    """Closed-set metrics on known test classes; open-set metrics for all post-hoc scores
    when a test-unknown class exists. `model` must expose forward_with_features()."""
    hidden = [p.unknown] if p.unknown is not None else []
    view = open_set_view(range(len(p.ds.class_names)), hidden)
    y_fit = view.remap(p.y_fit)
    res = {"known_classes": [p.ds.class_names[c] for c in view.known_classes]}
    y_te = view.remap(p.y_test)
    known_te = y_te >= 0
    L_te, Z_te = predict(model, p.X_test)
    res["closed_set"] = classification_metrics(y_te[known_te], L_te[known_te].argmax(1))
    if p.unknown is not None:
        L_cal, Z_cal = predict(model, p.X_cal)
        idx = np.random.default_rng(seed).permutation(len(p.X_fit))[:20000]
        _, Z_fit = predict(model, p.X_fit[idx])
        maha = Mahalanobis().fit(Z_fit, y_fit[idx], view.n_known)
        unk = ~known_te
        res["open_set"] = {}
        cal, te = {}, {}
        for m in OOD_METHODS:
            cal[m] = compute_scores(m, L_cal, Z_cal, maha)
            te[m] = compute_scores(m, L_te, Z_te, maha)
        # autoencoder reconstruction error and its fusion with the CNN's Mahalanobis score
        from .baselines import autoencoder_score
        ae = fitted_autoencoder(p, seed)
        cal["ae"], te["ae"] = autoencoder_score(ae, p.X_cal), autoencoder_score(ae, p.X_test)
        fus = ScoreFusion().fit([cal["mahalanobis"], cal["ae"]])
        cal["fusion"] = fus.score([cal["mahalanobis"], cal["ae"]])
        te["fusion"] = fus.score([te["mahalanobis"], te["ae"]])
        pred_cal = L_cal.argmax(1)
        for m in list(cal):
            args = (cal[m], te[m][known_te], te[m][unk], y_te[known_te], L_te[known_te].argmax(1),
                    L_te[unk].argmax(1), view.n_known)
            res["open_set"][m] = open_set_report(*args)                            # global threshold
            res["open_set"][m + "_pc"] = open_set_report(*args, pred_known_val=pred_cal)  # per-class
        res["unknown_class"] = p.ds.class_names[p.unknown]
    return res


def load_trained(path: str):
    from .model import build_model
    ck = torch.load(path, map_location=get_device(), weights_only=False)
    g = Genome.from_list(ck["genome"], ck["use_loss_gene"])
    model = build_model(g, ck["n_features"], ck["n_classes"]).to(get_device())
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def dump(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=conv)
