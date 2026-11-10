"""Dataset loading, normalisation, splits and open-set (leave-one-attack-out) helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

GAS_PIPELINE_CLASSES = {0: "Normal", 1: "NMRI", 2: "CMRI", 3: "MSCI", 4: "MPCI", 5: "MFCI",
                        6: "DoS", 7: "Recon"}
# default normalisation per dataset. Paper Sec. V-B uses L2 for Gas Pipeline, but in the 2014
# file `measurement` reaches +-2.3e38, which makes L2 / z-score / min-max all degenerate: with L2
# only ~4% of random genomes learned anything at the search budget. We therefore apply a signed
# log transform first and then z-score (see README, "Preprocessing deviation").
DEFAULT_NORM = {"gas_pipeline": "zscore", "bot_iot": "minmax", "power_system": "zscore",
                "edge_iiotset": "minmax", "synthetic": "zscore"}
DEFAULT_TRANSFORM = {"gas_pipeline": "slog"}


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    name: str
    normal_class: Optional[int] = 0
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- readers
def _read_table(path: str) -> pd.DataFrame:
    with open(path, "rb") as fh:
        head = fh.read(512).lower()
    if b"<html" in head or b"<!doctype html" in head:
        raise ValueError(f"{path} is a web page (e.g. a saved redirect), not a data file — "
                         "download the real file from the link inside it")
    if path.lower().endswith(".arff"):
        from scipy.io import arff
        try:
            data, _ = arff.loadarff(path)
            df = pd.DataFrame(data)
        except Exception:  # scipy cannot parse some ARFF dialects -> minimal manual parser
            df = _read_arff_manual(path)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
        return df
    return pd.read_csv(path, low_memory=False)


def _read_arff_manual(path: str) -> pd.DataFrame:
    cols, rows, in_data = [], [], False
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if in_data:
                rows.append([v.strip().strip("'\"") for v in s.split(",")])
            elif s.lower().startswith("@attribute"):
                name = s.split(None, 2)[1].strip("'\"")
                cols.append(name)
            elif s.lower().startswith("@data"):
                in_data = True
    return pd.DataFrame(rows, columns=cols)


def _to_numeric_frame(df: pd.DataFrame, max_categories: int = 1000) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        col = pd.to_numeric(df[c].replace("?", np.nan), errors="coerce")
        if col.notna().mean() > 0.95:
            out[c] = col.fillna(col.median() if col.notna().any() else 0.0)
        elif df[c].nunique() <= max_categories:  # low-cardinality string -> integer codes
            out[c] = df[c].astype("category").cat.codes.astype(float)
        # high-cardinality strings (IPs, payloads, timestamps) are dropped
    return pd.DataFrame(out)


def _finish(df: pd.DataFrame, label_col: str, drop: Sequence[str], name: str,
            class_map: Optional[Dict[int, str]] = None, normal_label=None) -> Dataset:
    drop = [c for c in drop if c in df.columns and c != label_col]
    labels = df[label_col].astype(str).str.strip()
    feats = _to_numeric_frame(df.drop(columns=drop + [label_col]))
    feats = feats.loc[:, feats.nunique() > 1]  # constant columns carry no information
    if class_map is not None:
        lab_num = pd.to_numeric(labels, errors="coerce").astype(int)
        uniq = sorted(lab_num.unique())
        y = np.searchsorted(uniq, lab_num.values)
        class_names = [class_map.get(u, str(u)) for u in uniq]
    else:
        uniq = sorted(labels.unique(), key=lambda s: (s.lower() != str(normal_label).lower(), s))
        idx = {u: i for i, u in enumerate(uniq)}
        y = labels.map(idx).values
        class_names = list(uniq)
    normal = 0 if (class_map is not None and 0 in uniq) or normal_label is not None else None
    return Dataset(feats.values.astype(np.float32), y.astype(np.int64), list(feats.columns),
                   class_names, name, normal_class=normal)


# ----------------------------------------------------------------------------- datasets
def load_gas_pipeline(path: str, label_col: Optional[str] = None) -> Dataset:
    """Morris & Gao (2014) gas-pipeline ICS dataset (ARFF or CSV).

    Uses the 8-class 'categorized' label (0 Normal, 1 NMRI, 2 CMRI, 3 MSCI, 4 MPCI,
    5 MFCI, 6 DoS, 7 Recon). The binary / specific result columns are dropped (they
    would leak the label).
    """
    df = _read_table(path)
    lower = {c.lower(): c for c in df.columns}
    if label_col is None:
        for cand in ["categorized result", "categorized_result", "result", "label", "class"]:
            if cand in lower:
                label_col = lower[cand]
                break
    if label_col is None:
        raise ValueError(f"Could not find the label column in {list(df.columns)}")
    drop = [c for c in df.columns if c.lower() in
            {"binary result", "binary_result", "specific result", "specific_result"}]
    # 'time' is an inter-packet interval in the 2014 file (informative, e.g. for Recon) but an
    # absolute timestamp in other releases; drop it only when it is a timestamp (monotonic).
    for c in df.columns:
        if c.lower() == "time" and pd.to_numeric(df[c], errors="coerce").is_monotonic_increasing:
            drop.append(c)
    return _finish(df, label_col, drop, "gas_pipeline", class_map=GAS_PIPELINE_CLASSES)


def load_edge_iiotset(path: str, max_rows: Optional[int] = None, seed: int = 0) -> Dataset:
    """Edge-IIoTset (Ferrag et al., 2022) — DNN-EdgeIIoT-dataset.csv, label 'Attack_type'.
    `max_rows` draws a stratified subsample after de-duplication (the file has ~2.2M rows)."""
    df = _read_table(path)
    drop = ["Attack_label", "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
            "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
            "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport",
            "udp.port", "mqtt.msg"]
    df = df.drop_duplicates()
    if max_rows and len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, stratify=df["Attack_type"], random_state=seed)
    return _finish(df, "Attack_type", drop, "edge_iiotset", normal_label="Normal")


def load_bot_iot(path: str, max_rows: Optional[int] = None, seed: int = 0) -> Dataset:
    """BoT-IoT '10 best features' CSV, label 'category' (Normal, DDoS, DoS, Reconnaissance, Theft).
    `max_rows` draws a stratified subsample (the full set is ~3.7M rows)."""
    df = _read_table(path)
    if max_rows and len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, stratify=df["category"], random_state=seed)
    drop = ["pkSeqID", "attack", "subcategory", "saddr", "daddr", "sport", "dport", "proto",
            "flgs", "state"]
    return _finish(df, "category", drop, "bot_iot", normal_label="Normal")


def load_csv(path: str, label_col: str, drop: Sequence[str] = (), normal_label=None,
             name: str = "custom") -> Dataset:
    return _finish(_read_table(path), label_col, list(drop), name, normal_label=normal_label)


def make_synthetic(n: int = 6000, n_features: int = 26, n_classes: int = 8, seed: int = 0) -> Dataset:
    """Imbalanced stand-in with Gas-Pipeline-like shape, for smoke tests only."""
    from sklearn.datasets import make_classification
    weights = np.array([0.6, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02][:n_classes])
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=12, n_redundant=6,
                               n_classes=n_classes, n_clusters_per_class=1, weights=weights / weights.sum(),
                               class_sep=1.5, random_state=seed)
    names = [GAS_PIPELINE_CLASSES[i] for i in range(n_classes)]
    return Dataset(X.astype(np.float32), y.astype(np.int64), [f"f{i}" for i in range(n_features)],
                   names, "synthetic", normal_class=0)


def load_dataset(name: str, path: Optional[str] = None, **kw) -> Dataset:
    if name == "synthetic":
        return make_synthetic(**kw)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file for '{name}' not found: {path}")
    if name == "gas_pipeline":
        return load_gas_pipeline(path, **kw)
    if name == "edge_iiotset":
        return load_edge_iiotset(path, **kw)
    if name == "bot_iot":
        return load_bot_iot(path, **kw)
    raise ValueError(name)


# ----------------------------------------------------------------------------- preprocessing
def make_scaler(method: str):
    return {"zscore": StandardScaler(), "minmax": MinMaxScaler(), "l2": Normalizer(norm="l2")}[method]


def apply_transform(X: np.ndarray, transform: Optional[str]) -> np.ndarray:
    """Element-wise transform applied before scaling. 'slog' = sign(x) * log(1 + |x|):
    monotone (keeps order, so extreme injected values stay extreme) but tames +-1e38."""
    if transform in (None, "none"):
        return X
    if transform == "slog":
        X = X.astype(np.float64)
        return (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
    raise ValueError(transform)


def normalise(train: np.ndarray, *others: np.ndarray, method: str = "zscore"):
    sc = make_scaler(method).fit(train)
    out = [sc.transform(train).astype(np.float32)] + [sc.transform(o).astype(np.float32) for o in others]
    return out if others else out[0]


def stratified_split(X, y, test_size: float, seed: int):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def subsample(X, y, fraction: float, seed: int):
    if fraction >= 1.0:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=fraction, stratify=y, random_state=seed)
    return Xs, ys


# ----------------------------------------------------------------------------- open set
@dataclass
class OpenSetView:
    """Known/unknown view of a label vector after hiding some classes."""
    known_classes: List[int]          # original ids kept, in order
    hidden_classes: List[int]         # original ids treated as unknown

    def remap(self, y: np.ndarray) -> np.ndarray:
        """Original ids -> 0..K-1 for known classes, -1 for hidden ones."""
        lut = {c: i for i, c in enumerate(self.known_classes)}
        return np.array([lut.get(int(v), -1) for v in y], dtype=np.int64)

    @property
    def n_known(self) -> int:
        return len(self.known_classes)


def open_set_view(all_classes: Sequence[int], hidden: Sequence[int]) -> OpenSetView:
    hidden = [int(h) for h in hidden]
    return OpenSetView([int(c) for c in all_classes if int(c) not in hidden], hidden)
