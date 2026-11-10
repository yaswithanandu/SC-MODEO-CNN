"""Loader checks on a tiny fake ARFF that mimics the gas-pipeline column layout."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modeo.data import load_gas_pipeline, open_set_view
from modeo.experiment import prepare, search_data


def _fake_arff(path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    cols = ["address", "function", "length", "setpoint", "gain", "pressure measurement",
            "binary result", "categorized result", "specific result", "time"]
    with open(path, "w") as fh:
        fh.write("@relation gas\n")
        for c in cols:
            fh.write(f"@attribute '{c}' numeric\n")
        fh.write("@data\n")
        for _ in range(n):
            cat = rng.integers(0, 8)
            row = [4, rng.integers(1, 17), rng.integers(10, 90), "?" if rng.random() < 0.1 else rng.random() * 20,
                   rng.random() + cat, rng.random() * 5, int(cat > 0), cat, cat * 3, rng.random() * 1e6]
            fh.write(",".join(map(str, row)) + "\n")


def test_gas_pipeline_loader(tmp_path):
    p = tmp_path / "gas.arff"
    _fake_arff(p)
    ds = load_gas_pipeline(str(p))
    assert ds.class_names == ["Normal", "NMRI", "CMRI", "MSCI", "MPCI", "MFCI", "DoS", "Recon"]
    assert not {"binary result", "categorized result", "specific result"} & set(ds.feature_names)
    assert "time" in ds.feature_names        # random (interval-like) time column is kept
    assert "address" not in ds.feature_names  # constant column dropped
    assert np.isfinite(ds.X).all() and ds.X.shape[0] == 400


def test_monotonic_time_is_dropped(tmp_path):
    p = tmp_path / "gas_ts.arff"
    _fake_arff(p)
    lines = open(p).read().splitlines()
    start = lines.index("@data") + 1
    for k in range(start, len(lines)):  # overwrite 'time' (last numeric column) with a timestamp
        parts = lines[k].split(",")
        parts[-1] = str(1.0e6 + k)
        lines[k] = ",".join(parts)
    open(p, "w").write("\n".join(lines) + "\n")
    assert "time" not in load_gas_pipeline(str(p)).feature_names


def test_slog_transform_tames_extremes():
    from modeo.data import apply_transform
    X = np.array([[2.3e38, 0.5], [-1.5e38, 0.6]], dtype=np.float64)
    Z = apply_transform(X, "slog")
    assert np.all(np.abs(Z) < 100) and Z[0, 0] > 0 > Z[1, 0] and Z[1, 1] > Z[0, 1]


def test_open_set_split_never_leaks_unknown(tmp_path):
    p = tmp_path / "gas.arff"
    _fake_arff(p, n=2000)
    pr = prepare("gas_pipeline", str(p), seed=0, unknown="DoS", use_f3=True, pseudo_unknown="next")
    assert pr.unknown == 6 and pr.pseudo_unknown == 7
    assert 6 not in set(pr.y_fit) and 6 not in set(pr.y_cal) and 6 in set(pr.y_test)
    sd, known = search_data(pr, 1.0, 0)
    assert 6 not in known and 7 not in known and sd.n_classes == 6
    assert len(sd.X_va_unknown) > 0 and sd.y_tr.max() == 5
    v = open_set_view(range(8), [6])
    assert list(v.remap(np.array([0, 6, 7]))) == [0, -1, 6]


def test_auto_pseudo_unknown_picks_hardest_class():
    """Class 3 overlaps the normal class, class 1 and 2 are far away -> 'auto' must pick 3."""
    from modeo.data import Dataset
    from modeo.experiment import hardest_pseudo_unknown
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (600, 4)), rng.normal(8, 1, (200, 4)), rng.normal(-8, 1, (200, 4)),
              rng.normal(0.3, 1, (200, 4))].astype(np.float32)
    y = np.r_[np.zeros(600), np.ones(200), np.full(200, 2), np.full(200, 3)].astype(int)
    ds = Dataset(X, y, [f"f{i}" for i in range(4)], ["Normal", "A", "B", "C"], "t", normal_class=0)
    assert hardest_pseudo_unknown(X, y, ds, unknown=None) == 3
    assert hardest_pseudo_unknown(X, y, ds, unknown=3) in (1, 2)   # never the test-unknown class
