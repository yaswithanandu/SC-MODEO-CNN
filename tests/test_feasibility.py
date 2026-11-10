"""Feasibility rule: useless models must never beat useful ones on MFLOPs."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sklearn.datasets import make_blobs
from modeo.data import normalise
from modeo.encoding import Genome
from modeo.pareto import non_dominated_sort
from modeo.search import INFEASIBLE_MFLOPS, SearchConfig, SearchData, evaluate_genome, majority_f1
from modeo.train import get_device


def test_majority_f1():
    y = np.array([0] * 80 + [1] * 20)
    assert np.isclose(majority_f1(y, "macro"), (2 * 0.8 / 1.8) / 2)


def test_useless_model_is_infeasible_and_dominated():
    X, y = make_blobs(n_samples=[1500, 500, 500, 500], n_features=18, cluster_std=1.0, random_state=1)
    perm = np.random.default_rng(0).permutation(len(y))  # imbalanced, separable, shuffled
    X, y = normalise(X[perm].astype(np.float32)), y[perm]
    d = SearchData(X[:2400], y[:2400], X[2400:], y[2400:], 4)
    cfg = SearchConfig.preset("baseline", search_epochs=3, base_batch=64)
    useless = Genome.from_list([3, "001", [0, 0, 6, 2, 0], [0, 0, 6, 2, 0], [0, 0, 6, 2, 0], [3, 0, 0, 0]])  # SGD, lr 2.5e-4
    useful = Genome.from_list([3, "001", [3, 0, 6, 2, 0], [3, 0, 6, 2, 0], [3, 0, 6, 2, 0], [0, 1, 3, 0]])   # Adam, lr 1e-3
    Fu, Fg = evaluate_genome(useless, d, cfg, get_device()), evaluate_genome(useful, d, cfg, get_device())
    assert useless.info["feasible"] is False and Fu[1] == INFEASIBLE_MFLOPS
    assert useful.info["feasible"] is True and Fg[1] < INFEASIBLE_MFLOPS
    assert useless.info["mflops"] < useful.info["mflops"]          # the useless one really is smaller...
    assert non_dominated_sort(np.array([Fu, Fg]))[0] == [1]          # ...but only the useful one is on the front
