"""Pareto utilities: non-dominated sorting, crowding distance (Eq. 8),
external-archive update (Algorithm 6) and hypervolume. All objectives are minimised."""
from __future__ import annotations

import random
from typing import List, Sequence

import numpy as np


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    """Fast non-dominated sorting (Deb et al., 2002). Returns fronts as index lists."""
    n = len(F)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Eq. (8): sum over objectives of |f_j^{d+1} - f_j^{d-1}| after sorting by f_j;
    boundary points get +inf. (Un-normalised, as written in the paper.)"""
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        d[order[0]] = d[order[-1]] = np.inf
        for k in range(1, n - 1):
            d[order[k]] += abs(F[order[k + 1], j] - F[order[k - 1], j])
    return d


def update_archive(archive: list, sn, a_max: int, rng: random.Random,
                   remove: str = "highest") -> list:
    """Algorithm 6. `archive` and `sn` are Genome objects with `.fitness` set.

    The paper says to remove "a candidate with the highest crowding distance" when the
    front exceeds A_max (remove='highest', faithful default). Standard NSGA-II practice
    removes the *lowest* (most crowded) one; remove='lowest' is offered for an ablation.
    """
    keys = {g.key() for g in archive}
    merged = archive + ([sn] if sn.key() not in keys else [])
    F = np.array([g.fitness for g in merged])
    front = [merged[i] for i in non_dominated_sort(F)[0]]
    while len(front) > a_max:
        cd = crowding_distance(np.array([g.fitness for g in front]))
        target = cd.max() if remove == "highest" else cd.min()
        cands = [i for i, v in enumerate(cd) if v == target]
        front.pop(rng.choice(cands))
    return front


def hypervolume(F: np.ndarray, ref: Sequence[float]) -> float:
    """Exact hypervolume (minimisation) by recursive slicing; fine for small fronts."""
    ref = np.asarray(ref, dtype=float)
    P = np.asarray([p for p in np.asarray(F, dtype=float) if np.all(p < ref)])
    if len(P) == 0:
        return 0.0
    if P.shape[1] == 1:
        return float(ref[0] - P[:, 0].min())
    P = P[np.argsort(P[:, -1])]
    hv = 0.0
    for i in range(len(P)):
        upper = P[i + 1, -1] if i + 1 < len(P) else ref[-1]
        depth = upper - P[i, -1]
        if depth > 0:
            hv += depth * hypervolume(P[: i + 1, :-1], ref[:-1])
    return float(hv)
