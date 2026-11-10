"""MODEO search loop (Algorithm 2) with Algorithm 5 fitness evaluation.

Objective vector (all minimised):
  F[0] = 1 - F1 on the validation set       (baseline: weighted F1, SC: macro F1)
  F[1] = MFLOPs of the decoded CNN
  F[2] = 1 - AUROC(known vs pseudo-unknown) (SC only, extension #13)
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score

from .encoding import Genome, mutate_population, random_genome
from .ood import Mahalanobis, auroc, compute_scores
from .pareto import non_dominated_sort, update_archive
from .train import get_device, predict, train_model


@dataclass
class SearchConfig:
    mode: str = "baseline"            # "baseline" (paper) or "sc" (SC-MODEO-CNN)
    generations: int = 200            # G   (Table II)
    a_max: int = 20                   # A_max
    c_low: int = 3                    # C_L
    c_high: int = 10                  # C_H
    base_batch: int = 256             # b (256 Gas Pipeline / Power System, 2048 BoT-IoT)
    search_epochs: int = 30           # epochs per fitness evaluation
    f1_average: str = "weighted"      # "weighted" (baseline) or "macro" (SC)
    use_loss_gene: bool = False       # SC: 5th setting gene
    use_f3: bool = False              # SC: zero-day objective
    f3_score: str = "mahalanobis"     # post-hoc score used inside f3
    feasibility_margin: float = 0.05  # a model is feasible only if its F1 beats the majority-class
                                      # predictor's F1 by this margin; infeasible models get the worst
                                      # MFLOPs / f3 so they never sit on the Pareto front (None = off)
    archive_remove: str = "highest"   # faithful to Alg. 6; "lowest" = NSGA-II style
    seed: int = 0
    out_dir: str = "runs/debug"
    maha_fit_max: int = 20000         # cap on training samples used to fit Mahalanobis

    @staticmethod
    def preset(mode: str, **overrides) -> "SearchConfig":
        if mode == "baseline":
            cfg = SearchConfig(mode="baseline", f1_average="weighted", use_loss_gene=False, use_f3=False)
        elif mode == "sc":
            cfg = SearchConfig(mode="sc", f1_average="macro", use_loss_gene=True, use_f3=True)
        else:
            raise ValueError(mode)
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


@dataclass
class SearchData:
    X_tr: np.ndarray
    y_tr: np.ndarray          # known classes remapped to 0..K-1
    X_va: np.ndarray
    y_va: np.ndarray
    n_classes: int
    X_va_unknown: Optional[np.ndarray] = None   # pseudo-unknown samples (f3)


INFEASIBLE_MFLOPS = 1e9


def majority_f1(y: np.ndarray, average: str) -> float:
    """F1 of the trivial 'always predict the most frequent class' model."""
    maj = np.bincount(y).argmax()
    return float(f1_score(y, np.full_like(y, maj), average=average, zero_division=0))


def evaluate_genome(g: Genome, data: SearchData, cfg: SearchConfig, device) -> List[float]:
    """Algorithm 5 (+ f3 in SC mode). Returns the objective vector and fills g.info.

    Feasibility rule (constrained domination, applied to both variants): a model whose F1
    does not beat the majority-class predictor by `feasibility_margin` gets MFLOPs = 1e9 and
    f3 = 1 in the objective vector, so any useful model dominates it and, among useless models,
    only the error counts. Without this, tiny models that learned nothing are Pareto-optimal on
    MFLOPs and the single-solution MODEO search can stay stuck on them (observed on Gas Pipeline).
    The real MFLOPs are kept in g.info."""
    model, tinfo = train_model(g, data.X_tr, data.y_tr, data.n_classes, cfg.search_epochs,
                               cfg.base_batch, seed=cfg.seed, device=device)
    mflops = model.mflops()
    g.info.update(tinfo)
    g.info["mflops"] = mflops
    if tinfo["diverged"]:
        g.info["f1"], g.info["feasible"] = 0.0, False
        return [1.0, INFEASIBLE_MFLOPS] + ([1.0] if cfg.use_f3 else [])
    logits, Z = predict(model, data.X_va, device=device)
    pred = logits.argmax(1)
    f1 = f1_score(data.y_va, pred, average=cfg.f1_average, zero_division=0)
    g.info["f1"] = float(f1)
    feasible = cfg.feasibility_margin is None or \
        f1 >= majority_f1(data.y_va, cfg.f1_average) + cfg.feasibility_margin
    g.info["feasible"] = bool(feasible)
    F = [1.0 - float(f1), mflops if feasible else INFEASIBLE_MFLOPS]
    if cfg.use_f3 and not feasible:
        g.info["auroc_pseudo_unknown"] = None
        F.append(1.0)
    elif cfg.use_f3:
        lu, zu = predict(model, data.X_va_unknown, device=device)
        maha = None
        if cfg.f3_score == "mahalanobis":
            idx = np.random.default_rng(cfg.seed).permutation(len(data.X_tr))[: cfg.maha_fit_max]
            _, Ztr = predict(model, data.X_tr[idx], device=device)
            maha = Mahalanobis().fit(Ztr, data.y_tr[idx], data.n_classes)
        a = auroc(compute_scores(cfg.f3_score, logits, Z, maha),
                  compute_scores(cfg.f3_score, lu, zu, maha))
        g.info["auroc_pseudo_unknown"] = a
        F.append(1.0 - a)
    return F


class MODEOSearch:
    def __init__(self, cfg: SearchConfig, data: SearchData):
        self.cfg, self.data = cfg, data
        self.rng = random.Random(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = get_device()
        self.cache: Dict[str, dict] = {}
        self.archive: List[Genome] = []
        self.history: List[dict] = []
        os.makedirs(cfg.out_dir, exist_ok=True)

    # -- persistence ---------------------------------------------------------
    @property
    def state_path(self) -> str:
        return os.path.join(self.cfg.out_dir, "search_state.json")

    def save(self, gen: int, sn: Genome):
        state = {"config": asdict(self.cfg), "generation": gen, "sn": sn.to_dict(),
                 "archive": [g.to_dict() for g in self.archive], "history": self.history,
                 "cache": self.cache, "rng": [self.rng.getstate()[0], list(self.rng.getstate()[1]), self.rng.getstate()[2]]}
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh)
        os.replace(tmp, self.state_path)

    def _restore(self):
        with open(self.state_path) as fh:
            st = json.load(fh)
        ulg = self.cfg.use_loss_gene

        def load(d):
            g = Genome.from_list(d["genome"], ulg)
            g.fitness, g.info = d["fitness"], d["info"]
            return g

        self.archive = [load(d) for d in st["archive"]]
        self.history, self.cache = st["history"], st["cache"]
        v, inner, gauss = st["rng"]
        self.rng.setstate((v, tuple(inner), gauss))
        return st["generation"] + 1, load(st["sn"])

    # -- main loop -----------------------------------------------------------
    def _fitness(self, g: Genome) -> List[float]:
        k = g.key()
        if k in self.cache:
            g.fitness, g.info = self.cache[k]["fitness"], dict(self.cache[k]["info"], cached=True,
                                                               mutation=g.info.get("mutation"))
            return g.fitness
        g.fitness = evaluate_genome(g, self.data, self.cfg, self.device)
        self.cache[k] = {"fitness": g.fitness, "info": dict(g.info)}
        return g.fitness

    def run(self, resume: bool = False, log=print) -> List[Genome]:
        cfg = self.cfg
        start = 0
        if resume and os.path.exists(self.state_path):
            start, sn = self._restore()
            log(f"resumed at generation {start}")
        else:
            sn = random_genome(cfg.c_low, cfg.c_high, self.rng, cfg.use_loss_gene)  # Alg. 2 line 1
        for t in range(start, cfg.generations):
            t0 = time.time()
            pop = mutate_population(sn, cfg.c_low, cfg.c_high, self.rng)       # line 5
            F = np.array([self._fitness(g) for g in pop])                        # line 6
            front = non_dominated_sort(F)[0]
            sn = pop[self.rng.choice(front)]                                     # line 7
            self.archive = update_archive(self.archive, sn, cfg.a_max, self.rng, cfg.archive_remove)  # line 8
            rec = {"generation": t, "seconds": time.time() - t0, "pop_size": len(pop),
                   "pop_fitness": F.tolist(), "selected": sn.to_dict(),
                   "archive_fitness": [g.fitness for g in self.archive]}
            self.history.append(rec)
            self.save(t, sn)
            best = min(self.archive, key=lambda g: g.fitness[0])
            n_feas = sum(bool(g.info.get("feasible", True)) for g in pop)
            log(f"gen {t + 1}/{cfg.generations}  pop={len(pop)} (feasible {n_feas})  |A|={len(self.archive)}  "
                f"best 1-F1={best.fitness[0]:.4f} ({best.info.get('mflops', best.fitness[1]):.3f} MFLOPs)  "
                f"{time.time() - t0:.1f}s")
        return self.archive


def pick_solutions(archive: List[Genome]) -> Dict[str, Genome]:
    """'best performance' = lowest error; 'best trade-off' = knee point (closest to the
    ideal point after min-max normalising each objective) — the paper does not define its
    trade-off rule, so this is our choice."""
    F = np.array([g.fitness for g in archive], dtype=float)
    lo, hi = F.min(0), F.max(0)
    Fn = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    best = min(range(len(archive)), key=lambda i: tuple(F[i]))  # lowest error, ties -> fewer MFLOPs
    return {"best_performance": archive[best],
            "best_tradeoff": archive[int(np.linalg.norm(Fn, axis=1).argmin())]}
