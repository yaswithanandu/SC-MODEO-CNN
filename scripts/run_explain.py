#!/usr/bin/env python3
"""Explain trained Pareto models (output of run_search.py).

For each model_*.pt in one or more run directories:
  * IG and GradientSHAP attributions on a stratified test sample
  * global + per-class feature importance (top-k printed for the domain sanity check)
  * deletion / insertion curves vs. a random-order reference
  * for zero-day runs: explanations of flagged unknown samples (nearest known class + top features)
Across all models given (different seeds and/or Pareto members): Kendall-tau stability.

  python scripts/run_explain.py --dataset gas_pipeline --data-path data/gas_final.arff \
      --runs runs/gas_pipeline_sc_U-DoS_s0 runs/gas_pipeline_sc_U-DoS_s1 --n-samples 500
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modeo.data import open_set_view  # noqa: E402
from modeo.experiment import dump, load_trained, prepare  # noqa: E402
from modeo.explain import (attributions, deletion_insertion, explain_zero_day_alerts,  # noqa: E402
                           global_importance, ranking_stability)
from modeo.ood import Mahalanobis  # noqa: E402
from modeo.train import predict  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-path")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--norm")
    ap.add_argument("--transform", choices=["none", "slog"])
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--n-alerts", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="runs/explain")
    a = ap.parse_args(argv)

    models = sorted(sum([glob.glob(os.path.join(r, "model_*.pt")) for r in a.runs], []))
    if not models:
        sys.exit("no model_*.pt found in the given run directories")
    report, overall_imp = {"models": {}}, {"ig": [], "gradshap": []}
    cache = {}
    for path in models:
        model, ck = load_trained(path)
        seed, norm = ck.get("seed", 0), a.norm or ck.get("norm")
        transform = a.transform or ck.get("transform")
        load_kw = ck.get("load_kw", {})
        key = (ck.get("unknown"), seed, norm, transform, tuple(sorted(load_kw.items())))
        if key not in cache:  # rebuild the exact split used for training
            cache[key] = prepare(a.dataset, a.data_path, seed, norm, ck.get("unknown"),
                                 transform=transform, **load_kw)
        p = cache[key]
        view = open_set_view(range(len(p.ds.class_names)), [p.unknown] if p.unknown is not None else [])
        y_te = view.remap(p.y_test)
        kn = np.where(y_te >= 0)[0]
        rng = np.random.default_rng(0)
        sel = rng.choice(kn, size=min(a.n_samples, len(kn)), replace=False)
        X, y = p.X_test[sel], y_te[sel]
        logits, _ = predict(model, X)
        pred = logits.argmax(1)
        baseline = p.X_fit.mean(0).astype(np.float32)
        known_names = [p.ds.class_names[c] for c in view.known_classes]
        entry = {"genome": ck["genome"]}
        for method in ["ig", "gradshap"]:
            attr = attributions(model, X, pred, method=method, baseline=baseline)
            imp = global_importance(attr, pred, view.n_known)
            overall_imp[method].append(imp["overall"])
            di = deletion_insertion(model, X, pred, attr, baseline)
            dr = deletion_insertion(model, X, pred, attr, baseline, random_ranking=True)
            top = {("overall" if k == "overall" else known_names[int(k)]):
                   [p.ds.feature_names[j] for j in np.argsort(-v)[: a.top_k]] for k, v in imp.items()}
            entry[method] = {"top_features": top, "importance": {k: v for k, v in imp.items()},
                             "deletion_auc": di["deletion_auc"], "insertion_auc": di["insertion_auc"],
                             "random_deletion_auc": dr["deletion_auc"],
                             "random_insertion_auc": dr["insertion_auc"],
                             "curves": {"fraction": di["fraction"], "deletion": di["deletion"],
                                        "insertion": di["insertion"], "random_deletion": dr["deletion"],
                                        "random_insertion": dr["insertion"]}}
            print(f"[{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}] {method}: "
                  f"deletion AUC={di['deletion_auc']:.3f} (random {dr['deletion_auc']:.3f})  "
                  f"insertion AUC={di['insertion_auc']:.3f} (random {dr['insertion_auc']:.3f})")
            for cname, feats in top.items():
                print(f"    {cname:>10}: {', '.join(feats)}")
        if p.unknown is not None:
            unk = np.where(y_te < 0)[0]
            idx = np.random.default_rng(0).permutation(len(p.X_fit))[:20000]
            _, Zf = predict(model, p.X_fit[idx])
            maha = Mahalanobis().fit(Zf, view.remap(p.y_fit)[idx], view.n_known)
            _, Zc = predict(model, p.X_cal)
            thr = np.percentile(maha.score(Zc), 95)
            _, Zu = predict(model, p.X_test[unk])
            flagged = unk[maha.score(Zu) > thr][: a.n_alerts]
            if len(flagged):
                entry["zero_day_alerts"] = explain_zero_day_alerts(
                    model, p.X_test[flagged], maha, p.ds.feature_names, known_names, baseline, a.top_k)
                near = [r["nearest_known_class"] for r in entry["zero_day_alerts"]]
                print(f"    zero-day ({p.ds.class_names[p.unknown]}) alerts: {len(flagged)} explained; "
                      f"nearest known classes: { {str(k): int(v) for k, v in zip(*np.unique(near, return_counts=True))} }")
        report["models"][path] = entry
    report["stability"] = {m: ranking_stability(v, top_k=a.top_k) for m, v in overall_imp.items() if len(v) > 1}
    for m, s in report["stability"].items():
        print(f"[stability] {m}: Kendall tau = {s['kendall_tau_mean']:.3f} ± {s['kendall_tau_std']:.3f} "
              f"over {s['n_pairs']} pairs; top-{a.top_k} overlap = {s[f'top{a.top_k}_overlap_mean']:.2f}")
    dump(report, os.path.join(a.out, "explain.json"))
    print(f"saved {os.path.join(a.out, 'explain.json')}")


if __name__ == "__main__":
    main()
