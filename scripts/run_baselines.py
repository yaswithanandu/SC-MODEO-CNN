#!/usr/bin/env python3
"""Hand-designed baselines under the same split: fixed CNN, LSTM, and (for zero-day) an
autoencoder anomaly detector. Optionally re-evaluates a searched genome with a different
loss (e.g. MODEO-CNN genome + CE vs. + focal) for the loss-gene ablation.

  python scripts/run_baselines.py --dataset gas_pipeline --data-path data/gas_final.arff \
      --unknown-class DoS --epochs 60
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modeo.baselines import autoencoder_score, handcrafted_cnn_genome, train_autoencoder, train_lstm  # noqa: E402
from modeo.data import open_set_view  # noqa: E402
from modeo.experiment import dump, evaluate_trained, final_evaluate, prepare  # noqa: E402
from modeo.ood import open_set_report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-path")
    ap.add_argument("--unknown-class")
    ap.add_argument("--norm")
    ap.add_argument("--transform", choices=["none", "slog"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--base-batch", type=int, default=256)
    ap.add_argument("--max-rows", type=int, help="must match the value used for run_search.py")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    load_kw = {"max_rows": a.max_rows} if a.max_rows else {}
    p = prepare(a.dataset, a.data_path, a.seed, a.norm, a.unknown_class, transform=a.transform, **load_kw)
    view = open_set_view(range(len(p.ds.class_names)), [p.unknown] if p.unknown is not None else [])
    y_fit = view.remap(p.y_fit)
    out = {}

    print("[baseline] hand-designed CNN")
    out["handcrafted_cnn"] = final_evaluate(handcrafted_cnn_genome(), p, a.epochs, a.base_batch, a.seed)

    print("[baseline] LSTM")
    lstm = train_lstm(p.X_fit, y_fit, view.n_known, epochs=a.epochs, batch=a.base_batch, seed=a.seed)
    out["lstm"] = evaluate_trained(lstm, p, a.seed)

    if p.unknown is not None:
        print("[baseline] autoencoder (trained on known classes only)")
        ae = train_autoencoder(p.X_fit, epochs=a.epochs, batch=a.base_batch, seed=a.seed)
        y_te = view.remap(p.y_test)
        kn = y_te >= 0
        s_cal, s_te = autoencoder_score(ae, p.X_cal), autoencoder_score(ae, p.X_test)
        dummy = np.zeros(kn.sum(), dtype=int)  # AE has no classifier: closed-set fields are not meaningful
        r = open_set_report(s_cal, s_te[kn], s_te[~kn], y_te[kn], dummy, np.zeros((~kn).sum(), int),
                            view.n_known)
        out["autoencoder"] = {k: r[k] for k in ["auroc", "fpr_at_95tpr", "unknown_detection_rate",
                                                 "known_false_alarm_rate", "threshold"]}

    for k, v in out.items():
        line = f"[baseline] {k}: "
        if "closed_set" in v:
            line += f"F1m={v['closed_set']['f1_macro']:.4f} F1w={v['closed_set']['f1_weighted']:.4f} "
        if "open_set" in v:
            line += "AUROC " + " ".join(f"{m}={r['auroc']:.3f}" for m, r in v["open_set"].items())
        if k == "autoencoder":
            line += f"AUROC={v['auroc']:.3f}"
        print(line)
    tag = f"{a.dataset}_baselines" + (f"_U-{a.unknown_class}" if a.unknown_class else "") + f"_s{a.seed}"
    dump(out, os.path.join(a.out or os.path.join("runs", tag), "baselines.json"))


if __name__ == "__main__":
    main()
