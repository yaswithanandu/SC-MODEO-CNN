#!/usr/bin/env python3
"""Re-evaluate already-trained models with the current open-set scores (adds the autoencoder
'ae' and the Mahalanobis+AE 'fusion' scores) without re-running any search or retraining CNNs.

  python scripts/rescore.py --dataset gas_pipeline --data-path data/gas_final.arff --runs-root runs
The original results.json is kept as results_pre_rescore.json.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modeo.experiment import dump, evaluate_trained, load_trained, prepare  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-path")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--force", action="store_true", help="rescore runs that already have 'fusion'")
    a = ap.parse_args(argv)
    cache, t0, n = {}, time.time(), 0
    for res_path in sorted(glob.glob(os.path.join(a.runs_root, "**", "results.json"), recursive=True)):
        d = os.path.dirname(res_path)
        res = json.load(open(res_path))
        sols = res.get("solutions", {})
        if not sols or not all("open_set" in r or "same_as" in r for r in sols.values()):
            continue  # closed-set run or not a search run
        if not a.force and all("fusion_pc" in r.get("open_set", {}) for r in sols.values() if "same_as" not in r):
            print("already rescored:", d)
            continue
        for name, r in sols.items():
            if "same_as" in r:
                continue
            model, ck = load_trained(os.path.join(d, f"model_{name}.pt"))
            key = (ck.get("unknown"), ck.get("seed", 0), ck.get("norm"), ck.get("transform"),
                   tuple(sorted(ck.get("load_kw", {}).items())))
            if key not in cache:
                cache[key] = prepare(a.dataset, a.data_path, ck.get("seed", 0), ck.get("norm"), ck.get("unknown"),
                                     transform=ck.get("transform"), **ck.get("load_kw", {}))
            new = evaluate_trained(model, cache[key], ck.get("seed", 0))
            r["closed_set"], r["open_set"] = new["closed_set"], new["open_set"]
        for name, r in sols.items():  # refresh duplicated entries
            if "same_as" in r:
                sols[name] = dict(sols[r["same_as"]], same_as=r["same_as"])
        if not os.path.exists(os.path.join(d, "results_pre_rescore.json")):
            shutil.copy(res_path, os.path.join(d, "results_pre_rescore.json"))
        dump(res, res_path)
        n += 1
        bp = sols.get("best_performance", next(iter(sols.values())))["open_set"]
        print(f"rescored {os.path.basename(d)}: AUROC maha={bp['mahalanobis']['auroc']:.3f} "
              f"ae={bp['ae']['auroc']:.3f} fusion={bp['fusion']['auroc']:.3f}  open-set F1 fusion "
              f"global={bp['fusion']['open_set_macro_f1']:.3f} per-class={bp['fusion_pc']['open_set_macro_f1']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"rescored {n} runs in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
