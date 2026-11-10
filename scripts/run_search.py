#!/usr/bin/env python3
"""Run one MODEO search (baseline MODEO-CNN or SC-MODEO-CNN), then fully train the
selected Pareto solutions and evaluate them on the held-out test set.

Examples
  # faithful baseline, paper budget (slow!)
  python scripts/run_search.py --dataset gas_pipeline --data-path data/gas_final.arff --mode baseline

  # SC-MODEO-CNN with DoS held out as the zero-day class, reduced budget
  python scripts/run_search.py --dataset gas_pipeline --data-path data/gas_final.arff --mode sc \
      --unknown-class DoS --generations 40 --search-epochs 5 --final-epochs 60 --search-fraction 0.3
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modeo.experiment import dump, final_evaluate, prepare, search_data  # noqa: E402
from modeo.search import MODEOSearch, SearchConfig, pick_solutions  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["gas_pipeline", "edge_iiotset", "bot_iot", "synthetic"])
    ap.add_argument("--data-path")
    ap.add_argument("--mode", choices=["baseline", "sc"], default="baseline")
    ap.add_argument("--unknown-class", help="test zero-day class (name or id); omit for closed-set runs")
    ap.add_argument("--pseudo-unknown", default="auto",
                    help="class hidden during search for f3: 'auto' (hardest attack class on training "
                         "data, default), 'next' (attack after U), or a class name/id")
    ap.add_argument("--norm", choices=["zscore", "minmax", "l2"])
    ap.add_argument("--transform", choices=["none", "slog"], help="default: slog for gas_pipeline, none otherwise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    # budget (defaults = paper Table II)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--search-epochs", type=int, default=30)
    ap.add_argument("--final-epochs", type=int, default=350)
    ap.add_argument("--search-fraction", type=float, default=1.0, help="stratified subsample used by the search")
    ap.add_argument("--a-max", type=int, default=20)
    ap.add_argument("--c-low", type=int, default=3)
    ap.add_argument("--c-high", type=int, default=10)
    ap.add_argument("--base-batch", type=int, default=256)
    # ablations / variants
    ap.add_argument("--no-loss-gene", action="store_true", help="SC ablation: drop the loss gene")
    ap.add_argument("--no-f3", action="store_true", help="SC ablation: drop the zero-day objective")
    ap.add_argument("--f1-average", choices=["weighted", "macro"])
    ap.add_argument("--f3-score", choices=["msp", "energy", "mahalanobis"], default="mahalanobis")
    ap.add_argument("--feasibility-margin", type=float, default=0.05,
                    help="F1 a model must gain over the majority-class predictor to be feasible; "
                         "negative = off (paper-faithful, can stall on Gas Pipeline)")
    ap.add_argument("--archive-remove", choices=["highest", "lowest"], default="highest")
    ap.add_argument("--final-which", choices=["picked", "all", "none"], default="picked",
                    help="fully train the 2 picked solutions, the whole archive, or nothing")
    ap.add_argument("--max-rows", type=int, help="BoT-IoT / Edge-IIoTset: stratified row cap when loading")
    ap.add_argument("--resume", action="store_true")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = SearchConfig.preset(
        a.mode, generations=a.generations, a_max=a.a_max, c_low=a.c_low, c_high=a.c_high,
        base_batch=a.base_batch, search_epochs=a.search_epochs, f3_score=a.f3_score,
        archive_remove=a.archive_remove, seed=a.seed, feasibility_margin=(None if a.feasibility_margin < 0 else a.feasibility_margin), f1_average=a.f1_average)
    if a.no_loss_gene:
        cfg.use_loss_gene = False
    if a.no_f3:
        cfg.use_f3 = False
    tag = f"{a.dataset}_{a.mode}" + (f"_U-{a.unknown_class}" if a.unknown_class else "") + \
          ("_noloss" if a.no_loss_gene else "") + ("_nof3" if a.no_f3 else "") + f"_s{a.seed}"
    cfg.out_dir = a.out or os.path.join("runs", tag)

    load_kw = {"max_rows": a.max_rows} if a.dataset in ("bot_iot", "edge_iiotset") and a.max_rows else {}
    p = prepare(a.dataset, a.data_path, a.seed, a.norm, a.unknown_class, a.pseudo_unknown,
                use_f3=cfg.use_f3, transform=a.transform, **load_kw)
    sd, known = search_data(p, a.search_fraction, a.seed)
    names = p.ds.class_names
    print(f"[data] {a.dataset}: {len(p.ds.y)} samples, {p.X_fit.shape[1]} features, classes={names}")
    print(f"[data] transform={p.transform} norm={p.norm}  test-unknown={names[p.unknown] if p.unknown is not None else None}  "
          f"pseudo-unknown={names[p.pseudo_unknown] if p.pseudo_unknown is not None else None}")
    print(f"[data] search train={len(sd.y_tr)} val={len(sd.y_va)} "
          f"val-unknown={0 if sd.X_va_unknown is None else len(sd.X_va_unknown)}  out={cfg.out_dir}")

    t0 = time.time()
    archive = MODEOSearch(cfg, sd).run(resume=a.resume)
    search_s = time.time() - t0
    dump({"config": vars(a), "search_seconds": search_s,
          "pseudo_unknown_resolved": names[p.pseudo_unknown] if p.pseudo_unknown is not None else None,
          "archive": [g.to_dict() for g in archive]}, os.path.join(cfg.out_dir, "archive.json"))

    if a.final_which == "none":
        return
    picked = pick_solutions(archive)
    todo = picked if a.final_which == "picked" else {f"archive_{i}": g for i, g in enumerate(archive)}
    results = {"search_seconds": search_s, "solutions": {}}
    done = {}
    for name, g in todo.items():
        if g.key() in done:  # same genome picked twice (e.g. small archive)
            results["solutions"][name] = dict(results["solutions"][done[g.key()]], same_as=done[g.key()])
            print(f"[final] {name} is the same genome as {done[g.key()]}; reusing its results")
            continue
        done[g.key()] = name
        print(f"[final] training {name}: {g.as_list()}  search fitness={g.fitness}")
        r = final_evaluate(g, p, a.final_epochs, cfg.base_batch, a.seed,
                           save_path=os.path.join(cfg.out_dir, f"model_{name}.pt"))
        results["solutions"][name] = r
        cs = r["closed_set"]
        msg = f"[final] {name}: acc={cs['accuracy']:.4f} F1w={cs['f1_weighted']:.4f} " \
              f"F1m={cs['f1_macro']:.4f} MFLOPs={r['mflops']:.3f}"
        if "open_set" in r:
            msg += "  AUROC " + " ".join(f"{m}={v['auroc']:.3f}" for m, v in r["open_set"].items())
        print(msg)
    dump(results, os.path.join(cfg.out_dir, "results.json"))


if __name__ == "__main__":
    main()
