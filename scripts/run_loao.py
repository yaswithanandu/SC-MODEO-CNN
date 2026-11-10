#!/usr/bin/env python3
"""Leave-one-attack-out protocol: rotate the test-unknown class through every attack
class and run run_search.py for each (all extra flags are forwarded).

  python scripts/run_loao.py --dataset gas_pipeline --data-path data/gas_final.arff --mode sc \
      --seeds 0 1 2 3 4 -- --generations 40 --search-epochs 5 --final-epochs 60 --search-fraction 0.3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modeo.data import load_dataset  # noqa: E402
import run_search  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-path")
    ap.add_argument("--mode", default="sc")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--classes", nargs="*", help="subset of attack classes to hold out (default: all)")
    a, rest = ap.parse_known_args()
    rest = [r for r in rest if r != "--"]
    ds = load_dataset(a.dataset, a.data_path)  # class names only (row cap does not change them)
    attacks = a.classes or [n for i, n in enumerate(ds.class_names) if i != ds.normal_class]
    for seed in a.seeds:
        for cls in attacks:
            print(f"\n===== LOAO: unknown={cls} seed={seed} mode={a.mode} =====")
            argv = ["--dataset", a.dataset, "--mode", a.mode, "--unknown-class", cls, "--seed", str(seed)]
            if a.data_path:
                argv += ["--data-path", a.data_path]
            run_search.main(argv + rest)


if __name__ == "__main__":
    main()
