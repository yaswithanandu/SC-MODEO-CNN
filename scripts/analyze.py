#!/usr/bin/env python3
"""Aggregate runs: mean ± std tables, paired Wilcoxon tests vs. the baseline, hypervolume.

  python scripts/analyze.py --runs-root runs --out runs/summary
Writes summary.csv (one row per run x solution), table.md and tests.json.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modeo.pareto import hypervolume, non_dominated_sort  # noqa: E402


def variant_name(cfg: dict) -> str:
    v = "MODEO-CNN" if cfg["mode"] == "baseline" else "SC-MODEO-CNN"
    if cfg.get("no_loss_gene"):
        v += " -loss"
    if cfg.get("no_f3"):
        v += " -f3"
    if cfg.get("f3_score") and cfg["mode"] == "sc" and cfg["f3_score"] != "mahalanobis":
        v += f" (f3={cfg['f3_score']})"
    if cfg["mode"] == "sc" and not cfg.get("no_f3"):
        rule = cfg.get("pseudo_unknown") or "next"   # runs made before 'auto' existed used 'next'
        if rule != "auto":
            v += f" [P={rule}]"
    return v


def load_runs(root: str) -> pd.DataFrame:
    rows = []
    for arch_path in glob.glob(os.path.join(root, "**", "archive.json"), recursive=True):
        d = os.path.dirname(arch_path)
        arch = json.load(open(arch_path))
        cfg = arch["config"]
        F = np.array([g["fitness"] for g in arch["archive"]], dtype=float)
        F = F[F[:, 1] < 1e9] if np.any(F[:, 1] < 1e9) else F[:0]  # drop infeasible placeholders
        base = {"run": d, "dataset": cfg["dataset"], "variant": variant_name(cfg),
                "unknown": cfg.get("unknown_class") or "-", "seed": cfg["seed"],
                "budget": f"G{cfg['generations']}-e{cfg['search_epochs']}-E{cfg['final_epochs']}-"
                          f"f{cfg['search_fraction']}-C{cfg['c_low']}..{cfg['c_high']}-b{cfg['base_batch']}",
                "search_hours": arch["search_seconds"] / 3600, "archive_size": len(F)}
        # 2-objective HV (error, MFLOPs) with a fixed reference so variants are comparable;
        # MFLOPs are scaled by the dataset-wide max across runs later, here we store raw fronts.
        base["_front2"] = F[:, :2].tolist()
        base["_front3"] = F.tolist() if F.shape[1] == 3 else None
        res_path = os.path.join(d, "results.json")
        if not os.path.exists(res_path):
            rows.append(dict(base, solution="-"))
            continue
        res = json.load(open(res_path))
        for sol, r in res["solutions"].items():
            row = dict(base, solution=sol, mflops=r["mflops"], loss=r["train"]["loss"])
            row.update({f"cs_{k}": v for k, v in r["closed_set"].items()})
            for m, o in r.get("open_set", {}).items():
                for k in ["auroc", "fpr_at_95tpr", "unknown_detection_rate", "known_false_alarm_rate",
                          "known_macro_f1_with_rejection", "open_set_macro_f1"]:
                    if k not in o:
                        continue
                    row[f"os_{m}_{k}"] = o[k]
            rows.append(row)
    return pd.DataFrame(rows)


def add_hypervolume(df: pd.DataFrame) -> pd.DataFrame:
    hv = []
    for ds, sub in df.groupby("dataset"):
        mmax = max([max(f[1] for f in fr) for fr in sub["_front2"] if fr] + [1e-9])
        for i, fr in sub["_front2"].items():
            if not fr:
                hv.append((i, 0.0)); continue
            F = np.array(fr, dtype=float)
            F[:, 1] /= mmax  # MFLOPs scaled to [0,1] per dataset
            F = F[non_dominated_sort(F)[0]]
            hv.append((i, hypervolume(F, [1.0, 1.1])))
    for i, v in hv:
        df.loc[i, "hv_err_mflops"] = v
    return df


def fmt(x):
    return f"{np.mean(x):.4f} ± {np.std(x):.4f}" if len(x) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="runs/summary")
    ap.add_argument("--solution", default="best_performance")
    ap.add_argument("--reference", default="MODEO-CNN")
    a = ap.parse_args()
    df = load_runs(a.runs_root)
    if df.empty:
        sys.exit("no runs found")
    df = add_hypervolume(df)
    os.makedirs(a.out, exist_ok=True)
    df.drop(columns=["_front2", "_front3"]).to_csv(os.path.join(a.out, "summary.csv"), index=False)

    sel = df[df["solution"] == a.solution] if (df["solution"] == a.solution).any() else df
    metrics = [c for c in ["cs_f1_macro", "cs_f1_weighted", "cs_accuracy", "mflops", "os_msp_auroc",
                           "os_energy_auroc", "os_mahalanobis_auroc", "os_mahalanobis_fpr_at_95tpr",
                           "os_mahalanobis_open_set_macro_f1", "os_ae_auroc", "os_fusion_auroc",
                           "os_fusion_fpr_at_95tpr", "os_fusion_open_set_macro_f1",
                           "os_fusion_pc_unknown_detection_rate", "os_fusion_pc_open_set_macro_f1", "hv_err_mflops", "search_hours"] if c in sel]
    lines = ["| dataset | variant | n runs | " + " | ".join(metrics) + " |",
             "|" + "---|" * (3 + len(metrics))]
    for (ds, bud, var), g in sel.groupby(["dataset", "budget", "variant"]):
        lines.append(f"| {ds} [{bud}] | {var} | {len(g)} | " +
                     " | ".join(fmt(g[m].dropna().values) for m in metrics) + " |")
    open(os.path.join(a.out, "table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # paired Wilcoxon signed-rank tests vs. the reference variant (pairs = same dataset/unknown/seed)
    tests = defaultdict(dict)
    key = ["dataset", "budget", "unknown", "seed"]  # only runs with identical budgets are paired
    ref = sel[sel["variant"] == a.reference].set_index(key)
    for var, g in sel[sel["variant"] != a.reference].groupby("variant"):
        g = g.set_index(key)
        common = g.index.intersection(ref.index)
        for m in metrics:
            if m not in g or m not in ref:
                continue
            x, y = g.loc[common, m].astype(float), ref.loc[common, m].astype(float)
            ok = x.notna() & y.notna()
            if ok.sum() < 5 or np.allclose(x[ok], y[ok]):
                tests[var][m] = {"n": int(ok.sum()), "p": None, "note": "need >= 5 non-identical pairs"}
                continue
            st = wilcoxon(x[ok], y[ok])
            tests[var][m] = {"n": int(ok.sum()), "statistic": float(st.statistic), "p": float(st.pvalue),
                             "median_diff": float(np.median(x[ok] - y[ok]))}
    json.dump(tests, open(os.path.join(a.out, "tests.json"), "w"), indent=2)
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
