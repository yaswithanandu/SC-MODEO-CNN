"""Build the paper's result tables from runs/ (CSV + LaTeX booktabs)."""
import glob, json, os, re, numpy as np, pandas as pd
R = "/home/claude/results"; OUT = "/home/claude/paper/tables"
df = pd.read_csv(f"{R}/summary_local/summary.csv")
bp = df[df.solution == "best_performance"].copy()
ORDER = ["MODEO-CNN", "SC-MODEO-CNN", "SC-MODEO-CNN -loss", "SC-MODEO-CNN -f3", "SC-MODEO-CNN (f3=msp)", "SC-MODEO-CNN [P=next]"]
NAMES = {"MODEO-CNN": "MODEO-CNN (reimpl.)", "SC-MODEO-CNN": "SC-MODEO-CNN (ours)", "SC-MODEO-CNN -loss": "\u2013 loss gene",
         "SC-MODEO-CNN -f3": "\u2013 $f_3$", "SC-MODEO-CNN (f3=msp)": "$f_3$ with MSP", "SC-MODEO-CNN [P=next]": "easy pseudo-unknown (Recon)"}
ms = lambda s: f"{s.mean():.3f} $\\pm$ {s.std(ddof=1):.3f}"
def latex(rows, cols, caption, label, path):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    tex = (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n"
           f"\\begin{{tabular}}{{l{'c' * (len(cols) - 1)}}}\n\\toprule\n" + " & ".join(cols) + r" \\" + "\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(path, "w").write(tex)

# hand-designed baselines
hb = []
for f in sorted(glob.glob(f"{R}/runs/gas_pipeline_baselines_U-DoS_s*/baselines.json")):
    b = json.load(open(f)); s = int(f.split("_s")[-1].split("/")[0])
    for k in ["handcrafted_cnn", "lstm"]:
        hb.append(dict(model=k, seed=s, f1m=b[k]["closed_set"]["f1_macro"], f1w=b[k]["closed_set"]["f1_weighted"],
                       maha_auroc=b[k]["open_set"]["mahalanobis"]["auroc"], maha_fpr=b[k]["open_set"]["mahalanobis"]["fpr_at_95tpr"]))
    hb.append(dict(model="autoencoder", seed=s, ae_auroc=b["autoencoder"]["auroc"], ae_fpr=b["autoencoder"]["fpr_at_95tpr"]))
hb = pd.DataFrame(hb); hb.to_csv(f"{OUT}/handcrafted_baselines.csv", index=False)

# T1 classification + complexity
rows, csv = [], []
for v in ORDER[:2]:
    g = bp[bp.variant == v]
    rows.append([NAMES[v], ms(g.cs_f1_macro), ms(g.cs_f1_weighted), ms(g.cs_accuracy), ms(g.mflops), ms(g.search_hours * 60)])
    csv.append(dict(model=v, n=len(g), f1_macro=g.cs_f1_macro.mean(), f1_macro_sd=g.cs_f1_macro.std(ddof=1), f1_weighted=g.cs_f1_weighted.mean(),
                    accuracy=g.cs_accuracy.mean(), mflops=g.mflops.mean(), mflops_sd=g.mflops.std(ddof=1), search_min=g.search_hours.mean() * 60))
for k, name in [("handcrafted_cnn", "Hand-designed CNN"), ("lstm", "LSTM")]:
    g = hb[hb.model == k]
    rows.append([name, ms(g.f1m), ms(g.f1w), "--", "--", "--"])
    csv.append(dict(model=name, n=len(g), f1_macro=g.f1m.mean(), f1_macro_sd=g.f1m.std(ddof=1), f1_weighted=g.f1w.mean()))
pd.DataFrame(csv).to_csv(f"{OUT}/t1_classification.csv", index=False)
latex(rows, ["Model", "Macro-F1", "Weighted-F1", "Accuracy", "MFLOPs", "Search (min)"],
      "Closed-set classification on the seven known Gas Pipeline classes (DoS held out), best-performance solution, mean $\\pm$ s.d. over 5 seeds.",
      "tab:classification", f"{OUT}/t1_classification.tex")

# T2 zero-day detection of DoS
rows, csv = [], []
SC = [("mahalanobis", "Mahalanobis"), ("ae", "Autoencoder"), ("fusion", "Fusion"), ("fusion_pc", "Fusion + per-class thr.")]
for v in ORDER[:2]:
    g = bp[bp.variant == v]
    for k, nm in SC:
        r = [NAMES[v] if k == "mahalanobis" else "", nm, ms(g[f"os_{k}_auroc"]), ms(g[f"os_{k}_fpr_at_95tpr"]),
             ms(g[f"os_{k}_unknown_detection_rate"]), ms(g[f"os_{k}_open_set_macro_f1"])]
        rows.append(r)
        csv.append(dict(model=v, score=nm, auroc=g[f"os_{k}_auroc"].mean(), fpr95=g[f"os_{k}_fpr_at_95tpr"].mean(),
                        detection=g[f"os_{k}_unknown_detection_rate"].mean(), open_f1=g[f"os_{k}_open_set_macro_f1"].mean(),
                        open_f1_sd=g[f"os_{k}_open_set_macro_f1"].std(ddof=1)))
pd.DataFrame(csv).to_csv(f"{OUT}/t2_zero_day.csv", index=False)
latex(rows, ["Model", "Score", "AUROC $\\uparrow$", "FPR@95 $\\downarrow$", "DoS detected $\\uparrow$", "Open-set macro-F1 $\\uparrow$"],
      "Zero-day detection of the unseen DoS attack. Thresholds accept 95\\% of known calibration traffic (globally, or per predicted class). Mean $\\pm$ s.d. over 5 seeds.",
      "tab:zeroday", f"{OUT}/t2_zero_day.tex")

# T3 ablations
rows, csv = [], []
for v in ORDER[1:]:
    g = bp[bp.variant == v]
    rows.append([NAMES[v], ms(g.cs_f1_macro), ms(g.mflops), ms(g.os_mahalanobis_auroc), ms(g.os_fusion_pc_open_set_macro_f1)])
    csv.append(dict(variant=v, f1_macro=g.cs_f1_macro.mean(), mflops=g.mflops.mean(), maha_auroc=g.os_mahalanobis_auroc.mean(),
                    fusion_pc_open_f1=g.os_fusion_pc_open_set_macro_f1.mean()))
pd.DataFrame(csv).to_csv(f"{OUT}/t3_ablations.csv", index=False)
latex(rows, ["Variant", "Macro-F1", "MFLOPs", "AUROC (Mahalanobis)", "Open-set F1 (fusion, per-class)"],
      "Ablation of SC-MODEO-CNN components (DoS held out, 5 seeds).", "tab:ablation", f"{OUT}/t3_ablations.tex")

# T4 loss / optimizer choices of the searched models
L = ["CE", "weighted CE", "class-balanced CE", "focal ($\\gamma$=1)", "focal ($\\gamma$=2)"]; O = ["SGD", "Adam", "Adadelta", "RMSprop"]
ch = []
for d in sorted(glob.glob(f"{R}/runs/gas_pipeline_*_U-DoS*_s*")):
    if "baselines" in d or not os.path.exists(f"{d}/results.json"): continue
    cfg = json.load(open(f"{d}/archive.json"))["config"]
    r = json.load(open(f"{d}/results.json"))["solutions"]["best_performance"]
    s = r["genome"][-1]
    ch.append(dict(run=os.path.basename(d), n_blocks=r["genome"][0], optimizer=O[s[1]], lr=(s[2] + 1) * 2.5e-4,
                   loss=L[s[4]] if len(s) > 4 else "CE (fixed)", train_loss_used=r["train"]["loss"]))
pd.DataFrame(ch).to_csv(f"{OUT}/t4_selected_settings.csv", index=False)

# T5 explainability (SC, auto pseudo-unknown)
ex = json.load(open(f"{R}/runs/explain_gas_pipeline_U-DoS/explain.json"))
xr = []
for path, e in ex["models"].items():
    for m in ["ig", "gradshap"]:
        xr.append(dict(model=path.split("/")[-2] + "/" + path.split("/")[-1], method=m, deletion=e[m]["deletion_auc"],
                       deletion_random=e[m]["random_deletion_auc"], insertion=e[m]["insertion_auc"], insertion_random=e[m]["random_insertion_auc"]))
xr = pd.DataFrame(xr); xr.to_csv(f"{OUT}/t5_faithfulness.csv", index=False)
rows = []
for m, nm in [("ig", "Integrated Gradients"), ("gradshap", "GradientSHAP")]:
    g = xr[xr.method == m]; st = ex["stability"][m]
    rows.append([nm, ms(g.deletion), ms(g.deletion_random), ms(g.insertion), ms(g.insertion_random),
                 f"{st['kendall_tau_mean']:.2f} $\\pm$ {st['kendall_tau_std']:.2f}", f"{st['top5_overlap_mean']:.2f}"])
latex(rows, ["Method", "Deletion AUC $\\downarrow$", "(random)", "Insertion AUC $\\uparrow$", "(random)", "Kendall $\\tau$", "Top-5 overlap"],
      "Faithfulness and stability of explanations for the 10 SC-MODEO-CNN models (5 seeds $\\times$ 2 Pareto picks).",
      "tab:xai", f"{OUT}/t5_xai.tex")

# Wilcoxon (SC vs MODEO-CNN)
json.dump(json.load(open(f"{R}/summary_local/tests.json")), open(f"{OUT}/wilcoxon_vs_modeo.json", "w"), indent=1)
print(pd.read_csv(f"{OUT}/t1_classification.csv").round(4).to_string()); print(pd.read_csv(f"{OUT}/t2_zero_day.csv").round(3).to_string())
print(pd.read_csv(f"{OUT}/t3_ablations.csv").round(3).to_string())
print(pd.DataFrame(ch).groupby(["run"]).first()[["optimizer", "loss", "n_blocks"]].to_string())
print(xr.groupby("method").mean(numeric_only=True).round(3))
