"""Paper figures from runs/ (vector PDF + PNG)."""
import glob, json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, OUT = "/home/claude/results/runs", "/home/claude/paper/images"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.5, "savefig.dpi": 300, "lines.linewidth": 2})
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); plt.close(fig)
VAR = {"baseline": ("MODEO-CNN", BLUE, "gas_pipeline_baseline_U-DoS_s{}"), "sc": ("SC-MODEO-CNN", ORANGE, "gas_pipeline_sc_U-DoS_Pauto_s{}")}
hist = {k: [json.load(open(f"{R}/{pat.format(s)}/search_state.json"))["history"] for s in range(5)] for k, (_, _, pat) in VAR.items()}

# Fig 1: search dynamics
fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
for k, (name, col, _) in VAR.items():
    feas = np.array([[np.mean(np.array(g["pop_fitness"])[:, 1] < 1e9) for g in h] for h in hist[k]])
    best = np.array([[min(f[0] for f in g["archive_fitness"]) for g in h] for h in hist[k]])
    x = np.arange(1, feas.shape[1] + 1)
    for a, y in [(ax[0], feas * 100), (ax[1], best)]:
        a.plot(x, y.mean(0), color=col, label=name); a.fill_between(x, y.min(0), y.max(0), color=col, alpha=0.15, lw=0)
ax[0].set(xlabel="Generation", ylabel="Feasible candidates (%)", title="(a) Share of useful candidates per generation", ylim=(0, 105))
ax[1].set(xlabel="Generation", ylabel="Best archive error  (1 − F1)", title="(b) Best search error", yscale="log")
ax[1].text(0.98, 0.95, "MODEO-CNN: weighted F1\nSC-MODEO-CNN: macro F1", transform=ax[1].transAxes, ha="right", va="top", color=INK2, fontsize=6.5)
ax[0].legend(loc="lower right"); fig.tight_layout(); save(fig, "fig_search_dynamics")

# Fig 2: final archives projected on (MFLOPs, error); filled = non-dominated on these two objectives
import sys; sys.path.insert(0, "/home/claude/sc_modeo")
from modeo.pareto import non_dominated_sort
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
for a, (k, (name, col, pat)) in zip(ax, VAR.items()):
    allA = []
    for s in range(5):
        A = np.array([g["fitness"] for g in json.load(open(f"{R}/{pat.format(s)}/archive.json"))["archive"]])
        allA.append(A[A[:, 1] < 1e9])
    A = np.vstack(allA)
    nd = np.zeros(len(A), bool); nd[non_dominated_sort(A[:, :2])[0]] = True
    a.scatter(A[~nd, 1], A[~nd, 0], s=14, facecolors="none", edgecolors=col, linewidths=0.8, alpha=0.7,
              label="archive member (all seeds)")
    o = np.argsort(A[nd, 1])
    a.plot(A[nd, 1][o], A[nd, 0][o], color=col, lw=1, alpha=0.6)
    a.scatter(A[nd, 1], A[nd, 0], s=20, color=col, edgecolors="white", linewidths=0.6, zorder=3,
              label="non-dominated in (MFLOPs, error)")
    a.set(xscale="log", yscale="log", xlabel="MFLOPs", ylabel="Search error (1 − F1)",
          title=f"({'ab'[list(VAR).index(k)]}) {name}: {len(A)} archive members, 5 seeds")
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
from matplotlib.lines import Line2D
hs = [Line2D([], [], ls="none", marker="o", mfc="none", mec=INK2, ms=4.5, label="archive member (5 seeds pooled)"),
      Line2D([], [], ls="-", lw=1, color=INK2, marker="o", mfc=INK2, mec="white", ms=5, label="non-dominated in (MFLOPs, error)")]
fig.legend(handles=hs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, "fig_pareto_archives")

# Fig 3: zero-day scores
df = pd.read_csv("/home/claude/results/summary_local/summary.csv"); bp = df[df.solution == "best_performance"]
scores = [("msp", "MSP"), ("energy", "Energy"), ("mahalanobis", "Mahal."), ("ae", "AE"), ("fusion", "Fusion")]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.7))
w = 0.38; x = np.arange(len(scores))
for j, (v, (name, col, _)) in enumerate([("MODEO-CNN", VAR["baseline"]), ("SC-MODEO-CNN", VAR["sc"])]):
    g = bp[bp.variant == v]
    for a, met in [(ax[0], "auroc"), (ax[1], "fpr_at_95tpr")]:
        m = [g[f"os_{s}_{met}"].mean() for s, _ in scores]; sd = [g[f"os_{s}_{met}"].std(ddof=1) for s, _ in scores]
        a.bar(x + (j - 0.5) * w, m, w * 0.92, yerr=sd, color=col, label=name, error_kw=dict(lw=0.8, ecolor=INK2, capsize=2))
for a in ax:
    a.set_xticks(x, [n for _, n in scores]); a.grid(axis="x", visible=False)
ax[0].axhline(0.5, color=GRAY, lw=1, ls="--", zorder=0); ax[0].text(4.62, 0.515, "chance", color=INK2, fontsize=6.5, ha="right")
ax[0].set(ylabel="AUROC (higher is better)", ylim=(0, 1.05), title="(a) Separating unseen DoS from known traffic")
ax[1].set(ylabel="FPR@95%TPR (lower is better)", ylim=(0, 1.05), title="(b) DoS accepted as known at 95% acceptance")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "fig_zero_day_scores")

# Fig 4: per-class F1, global vs per-class thresholds (fusion)
pc = pd.read_csv("/home/claude/paper/tables/per_class_f1_fusion.csv")
order = ["Normal", "NMRI", "CMRI", "MSCI", "MPCI", "MFCI", "Recon", "DoS (unknown)"]
fig, a = plt.subplots(figsize=(7, 2.4)); x = np.arange(len(order))
for j, (mode, col, lab) in enumerate([("global", BLUE, "global threshold"), ("per-class", AQUA, "per-class thresholds")]):
    g = pc[pc.threshold == mode].groupby("cls").f1
    a.bar(x + (j - 0.5) * w, g.mean().reindex(order), w * 0.92, yerr=g.std(ddof=1).reindex(order), color=col, label=lab,
          error_kw=dict(lw=0.8, ecolor=INK2, capsize=2))
a.set_xticks(x, [o.replace(" (unknown)", "\n(unseen)") for o in order]); a.grid(axis="x", visible=False)
a.set(ylabel="F1 after rejection", ylim=(0, 1.05))
a.text(5 - 0.5 * w, 0.03, "0", ha="center", color=INK2, fontsize=7)
a.set_title("Per-class F1 with the fused zero-day detector (10 models)", pad=18)
a.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=7)
fig.tight_layout(); save(fig, "fig_per_class_thresholds")

# Fig 5: feature importance (IG) across SC models
ex = json.load(open(f"{R}/explain_gas_pipeline_U-DoS/explain.json"))
import torch
names = torch.load(f"{R}/gas_pipeline_sc_U-DoS_Pauto_s0/model_best_performance.pt", map_location="cpu", weights_only=False)["feature_names"]
imps = np.array([np.array(e["ig"]["importance"]["overall"]) / np.sum(e["ig"]["importance"]["overall"]) for e in ex["models"].values()])
o = np.argsort(-imps.mean(0))[:10][::-1]
fig, a = plt.subplots(figsize=(3.4, 2.7))
a.barh(np.arange(10), imps.mean(0)[o] * 100, xerr=imps.std(0, ddof=1)[o] * 100, color=ORANGE, height=0.7,
       error_kw=dict(lw=0.8, ecolor=INK2, capsize=2))
a.set_yticks(np.arange(10), [names[i] for i in o]); a.grid(axis="y", visible=False)
a.set(xlabel="Share of total |IG attribution| (%)", title="Top-10 features, SC-MODEO-CNN (10 models)")
fig.tight_layout(); save(fig, "fig_feature_importance")

# Fig 6: deletion / insertion curves
fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
for a, curve, title in [(ax[0], "deletion", "(a) Deletion: remove top features first"), (ax[1], "insertion", "(b) Insertion: add top features first")]:
    fr = np.array(next(iter(ex["models"].values()))["ig"]["curves"]["fraction"]) * 100
    for key, col, lab, ls in [(curve, ORANGE, "Integrated Gradients order", "-"), ("random_" + curve, GRAY, "random order", "--")]:
        Y = np.array([e["ig"]["curves"][key] for e in ex["models"].values()])
        a.plot(fr, Y.mean(0), color=col, ls=ls, label=lab); a.fill_between(fr, Y.min(0), Y.max(0), color=col, alpha=0.12, lw=0)
    a.set(xlabel="Features changed (%)", ylabel="Predicted-class probability", ylim=(0, 1.02), title=title)
ax[0].legend(loc="upper right"); fig.tight_layout(); save(fig, "fig_faithfulness")
print(sorted(os.listdir(OUT)))
