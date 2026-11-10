import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/sc_modeo")
from modeo.experiment import load_trained, prepare, fitted_autoencoder
from modeo.data import open_set_view
from modeo.ood import Mahalanobis, ScoreFusion, class_thresholds
from modeo.baselines import autoencoder_score
from modeo.train import predict
from sklearn.metrics import f1_score
rows = []
for run in sorted(glob.glob("runs/gas_pipeline_baseline_U-DoS_s*")) + sorted(glob.glob("runs/gas_pipeline_sc_U-DoS_Pauto_s*")):
    model, ck = load_trained(os.path.join(run, "model_best_performance.pt"))
    p = prepare("gas_pipeline", "/home/claude/data/gas_final.arff", ck["seed"], ck["norm"], ck["unknown"], transform=ck["transform"])
    view = open_set_view(range(8), [p.unknown]); K = view.n_known
    y_fit, y_te = view.remap(p.y_fit), view.remap(p.y_test); kn = y_te >= 0
    idx = np.random.default_rng(ck["seed"]).permutation(len(p.X_fit))[:20000]
    _, Zf = predict(model, p.X_fit[idx]); maha = Mahalanobis().fit(Zf, y_fit[idx], K)
    L_cal, Zc = predict(model, p.X_cal); L_te, Z_te = predict(model, p.X_test)
    ae = fitted_autoencoder(p, ck["seed"])
    c = [maha.score(Zc), autoencoder_score(ae, p.X_cal)]; t = [maha.score(Z_te), autoencoder_score(ae, p.X_test)]
    f = ScoreFusion().fit(c); s_cal, s_te = f.score(c), f.score(t)
    pt = L_te.argmax(1)
    y_true = np.where(kn, y_te, K)
    closed = f1_score(y_te[kn], pt[kn], labels=list(range(K)), average=None, zero_division=0)
    for mode in ["global", "per-class"]:
        thr = np.percentile(s_cal, 95) if mode == "global" else class_thresholds(s_cal, L_cal.argmax(1), K)[pt]
        y_pred = np.where(s_te > thr, K, pt)
        f1 = f1_score(y_true, y_pred, labels=list(range(K + 1)), average=None, zero_division=0)
        for i in range(K + 1):
            name = p.ds.class_names[view.known_classes[i]] if i < K else "DoS (unknown)"
            rows.append(dict(run=os.path.basename(run), threshold=mode, cls=name, f1=f1[i], closed_f1=closed[i] if i < K else np.nan))
pd.DataFrame(rows).to_csv("/home/claude/paper/tables/per_class_f1_fusion.csv", index=False)
d = pd.DataFrame(rows); print(d.pivot_table(index="cls", columns="threshold", values=["f1", "closed_f1"], aggfunc="mean").round(3))
