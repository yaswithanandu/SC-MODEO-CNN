"""Hand-designed CNN + the same zero-day pipeline (Mahalanobis, AE, fusion, per-class thresholds)."""
import sys, json, time
sys.path.insert(0, "/home/claude/sc_modeo")
from modeo.baselines import handcrafted_cnn_genome
from modeo.experiment import prepare, final_evaluate, dump
seed = int(sys.argv[1])
t = time.time()
p = prepare("gas_pipeline", "/home/claude/data/gas_final.arff", seed, unknown="DoS")
r = final_evaluate(handcrafted_cnn_genome(), p, 60, 256, seed)
dump(r, f"/home/claude/diag/handcnn/seed{seed}.json")
o = r["open_set"]
print(f"seed {seed}: F1m={r['closed_set']['f1_macro']:.4f} MFLOPs={r['mflops']:.3f} | AUROC maha={o['mahalanobis']['auroc']:.3f} "
      f"ae={o['ae']['auroc']:.3f} fusion={o['fusion']['auroc']:.3f} | FPR95 maha={o['mahalanobis']['fpr_at_95tpr']:.3f} "
      f"fusion={o['fusion']['fpr_at_95tpr']:.3f} | openF1 fusion={o['fusion']['open_set_macro_f1']:.3f} "
      f"fusion_pc={o['fusion_pc']['open_set_macro_f1']:.3f} | {time.time()-t:.0f}s", flush=True)
