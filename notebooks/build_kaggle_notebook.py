"""Build the self-contained Kaggle notebook (code zip embedded as base64)."""
import base64, io, json, os, sys, zipfile

SRC = "/home/claude/sc_modeo"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/SC_MODEO_CNN_kaggle.ipynb"

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in ("runs", "__pycache__", ".pytest_cache")]
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, os.path.dirname(SRC)))
b64 = base64.b64encode(buf.getvalue()).decode()

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# SC-MODEO-CNN on Kaggle
Reimplementation of MODEO-CNN (Lu et al., IEEE TDSC 2025) + Security-Complete extensions.

**Before running:** Settings → Accelerator **GPU** (T4 or P100), Internet **On**, and *Add Input* → your Gas Pipeline dataset
(and/or `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`).

**How to use**
1. Run cells 1–4 interactively (≈10 min): they install, test, check the dataset and time a pilot run.
2. Set `PLAN` in cell 1 to the stages you want, then **Save Version → Save & Run All (Commit)**. That runs in the background (up to 12 h) and keeps `/kaggle/working`.
3. To continue after the 12 h limit: *Add Input* → this notebook's previous version output, re-commit. Finished runs are skipped and unfinished searches resume from their last generation.
""")

code(r"""
# ===== 1. CONFIG — edit here =====
DATASET    = "gas_pipeline"      # "gas_pipeline" | "edge_iiotset" | "synthetic" (no data needed)
DATA_PATH  = None                # None = auto-detect under /kaggle/input
DATA_URL   = "http://www.ece.uah.edu/~thm0009/icsdatasets/gas_final.arff"  # Gas Pipeline 2014 (needs Internet On); None to use DATA_PATH / inputs
MAX_ROWS   = None                # Edge-IIoTset: e.g. 200_000 (stratified cap); None for Gas Pipeline
UNKNOWN    = "DoS"               # zero-day class for main / ablation / baseline runs
SEEDS      = [0, 1, 2, 3, 4]
LOAO_SEEDS = [0, 1, 2]           # LOAO over every attack class is the most expensive stage
BUDGET     = dict(generations=40, search_epochs=5, final_epochs=60, search_fraction=0.3)
BASE_BATCH = 256                 # paper: 256 Gas Pipeline, 2048 BoT-IoT
# stages run in this order; start with ["check", "pilot"], then commit with the full list
PLAN = ["check", "pilot"]
# PLAN = ["check", "main", "ablations", "baselines", "explain", "summary"]
# PLAN = ["check", "rescore", "loao", "summary"]   # rescore = add AE + fusion scores to finished runs
TIME_LIMIT_H = 11.3              # stop starting new runs after this many hours (Kaggle limit is 12 h)
""")

code(r"""
# ===== 2. SETUP: unpack code, install captum, run unit tests =====
import base64, io, os, sys, zipfile, subprocess, time, glob, json, shutil
T0 = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
CODE_B64 = "__CODE_B64__"
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall(WORK)
REPO = os.path.join(WORK, "sc_modeo")
os.chdir(REPO)
try:
    import captum  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "captum"], check=False)
import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (enable GPU in Settings!)")
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)

# optional: download the dataset (Kaggle working dir is wiped between sessions, so this reruns each time)
if DATA_URL:
    dest = os.path.join(WORK, os.path.basename(DATA_URL))
    if not os.path.exists(dest) or os.path.getsize(dest) < 10_000:
        subprocess.run(["wget", "-q", "-O", dest, DATA_URL], check=False)
    head = open(dest, "rb").read(200).lower() if os.path.exists(dest) else b""
    if b"<html" in head or not head:
        raise RuntimeError(f"download of {DATA_URL} failed or returned a web page — check Internet is On")
    print("downloaded", dest, os.path.getsize(dest) // 1024, "KB")
    DATA_PATH = dest

# resume: copy runs/ from an attached previous version of this notebook, if any
RUNS = os.path.join(REPO, "runs")
os.makedirs(RUNS, exist_ok=True)
for prev in glob.glob("/kaggle/input/**/sc_modeo/runs", recursive=True):
    print("restoring previous runs from", prev)
    shutil.copytree(prev, RUNS, dirs_exist_ok=True)
""")

code(r"""
# ===== 3. HELPERS =====
def find_data():
    if DATASET == "synthetic" or DATA_PATH:
        return DATA_PATH
    pats = {"gas_pipeline": ["*.arff", "*gas*.csv"], "edge_iiotset": ["DNN-EdgeIIoT-dataset.csv"]}[DATASET]
    hits = []
    for p in pats:
        hits += glob.glob(f"/kaggle/input/**/{p}", recursive=True)
    hits = [h for h in hits if "/sc_modeo/" not in h]
    if not hits:
        raise FileNotFoundError(f"no {pats} under /kaggle/input — add your dataset as an input")
    print("data file:", hits[0], "" if len(hits) == 1 else f"(also found: {hits[1:]})")
    return hits[0]

DATA = find_data()
COMMON = ["--dataset", DATASET] + (["--data-path", DATA] if DATA else []) + \
         (["--max-rows", str(MAX_ROWS)] if MAX_ROWS else [])
BUD = ["--generations", str(BUDGET["generations"]), "--search-epochs", str(BUDGET["search_epochs"]),
       "--final-epochs", str(BUDGET["final_epochs"]), "--search-fraction", str(BUDGET["search_fraction"]),
       "--base-batch", str(BASE_BATCH)]

def hours():
    return (time.time() - T0) / 3600

def run(script, args, log=None):
    cmd = [sys.executable, "-u", f"scripts/{script}"] + args
    print("\n$", " ".join(cmd[2:]), flush=True)
    t = time.time()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        f = open(log, "a") if log else None
        for line in p.stdout:
            print(line, end="", flush=True)
            if f: f.write(line)
        if f: f.close()
    print(f"[{script} finished in {(time.time() - t) / 60:.1f} min, exit {p.returncode}, session {hours():.2f} h]")
    return p.returncode == 0

def tag(mode, unknown, seed, extra=""):
    # SC runs use the 'auto' (hardest) pseudo-unknown rule; '_Pauto' keeps them apart from
    # older runs that used the 'next' rule, so attached old outputs are not mistaken for them
    p = "_Pauto" if mode == "sc" and extra != "_nof3" else ""
    return f"{DATASET}_{mode}_U-{unknown}{p}{extra}_s{seed}"

def search(mode, unknown, seed, extra_args=(), extra_tag=""):
    out = os.path.join("runs", tag(mode, unknown, seed, extra_tag))
    if os.path.exists(os.path.join(out, "results.json")):
        print("skip (done):", out); return True
    if hours() > TIME_LIMIT_H:
        print("time limit reached, not starting", out); return False
    args = COMMON + BUD + ["--mode", mode, "--unknown-class", unknown, "--seed", str(seed), "--out", out] + list(extra_args)
    if os.path.exists(os.path.join(out, "search_state.json")):
        args.append("--resume")
    os.makedirs(out, exist_ok=True)
    return run("run_search.py", args, log=os.path.join(out, "log.txt"))
""")

code(r"""
# ===== 4. CHECK the dataset (verify classes / features before spending GPU time) =====
if "check" in PLAN:
    sys.path.insert(0, REPO)
    from modeo.data import load_dataset
    import numpy as np, pandas as pd
    kw = {"max_rows": MAX_ROWS} if MAX_ROWS else {}
    ds = load_dataset(DATASET, DATA, **kw)
    print(f"{len(ds.y)} rows, {ds.X.shape[1]} features after dropping constant/leaky columns")
    print("features:", ds.feature_names)
    print(pd.Series(np.bincount(ds.y), index=ds.class_names, name="count").to_string())
    if DATASET == "gas_pipeline":
        assert len(ds.y) == 97019 and ds.X.shape[1] == 18, "unexpected Gas Pipeline file (expect 97,019 rows, 18 features)"
        print("OK: matches the 2014 Gas Pipeline release used in the paper")
    assert UNKNOWN in ds.class_names or DATASET == "synthetic", f"UNKNOWN='{UNKNOWN}' not in {ds.class_names}"
""")

code(r"""
# ===== 5. PILOT: tiny budget, to time one search on this GPU =====
if "pilot" in PLAN:
    t = time.time()
    ok = run("run_search.py", COMMON + ["--mode", "sc", "--unknown-class", UNKNOWN, "--seed", "0",
             "--generations", "3", "--search-epochs", str(BUDGET["search_epochs"]), "--final-epochs", "5",
             "--search-fraction", str(BUDGET["search_fraction"]), "--base-batch", str(BASE_BATCH),
             "--out", "runs/_pilot"])
    per_gen = (time.time() - t) / 3
    est = per_gen * BUDGET["generations"] / 3600
    fmt = lambda h: f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"
    print(f"\n≈{per_gen:.0f} s per generation (rough: includes final training) → one full search ≈ {fmt(est)}")
    print(f"main stage ({2 * len(SEEDS)} searches) ≈ {fmt(2 * len(SEEDS) * est)}; "
          f"ablations ({3 * len(SEEDS)}) ≈ {fmt(3 * len(SEEDS) * est)}; "
          f"LOAO ({2 * 7 * len(LOAO_SEEDS)} searches) ≈ {fmt(14 * len(LOAO_SEEDS) * est)}  (Kaggle session limit: 12 h)")
    shutil.rmtree("runs/_pilot", ignore_errors=True)
""")

code(r"""
# ===== 5b. RESCORE finished runs with the autoencoder + fusion zero-day scores (no retraining) =====
if "rescore" in PLAN:
    run("rescore.py", ["--dataset", DATASET] + (["--data-path", DATA] if DATA else []) + ["--runs-root", "runs"])
""")

code(r"""
# ===== 6. MAIN: MODEO-CNN vs SC-MODEO-CNN, same zero-day class, all seeds =====
if "main" in PLAN:
    for s in SEEDS:
        for mode in ["baseline", "sc"]:
            search(mode, UNKNOWN, s)
""")

code(r"""
# ===== 7. ABLATIONS (SC components), all seeds =====
if "ablations" in PLAN:
    for s in SEEDS:
        search("sc", UNKNOWN, s, ["--no-loss-gene"], "_noloss")
        search("sc", UNKNOWN, s, ["--no-f3"], "_nof3")  # no f3 -> no pseudo-unknown
        search("sc", UNKNOWN, s, ["--f3-score", "msp"], "_f3msp")
""")

code(r"""
# ===== 8. HAND-DESIGNED BASELINES (CNN, LSTM, autoencoder) =====
if "baselines" in PLAN:
    for s in SEEDS:
        out = f"runs/{DATASET}_baselines_U-{UNKNOWN}_s{s}"
        if os.path.exists(os.path.join(out, "baselines.json")) or hours() > TIME_LIMIT_H:
            continue
        run("run_baselines.py", COMMON + ["--unknown-class", UNKNOWN, "--seed", str(s),
            "--epochs", str(BUDGET["final_epochs"]), "--base-batch", str(BASE_BATCH), "--out", out])
""")

code(r"""
# ===== 9. LEAVE-ONE-ATTACK-OUT: every attack class as the zero-day class =====
if "loao" in PLAN:
    sys.path.insert(0, REPO)
    from modeo.data import load_dataset
    ds0 = load_dataset(DATASET, DATA, **({"max_rows": MAX_ROWS} if MAX_ROWS else {}))
    attacks = [c for i, c in enumerate(ds0.class_names) if i != ds0.normal_class]
    print("LOAO classes:", attacks)
    for s in LOAO_SEEDS:
        for cls in attacks:
            for mode in ["baseline", "sc"]:
                search(mode, cls, s)
""")

code(r"""
# ===== 10. EXPLAINABILITY on the SC models of every seed =====
if "explain" in PLAN:
    dirs = sorted(glob.glob(f"runs/{DATASET}_sc_U-{UNKNOWN}_Pauto_s*"))
    dirs = [d for d in dirs if glob.glob(os.path.join(d, "model_*.pt"))]
    if dirs:
        run("run_explain.py", ["--dataset", DATASET] + (["--data-path", DATA] if DATA else []) +
            ["--runs"] + dirs + ["--n-samples", "500", "--out", f"runs/explain_{DATASET}_U-{UNKNOWN}"])
""")

code(r"""
# ===== 11. SUMMARY tables + Wilcoxon + hypervolume; zip everything for download =====
if "summary" in PLAN:
    run("analyze.py", ["--runs-root", "runs", "--out", "runs/summary"])
    from IPython.display import Markdown, display
    if os.path.exists("runs/summary/table.md"):
        display(Markdown(open("runs/summary/table.md").read()))
shutil.make_archive(os.path.join(WORK, "sc_modeo_runs"), "zip", REPO, "runs")
print("download:", os.path.join(WORK, "sc_modeo_runs.zip"), f"| session time {hours():.2f} h")
""")

for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
txt = json.dumps(nb, indent=1).replace("__CODE_B64__", b64)
open(OUT, "w").write(txt)
print("wrote", OUT, len(txt) // 1024, "KB")
