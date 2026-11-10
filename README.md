# SC-MODEO-CNN

Course project (CNS, IIITDM Kancheepuram): reimplementation of **MODEO-CNN** (Lu et al., IEEE TDSC 22(4), 2025)
and the **Security-Complete** extensions — imbalance-aware search, a zero-day objective, a fused
Mahalanobis + autoencoder zero-day detector with per-class thresholds, and quantitatively checked explanations.

Authors: Yaswitha Nandu Yalamanchali, T. Harshini Sri Savitha, Pradeep Kumar Gajendra Bhale.

## Repository layout
| Path | Content |
|---|---|
| `modeo/`, `scripts/`, `tests/` | the method, experiment scripts and tests (details below) |
| `notebooks/SC_MODEO_CNN_kaggle.ipynb` | self-contained Kaggle notebook used for all GPU runs (code embedded); `build_kaggle_notebook.py` regenerates it |
| `paper/` | LaTeX paper: `SC_MODEO_CNN.tex`, `refs.bib`, `images/` |
| `results/` | per-run `results.json` / `archive.json` (Gas Pipeline, DoS as unseen attack, 5 seeds), summary tables (`summary/`, `tables/`), hand-designed CNN with the fused detector |
| `analysis/` | scripts that produced the paper's tables and figures from `results/` (paths inside point to our working directories; adjust before reuse) |

Data is not included: download `gas_final.arff` (Morris & Gao, 2014) and pass its path with `--data-path`,
or let the notebook download it (`DATA_URL`).

```
modeo/
  encoding.py   hybrid variable-length encoding (Alg. 3), n+3 mutation (Alg. 4), gene decoding (Eqs. 1-6)
  model.py      genome -> blocks-based 1-D CNN, analytical MFLOPs
  pareto.py     non-dominated sort, crowding distance (Eq. 8), archive update (Alg. 6), hypervolume
  search.py     MODEO loop (Alg. 2) + fitness (Alg. 5) + SC objectives
  train.py      training / inference (optimizers, L1/L2, losses)
  losses.py     CE, inverse-frequency CE, class-balanced CE, focal (gamma 1, 2)       [SC]
  ood.py        MSP, energy, Mahalanobis scores; AUROC, FPR@95TPR, open-set F1          [SC]
  explain.py    IG / GradientSHAP, deletion-insertion, Kendall tau, zero-day alert reports [SC]
  data.py       Gas Pipeline, Edge-IIoTset, BoT-IoT loaders, normalisation, open-set views
  experiment.py split protocol, final training + evaluation shared by scripts
  baselines.py  hand-designed CNN, LSTM, autoencoder
scripts/
  run_search.py    one search + final training/evaluation of the picked Pareto solutions
  run_loao.py      rotate the zero-day class through every attack
  run_baselines.py hand-designed baselines under the same split
  run_explain.py   explainability on saved models
  analyze.py       mean ± std tables, Wilcoxon tests, hypervolume
tests/            unit tests (pytest) + an end-to-end smoke test on synthetic data
```

## Setup
```bash
pip install -r requirements.txt          # torch >= 2.1, captum, scikit-learn, scipy, pandas
python -m pytest -q tests                # unit tests, ~10 s
bash tests/smoke.sh                      # end-to-end on synthetic data, ~2 min
```
A GPU is used automatically when available.

## Data
Put files under `data/` (not included):

| `--dataset` | File | Label used |
|---|---|---|
| `gas_pipeline` | Morris & Gao (2014) gas-pipeline ARFF/CSV (97,019 rows, 8 classes) | categorized result: 0 Normal, 1 NMRI, 2 CMRI, 3 MSCI, 4 MPCI, 5 MFCI, 6 DoS, 7 Recon |
| `edge_iiotset` | `DNN-EdgeIIoT-dataset.csv` (Ferrag et al., 2022) | `Attack_type` |
| `bot_iot` | BoT-IoT "10 best features" CSV | `category` (use `--max-rows` to subsample) |
| `synthetic` | generated | for smoke tests only |

The gas-pipeline loader auto-detects the label column (`categorized result` / `result` / `label`)
and drops `binary result` / `specific result`, which would leak the label. Check the printed
class list on the first run.

## What follows the paper and what we had to assume
| Item | Source |
|---|---|
| Encoding `[n, topo, conv_1..n, setting]`, gene ranges, decoding tables | Paper Eqs. (1)-(6) |
| Topology bit order (source-major: 1->2, 1->3, …, 2->3, …) | Paper Fig. 2 (checked against the drawing) |
| n+3 mutation operators | Paper Alg. 4 |
| MODEO loop, random pick from Pareto front of P_t | Paper Alg. 2 |
| Archive update; remove the **highest**-crowding member when over A_max | Paper Alg. 6 (faithful; this removes extreme points, so `--archive-remove lowest` is offered as a variant) |
| G=200, A_max=20, C_L=3, C_H=10, b=256 (2048 BoT-IoT), 30 / 350 epochs | Paper Table II |
| Objectives: 1−F1 and total MFLOPs | Paper Alg. 5 / Eq. 7 (F1 averaging not stated: **weighted** for the baseline) |
| Normalisation: Min-Max BoT-IoT, z-score Power System | Paper Sec. V-B |
| Gas Pipeline: **signed-log transform + z-score** instead of the paper's L2; `time` kept | **Deviation**, see below |
| Tabular row treated as a 1-channel 1-D signal (Conv1d) | **Assumption** |
| Multi-input blocks: channel concatenation after aligning lengths | **Assumption** (DenseNet-style, as the paper motivates) |
| Block with no selected predecessor reads the previous block | **Assumption** |
| 'same' conv padding; pooling stride 2 (length halves); head = GAP + Linear | **Assumption** |
| L1/L2 strength 1e-4; plain SGD (no momentum) | **Assumption** |
| **Feasibility rule** (both variants): a model must beat the majority-class predictor's F1 by 0.05, otherwise its MFLOPs/f3 are set to the worst value | **Deviation**: without it, tiny models that learned nothing are Pareto-optimal on MFLOPs and the single-solution search stalled on Gas Pipeline (1−F1 stuck at 0.864 for every candidate). `--feasibility-margin -1` disables it |
| "Best trade-off" solution = knee point (closest to ideal after min-max scaling) | **Assumption** (paper does not define it) |
| Split: 80/20 stratified train/test; 10% of train held out for threshold calibration; search on an 80/20 split of the rest | Ours (the paper uses 10-fold CV for Gas Pipeline) |

## Preprocessing deviation (Gas Pipeline)
In the 2014 `gas_final.arff`, the `measurement` column ranges from −1.5e38 to +2.3e38
(corrupted/injected readings), and 8 columns are constant. Measured on the real file, at the
search budget (30% subsample, 5 epochs):

| Preprocessing | Fixed CNN macro-F1 | Random genomes that learn (macro-F1 > 0.3) |
|---|---|---|
| L2 row normalisation (paper) | 0.47 | 4% (median 0.10) |
| z-score | 0.84 | – |
| signed log, then z-score (**ours**) | **0.95** | 21% (best 0.95) |

Signed log is `sign(x)·log(1+|x|)`: monotone, so extreme injected values stay extreme, but
±1e38 becomes ±88. Use `--transform none --norm l2` to reproduce the paper's choice.
`time` in this file is an inter-packet interval (Recon ≈ 1.005 vs ≈ 1.15 for other classes),
not a timestamp, so it is kept (the paper's 26 features include it); it is dropped only when
it is monotonic (a timestamp). Result: 18 features (26 − 8 constant).
With the paper's learning-rate range (≤ 2e-3), SGD and Adadelta almost never learn in a
5-epoch evaluation; the search has to discover Adam/RMSprop, which it does because mutations
keep the parent's `setting` except in SC_1 and SC_{n+3}.

## SC-MODEO-CNN extensions (`--mode sc`)
| # | Change | Where | Ablation flag |
|---|---|---|---|
| 1 | 5th `setting` gene: `ce`, `weighted_ce`, `cb_ce` (β=0.999), `focal1`, `focal2` | `encoding.py`, `losses.py` | `--no-loss-gene` |
| 2 | f1 = 1 − **macro**-F1 | `search.py` | `--f1-average weighted` |
| 3 | Mutation unchanged; SC_{n+3} resamples the loss gene | `encoding.py` | – |
| 4 | Post-hoc zero-day scores on the penultimate embedding: MSP, energy, Mahalanobis | `ood.py` | evaluated for every model |
| 5 | f3 = 1 − AUROC(known vs pseudo-unknown class P) | `search.py` | `--no-f3`, `--f3-score` |
| 6 | P ≠ test-unknown U; U never touches search or training | `experiment.py` | – |
| 7 | LOAO over all attacks | `run_loao.py` | – |
| 4b | Autoencoder reconstruction error (`ae`) and **fusion** = mean of calibration-z-scored log Mahalanobis and log AE error; reported for every model | `ood.py`, `experiment.py`; `scripts/rescore.py` adds them to finished runs | – |
| 9–13 | IG/GradientSHAP, deletion/insertion vs random, Kendall τ, per-class top features, zero-day alert reports | `explain.py`, `run_explain.py` | – |

By default P is the attack class after U (cyclic order). Open-set thresholds are set on
the calibration split at 95% known-sample acceptance, so no test information is used.

## Running experiments
Paper budget is ~0.5 GPU-day per Gas Pipeline run. For the course paper use a reduced budget,
**identical for every method**, and state it in the paper, e.g.:
```bash
B="--generations 40 --search-epochs 5 --final-epochs 60 --search-fraction 0.3"
D="--dataset gas_pipeline --data-path data/gas_final.arff"

# 1) baseline MODEO-CNN vs SC-MODEO-CNN, one zero-day class, 5 seeds
for s in 0 1 2 3 4; do
  python scripts/run_search.py $D --mode baseline --unknown-class DoS --seed $s $B
  python scripts/run_search.py $D --mode sc       --unknown-class DoS --seed $s $B
done

# 2) full LOAO for SC (and the same for --mode baseline)
python scripts/run_loao.py $D --mode sc --seeds 0 1 2 3 4 -- $B

# 3) ablations
python scripts/run_search.py $D --mode sc --unknown-class DoS --no-loss-gene $B
python scripts/run_search.py $D --mode sc --unknown-class DoS --no-f3 $B
python scripts/run_search.py $D --mode sc --unknown-class DoS --f3-score msp $B

# 4) hand-designed baselines (CNN, LSTM, autoencoder)
python scripts/run_baselines.py $D --unknown-class DoS --epochs 60

# 5) explainability on the saved models of several seeds
python scripts/run_explain.py $D --runs runs/gas_pipeline_sc_U-DoS_s*

# 6) tables + Wilcoxon + hypervolume
python scripts/analyze.py --runs-root runs --out runs/summary
```
Searches checkpoint after every generation; rerun the same command with `--resume` to continue.

Each run directory has `search_state.json` (full history, so you can plot convergence),
`archive.json` (final Pareto set), `results.json` (test metrics per picked solution) and
`model_*.pt`.
