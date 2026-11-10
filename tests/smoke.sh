#!/bin/bash
# End-to-end smoke test on synthetic data (~2 min on CPU). Run from the repo root.
set -e
PY=${PY:-python}
B="--generations 2 --search-epochs 1 --final-epochs 1 --search-fraction 0.4 --c-high 4 --base-batch 64"
${PY:-python} scripts/run_search.py --dataset synthetic --mode baseline --unknown-class DoS $B --out runs/_smoke/base
${PY:-python} scripts/run_search.py --dataset synthetic --mode sc --unknown-class DoS $B --out runs/_smoke/sc
${PY:-python} scripts/run_baselines.py --dataset synthetic --unknown-class DoS --epochs 1 --out runs/_smoke/baselines
${PY:-python} scripts/run_explain.py --dataset synthetic --runs runs/_smoke/base runs/_smoke/sc --n-samples 50 --n-alerts 3 --out runs/_smoke/explain
${PY:-python} scripts/analyze.py --runs-root runs/_smoke --out runs/_smoke/summary > /dev/null
echo "SMOKE TEST PASSED"
