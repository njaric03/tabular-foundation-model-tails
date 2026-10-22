#!/bin/sh
# The real-data slope of part one at Hill shares 0.05, 0.10 and 0.20 (22.10.2026). One fit
# per (dataset, model, seed) serves all three; k = 0.10 repeats the stored file. Two
# streams on 6 threads each, one output each (rule 1). Predictions K1 and K2 are in
# PREREGISTRATION.md.
#
#     sh run/dissociation_kshare.sh tabicl
#     sh run/dissociation_kshare.sh tabpfn
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}

case "$1" in
tabicl) M=TabICLv2 ;;
tabpfn) M=TabPFN-V3,GBM ;;
*) echo "usage: sh run/dissociation_kshare.sh tabicl|tabpfn" >&2; exit 2 ;;
esac
MODELS=$M SEEDS=3 N_EST=4 N_BOOT=100 K_SHARES=0.05,0.1,0.2 \
  OUTPUT=dissociation_real_slope_kshare_$1.csv \
  $PY -u experiments/h1_shape_vs_scale/dissociation_real_slope.py
