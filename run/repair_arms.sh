#!/bin/sh
# The four repair sweeps of part three: the context arms on the generator, on real leverage
# subsamples and under a unit error, and the conformal output patch. About 30 hours on CPU,
# most of it tail_splice; each command is resumable and writes its own OUTPUT (rule 1).
#
# N_EST=1 so no repair is confounded with the ensemble's own loss of tail. CLIP_C is a grid
# in the context scripts and one cap in tail_splice.py, so it is set per command. The
# U_SHARES grid is part of tail_splice's arm names, so another grid cannot resume these rows.
#
#     sh run/repair_arms.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=venv-tfm/Scripts/python.exe

MODEL=TabPFN-V3 XI=0.7,0.9 SD_SHIFTS=1,4,20,50 CLIP_C=20,50,200 SEEDS=20 \
  N_EST=1 N_TRAIN=2000 N_TEST=900 OUTPUT=clip_context.csv \
  $PY -u experiments/h3_repair/clip_context.py

MODELS=TabPFN-V3,TabICLv2,GBM CLIP_C=20,50,200 N_EST=1 OUTPUT=clip_context_real.csv \
  $PY -u experiments/h3_repair/clip_context_real.py

MODELS=TabPFN-V3,TabICLv2,GBM REPEATS=10 N_FIT=2000 N_TEST=1000 N_EST=1 \
  OUTPUT=unit_error_real.csv \
  $PY -u experiments/h3_repair/unit_error_real.py

MODELS=TabPFN-V3,TabICLv2,GBM REPEATS=10 N_FIT=1500 N_CALIB=500 N_TEST=1000 \
  N_EST=1 CLIP_C=200 U_SHARES=0.8,0.85,0.9,0.95 OUTPUT=tail_splice.csv \
  $PY -u experiments/h3_repair/tail_splice.py
