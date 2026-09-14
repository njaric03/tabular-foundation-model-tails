#!/bin/sh
# EXAONE shape and scale at n_est=1, the setting of the main H1 table. The shape share
# depends strongly on the ensemble size (TabPFN-V3: 19% at 4 members, 42% at 1), so the
# n_est=4 rows cannot enter that table.
set -x
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
PY=venv-tabfm/Scripts/python.exe

MODELS=EXAONE N_EST=1 SEEDS=20 OUTPUT=shape_of_x_exaone_nest1.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py

MODELS=EXAONE N_EST=1 SEEDS=20 OUTPUT=scale_of_x_exaone_nest1.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py
