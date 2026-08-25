#!/bin/sh
# Stream C: oblik i skala pri n_est=1, 20 seedova.
#
# ZASTO POSEBNO. Udeo uhvacenog OBLIKA zavisi od velicine ansambla, i to jako:
# `findings/h1/dissociation.md` §18 daje TabPFN-V3 19% pri n_est=4 naspram 42% pri n_est=1.
# Glavna tabela u `findings/NALAZI.md` §3.1 (GBM 45, TabICLv2 44, TabPFN-V3 42, TabDPT 29) je
# n_est=1 tabela. EXAONE je do sada meren samo na n_est=4, pa u tu tabelu jos ne sme.
# Ovaj run daje red koji tamo sme.
set -x
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
PY=venv-tabfm/Scripts/python.exe

MODELS=EXAONE N_EST=1 SEEDS=20 OUTPUT=shape_of_x_exaone_nest1.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py

MODELS=EXAONE N_EST=1 SEEDS=20 OUTPUT=scale_of_x_exaone_nest1.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py
