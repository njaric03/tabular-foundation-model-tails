#!/bin/sh
# EXAONE's response to one row over five seeds, as for the other models. The xi = 0.9 block
# was measured before n_est was a column and carries the headline +671% at dose 100, so it
# is measured again with n_est recorded; cells already keyed with n_est=4 are skipped.
# Afterwards the old xi = 0.9 rows with an empty n_est go to the archive.
#
#     sh run/exaone_influence.sh
set -x
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
PY=venv-tabfm/Scripts/python.exe

MODELS=EXAONE XI=0.3,0.7,0.9 DOSES=1,5,10,100 SEEDS=5 N_EST=4 OUTPUT=influence_exaone.csv \
  $PY -u experiments/h2_leverage/influence.py
