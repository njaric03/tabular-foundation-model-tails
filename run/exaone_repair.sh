#!/bin/sh
# EXAONE in part three: target transforms on the generator and winsorising on real tables.
# EXAONE lives in venv-tabfm. About 3 hours on CPU.
#
#     sh run/exaone_repair.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=venv-tabfm/Scripts/python.exe

MODELS=EXAONE XI=0.3,0.9 SEEDS=5 N_EST=1 OUTPUT=repair_exaone.csv \
  $PY -u experiments/h3_repair/repair.py

MODELS=EXAONE SEEDS=3 N_EST=1 OUTPUT=repair_real_exaone.csv \
  $PY -u experiments/h3_repair/repair_real.py
