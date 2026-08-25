#!/bin/sh
# EXAONE is the only one of the five models with no third-part measurement.
#
# WHY
# ---
# EXAONE-Tabular is second on TabArena and one of the two models whose response
# to a contaminated row is bimodal, so it is exactly the model where the price
# of a repair is most interesting. It is measured for the first part and the
# second, and absent from the third.
#
# EXAONE needs venv-tabfm, which carries a torch version incompatible with the
# system Python.
#
# COST
# ----
# About 3 hours on CPU.
#
#     sh run/run_exaone_repair.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=venv-tabfm/Scripts/python.exe

MODELS=EXAONE XI=0.3,0.9 SEEDS=5 N_EST=1 OUTPUT=repair_exaone.csv \
  $PY -u experiments/h3_repair/repair.py

MODELS=EXAONE SEEDS=3 N_EST=1 OUTPUT=repair_real_exaone.csv \
  $PY -u experiments/h3_repair/repair_real.py
