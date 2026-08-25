#!/bin/sh
# Produce the CSV that the repair finding already cites.
#
# WHY
# ---
# findings/h3/repair.md quotes a mean-only repair result for TabDPT and TabFM
# (a truncated-mean ratio moving from 1.073 to 0.773) and cites a file that was
# never committed. Either the measurement is reproduced or the sentence comes
# out of the finding. This script reproduces it.
#
# TabDPT runs from its own venv. TabFM needs its 6.6 GB weights, so it is left
# commented out; fetch them with run/download_tabfm.sh first and about 10 GB
# free, then uncomment.
#
# COST
# ----
# About 1 hour on CPU for TabDPT alone.
#
#     sh run/run_repair_mean.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

MODEL=TabDPT XI=0.5,0.7,0.9 SEEDS=5 OUTPUT=repair_mean_tabdpt.csv \
  venv-tabdpt/Scripts/python.exe -u experiments/h3_repair/repair_mean.py

# MODEL=TabFM XI=0.5,0.9 SEEDS=3 OUTPUT=repair_mean_tabfm.csv \
#   venv-tabfm/Scripts/python.exe -u experiments/h3_repair/repair_mean.py
