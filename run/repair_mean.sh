#!/bin/sh
# Winsorising for the mean-only models. TabDPT runs from its own venv, about 1 hour. TabFM
# needs its 6.6 GB of weights (run/download_tabfm.sh) and is left commented out.
#
#     sh run/repair_mean.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

MODEL=TabDPT XI=0.5,0.7,0.9 SEEDS=5 OUTPUT=repair_mean_tabdpt.csv \
  venv-tabdpt/Scripts/python.exe -u experiments/h3_repair/repair_mean.py

# MODEL=TabFM XI=0.5,0.9 SEEDS=3 OUTPUT=repair_mean_tabfm.csv \
#   venv-tabfm/Scripts/python.exe -u experiments/h3_repair/repair_mean.py
