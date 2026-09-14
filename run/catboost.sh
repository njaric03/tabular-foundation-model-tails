#!/bin/sh
# CatBoost, the strongest tree model in TabArena regression, as the control in each part
# that needs one. Needs `pip install catboost`. About 2 hours on CPU.
#
#     sh run/catboost.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}
M=CB

# H1: does it track the conditional shape better than the foundation models?
MODELS=$M N_EST=1 SEEDS=20 OUTPUT=shape_of_x_catboost.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py
MODELS=$M N_EST=1 SEEDS=20 OUTPUT=scale_of_x_catboost.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py

# H2: it does not standardise the target
MODELS=$M XI=0.3,0.7,0.9 DOSES=1,5,10,100 SEEDS=5 N_EST=1 OUTPUT=influence_catboost.csv \
  $PY -u experiments/h2_leverage/influence.py
