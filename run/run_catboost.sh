#!/bin/sh
# Add CatBoost as the tree baseline that actually matters.
#
# WHY
# ---
# The contrast the thesis draws is "foundation models against trees", and it
# currently runs against sklearn gradient boosting, which is pedagogical, and
# XGBoost. On TabArena the best tree model in regression is CatBoost, at 1484
# Elo, and that is the number the abstract compares against. Measuring the
# control that the comparison names is cheap and removes an obvious question.
#
# CatBoost is not installed by default:
#     pip install catboost
#
# COST
# ----
# About 2 hours on CPU across the three places a control belongs.
#
#     sh run/run_catboost.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}   # another environment: PY=... sh run/...
M=CB

# H1: does it track the conditional shape any better than the others?
MODELS=$M N_EST=1 SEEDS=20 OUTPUT=shape_of_x_catboost.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py
MODELS=$M N_EST=1 SEEDS=20 OUTPUT=scale_of_x_catboost.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py

# H2: it does not standardise the target, so it should not react at all
MODELS=$M XI=0.3,0.7,0.9 DOSES=1,5,10,100 SEEDS=5 N_EST=1 \
  OUTPUT=influence_catboost.csv \
  $PY -u experiments/h2_leverage/influence.py
