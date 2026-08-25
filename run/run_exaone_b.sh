#!/bin/sh
# Stream B: 20 seedova oblik -> 20 seedova skala -> sredina -> intervencija na xi=0,3 i 0,7.
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=venv-tabfm/Scripts/python.exe

MODELS=EXAONE N_EST=4 SEEDS=20 OUTPUT=shape_of_x_exaone_20.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py

MODELS=EXAONE N_EST=4 SEEDS=20 OUTPUT=scale_of_x_exaone_20.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py

XI=0.5,0.7,0.9 SEEDS=3 N_TRAIN=1000 N_TEST=200 N_EST=1 OUTPUT=mean_exaone.csv \
  $PY -u experiments/h2_leverage/mean_exaone.py

XI=0.3,0.7 DOSES=1,5,10,100 SEEDS=3 MODELS=EXAONE N_EST=4 OUTPUT=influence_exaone.csv \
  $PY -u experiments/h2_leverage/influence.py
