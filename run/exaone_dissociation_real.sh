#!/bin/sh
# EXAONE on the real-data dissociation, at n_est=1 like the published run of the other
# models. EXAONE attends across features, so its cost grows with the width; the tables are
# ordered from 6 to 130 features and the narrow ones finish first. A first attempt at
# n_est=4 wrote nothing in 2h44m.
set -x
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY=venv-tabfm/Scripts/python.exe
BY_WIDTH="CPS1988,218_house_8L,houses,diamonds,particulate-matter-ukair-2017,house_16H,\
OnlineNewsPopularity,Buzzinsocialmedia_Twitter,superconduct,Allstate_Claims_Severity"

MODELS=EXAONE N_EST=1 SEEDS=3 DATASETS="$BY_WIDTH" OUTPUT=dissociation_real_exaone.csv \
  $PY -u experiments/h1_shape_vs_scale/dissociation_real.py
