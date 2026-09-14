#!/bin/sh
# The real-data leverage sweep widened from three tables to nine, by lowering the entry
# threshold from an sd shift of 2 to 1.5. The six added tables have lighter tails, so they
# show whether the effect follows the leverage or the tail index. About 4 hours, resumable.
#
#     sh run/prevalence_models_wide.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}

SETS="freMTPL2sev,Brazilian_houses,OnlineNewsPopularity,nyc-taxi-green-dec-2016,\
Job_Profitability,stock_fardamento02,Buzzinsocialmedia_Twitter,\
Airlines_DepDelay_10M,particulate-matter-ukair-2017"

MODELS=GBM,XGB,TabICLv2,TabPFN-V3 DATASETS="$SETS" \
  N_EST=1 N_PER_BIN=5 OUTPUT=prevalence_models_wide.csv \
  $PY -u experiments/h2_leverage/prevalence_models.py
