#!/bin/sh
# Empirical coverage of the predicted quantiles on ten real tables, clean and under one
# leverage row. At 6000 test rows 0.99 expects 60 exceedances and 0.999 six, so the count
# is recorded with every coverage. About 5 hours on CPU, resumable. Add "sint" to DATASETS
# for the generator, where the true coverage is known.
#
#     MODELS=TabICLv2,GBM sh run/coverage.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}

SETS="freMTPL2sev,OnlineNewsPopularity,diamonds,particulate-matter-ukair-2017,\
Buzzinsocialmedia_Twitter,CPS1988,218_house_8L,superconduct,houses,house_16H"

MODELS=${MODELS:-GBM,XGB,TabICLv2,TabPFN-V3} DATASETS="$SETS" \
  DOSES=0,100 SEEDS=3 N_EST=1 OUTPUT=coverage.csv \
  $PY -u experiments/h1_shape_vs_scale/empirical_coverage.py
