#!/bin/sh
# The response to one row, dosed on the sd shift instead of a multiple of the maximum.
#
# A dose of 100 * max(y_train) is not one treatment: at xi = 0.7 it shifts the sd by 50 to
# 97 across seeds, and the range that public data actually contains (2/99 tables above 4,
# the worst 19.9) is barely measured. The targets are read off the prevalence survey: 2 and
# 4 are its thresholds, 10 its 1-in-99 case, 20 the worst table, 50 the old regime.
# `dose_mode` is in the key, so these rows cannot mix with the old ones. About 2 hours.
#
#     MODELS=TabICLv2,GBM sh run/influence_sd.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}

MODELS=${MODELS:-TabICLv2,TabPFN-V3,GBM} XI=0.3,0.7 \
  DOSE_MODE=sd DOSES=2,4,10,20,50 SEEDS=5 N_EST=1 OUTPUT=influence_sd.csv \
  $PY -u experiments/h2_leverage/influence.py
