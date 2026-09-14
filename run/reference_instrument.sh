#!/bin/sh
# The real-data shape reference measured with the model's own instrument, and the null
# distribution of the feature selection on permuted targets. No model is called; about 90
# minutes, most of it the permutations.
#
#     sh run/reference_instrument.sh
set -x
PY=${PY:-python}

SEEDS=3 N_PERM=20 N_BOOT=200 OUTPUT=reference_instrument.csv \
  $PY -u experiments/h1_shape_vs_scale/reference_instrument.py
