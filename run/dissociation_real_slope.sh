#!/bin/sh
# The real-data dissociation as a slope across every usable feature, corrected for noise in
# the reference. It replaces the steepest-feature quotient, whose feature is mostly
# selection noise (reference_instrument.csv). About 90 minutes on CPU; the bootstrap is over
# data rows, not model fits.
#
#     MODELS=TabICLv2,GBM sh run/dissociation_real_slope.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}

MODELS=${MODELS:-TabICLv2,TabPFN-V3,GBM} SEEDS=3 N_EST=4 N_BOOT=100 \
  OUTPUT=dissociation_real_slope.csv \
  $PY -u experiments/h1_shape_vs_scale/dissociation_real_slope.py
