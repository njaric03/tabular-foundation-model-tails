#!/bin/sh
# Raise the one free recommendation from three seeds to ten.
#
# WHY
# ---
# "Use one ensemble member and gain about 9% of tail accuracy at no cost to the
# median" is the only recommendation in the thesis that a reader can apply the
# same day. It currently rests on three seeds, one model and one tail index.
#
# findings/NALAZI.md section 6 records three separate occasions where a median
# over three seeds gave the wrong headline number, twice over a distribution
# that turned out to be bimodal. Quoting a recommendation on that basis, in a
# document whose own methodological point is that few seeds mislead, is not
# defensible. Ten seeds, three tail indices, both models.
#
# COST
# ----
# About 1 hour on CPU.
#
#     sh run/run_vincentization_10.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}   # another environment: PY=... sh run/...

MODELS=TabICLv2,TabPFN-V3 SEEDS=10 XI=0.3,0.7,0.9 \
  OUTPUT=vincentization_10seeds.csv \
  $PY -u experiments/h1_shape_vs_scale/vincentization.py
