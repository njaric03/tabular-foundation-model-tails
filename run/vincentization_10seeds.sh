#!/bin/sh
# The single-member recommendation on ten seeds, three tail indices and both models,
# instead of three seeds; a median over few seeds has given a wrong headline three times.
# About 1 hour on CPU.
#
#     sh run/vincentization_10seeds.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}

MODELS=TabICLv2,TabPFN-V3 SEEDS=10 XI=0.3,0.7,0.9 OUTPUT=vincentization_10seeds.csv \
  $PY -u experiments/h1_shape_vs_scale/vincentization.py
