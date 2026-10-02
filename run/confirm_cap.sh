#!/bin/sh
# Confirmation of the frozen cap clip_200, 1.10.2026: the unit error on new subsamples with
# a random corrupted row beside the visible one, and the legitimate high-risk segment on the
# generator. Each about two hours on CPU; separate outputs, so they can run side by side
# (rule 1). The predictions are in the docstrings and in PREREGISTRATION.md.
#
#     sh run/confirm_cap.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}

MODELS=TabPFN-V3,TabICLv2,GBM REPEATS=20 N_FIT=2000 N_TEST=1000 N_EST=1 CLIP_C=200 \
  OUTPUT=unit_error_confirm.csv \
  $PY -u experiments/h3_repair/unit_error_confirm.py

MODELS=TabPFN-V3,TabICLv2,GBM XI=0.7 SD_SHIFTS=1,20 CLIP_C=200 SEEDS=20 N_TRAIN=2000 \
  N_TEST=5000 N_EST=1 OUTPUT=legit_segment.csv \
  $PY -u experiments/h3_repair/legit_segment.py
