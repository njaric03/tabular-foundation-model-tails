#!/bin/sh
# The confirmation of 1.10.2026 again, with a log arm beside raw and clip_200 and with
# TabPFN-V3 at the package default of eight ensemble members (20.10.2026). Same draws and
# seeds. One part per call, so the two parts can run side by side on 6 threads each
# (rule 1); each runs its one-member file first, then the eight-member one. The
# predictions are in PREREGISTRATION.md, the verdicts from analysis/confirm_ensemble_log.py.
#
#     sh run/confirm_ensemble_log.sh unit
#     sh run/confirm_ensemble_log.sh segment
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}

case "$1" in
unit)
  MODELS=TabPFN-V3,TabICLv2,GBM REPEATS=20 N_FIT=2000 N_TEST=1000 N_EST=1 CLIP_C=200 \
    ARMS=raw,clip,log OUTPUT=unit_error_confirm_log.csv \
    $PY -u experiments/h3_repair/unit_error_confirm.py
  MODELS=TabPFN-V3 REPEATS=20 N_FIT=2000 N_TEST=1000 N_EST=8 CLIP_C=200 \
    ARMS=raw,clip,log OUTPUT=unit_error_confirm_nest8.csv \
    $PY -u experiments/h3_repair/unit_error_confirm.py
  ;;
segment)
  MODELS=TabPFN-V3,TabICLv2,GBM XI=0.7 SD_SHIFTS=1,20 CLIP_C=200 SEEDS=20 N_TRAIN=2000 \
    N_TEST=5000 N_EST=1 ARMS=raw,clip,log OUTPUT=legit_segment_log.csv \
    $PY -u experiments/h3_repair/legit_segment.py
  MODELS=TabPFN-V3 XI=0.7 SD_SHIFTS=1,20 CLIP_C=200 SEEDS=20 N_TRAIN=2000 \
    N_TEST=5000 N_EST=8 ARMS=raw,clip,log OUTPUT=legit_segment_nest8.csv \
    $PY -u experiments/h3_repair/legit_segment.py
  ;;
*)
  echo "usage: sh run/confirm_ensemble_log.sh unit|segment" >&2
  exit 2
  ;;
esac
