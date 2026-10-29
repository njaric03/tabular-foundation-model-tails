#!/bin/sh
# The conditional cap beside raw and clip_200 on the draws and seeds of the confirmation
# of 1.10.2026, one member (29.10.2026). One part per call, so the two parts can run side
# by side on 6 threads each (rule 1). raw and clip_200 are measured again and must equal
# the stored rows of 20.10.2026; log is read from those files. The predictions are in
# PREREGISTRATION.md, the verdicts from analysis/conditional_cap.py.
#
#     sh run/conditional_cap.sh unit
#     sh run/conditional_cap.sh segment
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}

case "$1" in
unit)
  MODELS=TabPFN-V3,TabICLv2,GBM REPEATS=20 N_FIT=2000 N_TEST=1000 N_EST=1 CLIP_C=200 \
    ARMS=raw,clip,cclip OUTPUT=unit_error_confirm_cclip.csv \
    $PY -u experiments/h3_repair/unit_error_confirm.py
  ;;
segment)
  MODELS=TabPFN-V3,TabICLv2,GBM XI=0.7 SD_SHIFTS=1,20 CLIP_C=200 SEEDS=20 N_TRAIN=2000 \
    N_TEST=5000 N_EST=1 ARMS=raw,clip,cclip OUTPUT=legit_segment_cclip.csv \
    $PY -u experiments/h3_repair/legit_segment.py
  ;;
*)
  echo "usage: sh run/conditional_cap.sh unit|segment" >&2
  exit 2
  ;;
esac
