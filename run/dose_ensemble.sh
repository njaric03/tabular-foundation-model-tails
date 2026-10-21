#!/bin/sh
# The dose curve of clip_context.py again, with raw, clip_200 and log only, at one and eight
# members, for TabPFN-V3 on CPU (venv-tfm, as every V3 row) and TabPFN-3.5 on the GPU
# (venv-tfm9cu), with the expected pinball beside the drawn one (22.10.2026). The two parts
# write different files and run side by side (rule 1). The CPU part runs the shifts the
# predictions name first. After its dose curve the GPU part runs the measurements of
# run/new_models.sh for TabPFN-3.5, which carry no prediction, as for LimiX-2 and Causilo.
# Predictions in PREREGISTRATION.md, verdicts from analysis/dose_ensemble.py.
#
#     sh run/dose_ensemble.sh cpu
#     sh run/dose_ensemble.sh gpu
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
COMMON="XI=0.7,0.9 SEEDS=20 N_TRAIN=2000 N_TEST=900 CLIP_C=200 ARMS=raw,clip,log"

case "$1" in
cpu)
  PY=${PY:-venv-tfm/Scripts/python.exe}
  env $COMMON MODEL=TabPFN-V3 SD_SHIFTS=1,4,20,50 N_EST=1 \
    OUTPUT=clip_context_dose_v3_nest1.csv $PY -u experiments/h3_repair/clip_context.py
  env $COMMON MODEL=TabPFN-V3 SD_SHIFTS=1,50 N_EST=8 \
    OUTPUT=clip_context_dose_v3_nest8.csv $PY -u experiments/h3_repair/clip_context.py
  env $COMMON MODEL=TabPFN-V3 SD_SHIFTS=1,4,20,50 N_EST=8 \
    OUTPUT=clip_context_dose_v3_nest8.csv $PY -u experiments/h3_repair/clip_context.py
  ;;
gpu)
  PY=${PY:-venv-tfm9cu/Scripts/python.exe}
  export TABPFN_DEVICE=cuda
  for N in 1 8; do
    env $COMMON MODEL=TabPFN-v3.5 SD_SHIFTS=1,4,20,50 N_EST=$N \
      OUTPUT=clip_context_dose_tabpfn35_nest$N.csv $PY -u experiments/h3_repair/clip_context.py
  done
  PY=$PY MODELS=TabPFN-v3.5 TAG=tabpfn35 sh run/new_models.sh
  ;;
*)
  echo "usage: sh run/dose_ensemble.sh cpu|gpu" >&2
  exit 2
  ;;
esac
