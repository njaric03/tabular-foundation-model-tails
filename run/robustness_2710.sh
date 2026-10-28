#!/bin/sh
# Robustness checks of the leverage result, 27.10.2026: the position of the leverage row,
# a context of 10 000 rows, and the default ensemble with two more real tables. Three
# parts with their own outputs (rule 1), four threads each. What each could change is in
# PREREGISTRATION.md.
#
#     sh run/robustness_2710.sh position
#     sh run/robustness_2710.sh n10k
#     sh run/robustness_2710.sh real
#     sh run/robustness_2710.sh loss10k   (six threads; added after the context-size check)
set -x
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PY:-venv-tfm/Scripts/python.exe}
INF="DOSE_MODE=sd MODELS=TabICLv2,TabPFN-V3,GBM"

case "$1" in
position)
  for P in centre edge random; do
    env $INF DOSES=2,4,10,20,50 XI=0.3,0.7 POSITION=$P OUTPUT=influence_sd_pos_$P.csv \
      $PY -u experiments/h2_leverage/influence.py
  done
  ;;
n10k)
  env $INF DOSES=4,20 XI=0.7 N_TRAIN=10000 POSITION=centre OUTPUT=influence_sd_n10k.csv \
    $PY -u experiments/h2_leverage/influence.py
  ;;
real)
  V="VARIANTS=without_max,raw,clip_200 CLIP_C=200"
  env $V DATASETS=freMTPL2sev MODELS=TabPFN-V3 N_EST=8 OUTPUT=clip_context_real_nest8.csv \
    $PY -u experiments/h3_repair/clip_context_real.py
  env $V DATASETS=Brazilian_houses,OnlineNewsPopularity MODELS=TabPFN-V3,TabICLv2,GBM \
    N_EST=1 OUTPUT=clip_context_real_wide.csv $PY -u experiments/h3_repair/clip_context_real.py
  env $V DATASETS=Brazilian_houses,OnlineNewsPopularity MODELS=TabPFN-V3 N_EST=8 \
    OUTPUT=clip_context_real_wide_nest8.csv $PY -u experiments/h3_repair/clip_context_real.py
  ;;
loss10k)
  # The repairs' loss at 10 000 rows: raw, clip_200 and log on the expected pinball.
  export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
  for N in 1 8; do
    env MODEL=TabPFN-V3 XI=0.7 SD_SHIFTS=1,20 SEEDS=10 N_TRAIN=10000 N_TEST=900 \
      CLIP_C=200 ARMS=raw,clip,log N_EST=$N OUTPUT=clip_context_n10k_nest$N.csv \
      $PY -u experiments/h3_repair/clip_context.py
  done
  ;;
esac
