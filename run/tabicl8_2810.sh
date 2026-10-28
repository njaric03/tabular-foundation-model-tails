#!/bin/sh
# TabICLv2 at its default ensemble of eight, 28.10.2026: the leverage on the generator and
# on freMTPL2sev, then the repairs' loss at 2000 and 10 000 rows. Two parts with their own
# outputs (rule 1). Rules T1 to T5 are in PREREGISTRATION.md.
#
#     sh run/tabicl8_2810.sh leverage   (four threads)
#     sh run/tabicl8_2810.sh loss       (six threads)
set -x
PY=${PY:-venv-tfm/Scripts/python.exe}

case "$1" in
leverage)
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  INF="DOSE_MODE=sd MODELS=TabICLv2 N_EST=8 DOSES=4,20 XI=0.7 POSITION=centre"
  env $INF OUTPUT=influence_sd_tabicl8.csv $PY -u experiments/h2_leverage/influence.py
  env $INF N_TRAIN=10000 OUTPUT=influence_sd_tabicl8_n10k.csv \
    $PY -u experiments/h2_leverage/influence.py
  env VARIANTS=without_max,raw,clip_200,log CLIP_C=200 DATASETS=freMTPL2sev \
    MODELS=TabICLv2 N_EST=8 OUTPUT=clip_context_real_tabicl8.csv \
    $PY -u experiments/h3_repair/clip_context_real.py
  ;;
loss)
  export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
  for N in 1 8; do
    for T in 2000 10000; do
      env MODEL=TabICLv2 XI=0.7 SD_SHIFTS=1,20 SEEDS=10 N_TRAIN=$T N_TEST=900 \
        CLIP_C=200 ARMS=raw,clip,log N_EST=$N \
        OUTPUT=clip_context_tabicl_n$((T / 1000))k_nest$N.csv \
        $PY -u experiments/h3_repair/clip_context.py
    done
  done
  ;;
esac
