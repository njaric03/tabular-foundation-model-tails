#!/bin/sh
# TabPFN-V3 at eight members with every member on the untransformed target, 28.10.2026: the
# test of why TabICLv2's leverage survives its ensemble and TabPFN-V3's does not. Two
# parts with their own outputs (rule 1). Rules Y0 to Y3 are in PREREGISTRATION.md.
#
#     sh run/tabpfn_y_2810.sh loss        (six threads)
#     sh run/tabpfn_y_2810.sh leverage    (four threads)
set -x
PY=${PY:-venv-tfm/Scripts/python.exe}

case "$1" in
loss)
  export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
  CC="MODEL=TabPFN-V3 ARMS=raw XI=0.7 SD_SHIFTS=1,20 SEEDS=20 N_TRAIN=2000 N_TEST=900"
  env $CC N_EST=1 OUTPUT=clip_context_ytf_nest1.csv \
    $PY -u experiments/h3_repair/clip_context.py
  env $CC N_EST=8 OUTPUT=clip_context_ytf_nest8.csv \
    $PY -u experiments/h3_repair/clip_context.py
  env $CC N_EST=8 TABPFN_Y=none OUTPUT=clip_context_ytf_nest8_none.csv \
    $PY -u experiments/h3_repair/clip_context.py
  ;;
leverage)
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  INF="DOSE_MODE=sd MODELS=TabPFN-V3 N_EST=8 DOSES=4,20 XI=0.7 POSITION=centre"
  env $INF OUTPUT=influence_sd_tabpfn8.csv $PY -u experiments/h2_leverage/influence.py
  env $INF TABPFN_Y=none OUTPUT=influence_sd_tabpfn8_none.csv \
    $PY -u experiments/h2_leverage/influence.py
  ;;
esac
