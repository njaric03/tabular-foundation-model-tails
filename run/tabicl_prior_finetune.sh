#!/bin/sh
# Continued pre-training of TabICLv2: FT-A0 (its own prior as generated) and FT-A1 (the
# same tables with heavy tails and contaminated contexts) on two seeds, from
# tabicl-regressor-v2-20260212.ckpt. 20000 tables of 1024 rows each at a batch of 4 with
# the column and row stages frozen: 1.5 GB and about 6.8 tables per second on the
# RTX 2060, so about 50 minutes per run. Seed 1 of both arms comes first, so a stop halfway
# still leaves one pair. The predictions are in the docstring of the script.
#
#     sh run/tabicl_prior_finetune.sh
set -x
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 MLFLOW_DISABLE_AGENT_HINT=1
PY=${PY:-venv-tfmp/Scripts/python.exe}
D=.cache/tfmp/dumps
DUMPS=${DUMPS:-$D/base_s2026_14000.h5,$D/base_s2027_14000.h5}
TABLES=${TABLES:-20000}
SEEDS=${SEEDS:-"1 2"}
ARMS=${ARMS:-"A0 A1"}

for SEED in $SEEDS; do
  for ARM in $ARMS; do
    if [ -f ".cache/tfmp/tabicl_ft/TabICLv2-FT-$ARM-s$SEED.ckpt" ]; then
      continue
    fi
    $PY -u experiments/h3_repair/tabicl_prior_finetune.py --dump "$DUMPS" \
      --arm $ARM --seed $SEED --tables $TABLES --log_every 50 --ckpt_every 250 || exit 1
  done
done
