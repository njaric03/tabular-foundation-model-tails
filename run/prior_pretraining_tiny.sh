#!/bin/sh
# The 2 x 2 on 50-row tables, the size TFM-Playground released its regressor at, and the
# first setting here whose models learn anything.
#
# The two earlier settings (1024 and 256 rows) used TFM-Playground's default learning rate
# of 1e-4 and produced models with an R2 of about -0.01 on a plain linear task, where the
# released regressor gets +0.46. At 3e-4 and 10000 steps of 16 tables, eleven minutes on
# the RTX 2060, the same code reaches +0.82. What was missing was the learning rate, not
# the tables: the step rate is fixed by launch overhead at about 15 steps per second, so
# gradient checkpointing is off here and the batch is free data.
#
# Contexts are 20 to 39 rows, so the evaluation runs at N_TRAIN=35. That is far below the
# hundreds of rows the thesis measures: this tests the mechanism, not the size of the
# effect.
#
#     sh run/prior_pretraining_tiny.sh
#     ARMS="A0B0 A1B0" SEEDS=41 sh run/prior_pretraining_tiny.sh
set -x
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 MLFLOW_DISABLE_AGENT_HINT=1
PY=${PY:-venv-tfmp/Scripts/python.exe}
D=.cache/tfmp/dumps
# tiny8 holds 8 features; the first tiny dumps had 3, and the generator of this repo draws
# 5, so the adapter refused every cell of the first gate.
DUMPS=${DUMPS:-$D/tiny8_s7026_25000.h5,$D/tiny8_s7027_25000.h5,$D/tiny8_s7028_25000.h5,$D/tiny8_s7029_25000.h5}
ARMS=${ARMS:-"A0B0 A1B0 A0B1 A1B1"}
SEEDS=${SEEDS:-"61 62"}
STEPS=${STEPS:-8000}
LR=${LR:-3e-4}
WARMUP=${WARMUP:-1000}
BATCH=${BATCH:-16}
BUCKETS=${BUCKETS:-100}

for SEED in $SEEDS; do
  for ARM in $ARMS; do
    if [ -f ".cache/tfmp/checkpoints/nanoTabPFN-$ARM-s$SEED.pth" ]; then
      continue
    fi
    $PY -u experiments/h3_repair/prior_pretraining_train.py --dump "$DUMPS" \
      --arm $ARM --seed $SEED --steps $STEPS --batch_size $BATCH --accumulate 1 --lr $LR \
      --warmup $WARMUP --n_buckets $BUCKETS --edge_tables 2000 --target_clip 10000 \
      --grad_ckpt 0 --log_every 1000 --ckpt_every 2500 || exit 1
  done
done
