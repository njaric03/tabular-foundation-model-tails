#!/bin/sh
# The same 2 x 2 as run/prior_pretraining.sh on smaller tables: 256 rows and up to 16
# features instead of 1024 and 32. The first setting failed its own gate with a model that
# predicts nearly a constant (R2 -0.006 on held-out prior tables), and the attention
# between datapoints costs the square of the rows, so this one buys tables with rows:
# 320000 tables of 256 rows in about three hours against 56000 of 1024 rows, at 30 tables
# per second with a batch of 32. Contexts are 128 to 224 rows, so the evaluation runs at
# N_TRAIN=200.
#
# ARMS and SEEDS can be overridden; by default this runs A0B0 alone, because the other
# arms are only worth training if the gate reproduces the failure:
#
#     sh run/prior_pretraining_small.sh
#     ARMS="A1B0" sh run/prior_pretraining_small.sh
set -x
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MLFLOW_DISABLE_AGENT_HINT=1
PY=${PY:-venv-tfmp/Scripts/python.exe}
D=.cache/tfmp/dumps
DUMPS=$D/small_s5026_60000.h5,$D/small_s5027_60000.h5,$D/small_s5028_60000.h5,$D/small_s5029_60000.h5
# Seeds 11 and 12 belong to this setting; 1 to 3 are the 1024-row one and must not be
# overwritten.
ARMS=${ARMS:-A0B0}
SEEDS=${SEEDS:-11}

for SEED in $SEEDS; do
  for ARM in $ARMS; do
    if [ -f ".cache/tfmp/checkpoints/nanoTabPFN-$ARM-s$SEED.pth" ]; then
      continue
    fi
    $PY -u experiments/h3_repair/prior_pretraining_train.py --dump "$DUMPS" \
      --arm $ARM --seed $SEED --steps 10000 --batch_size 32 --accumulate 1 \
      --n_buckets 1000 --edge_tables 2000 --target_clip 10000 \
      --log_every 100 --ckpt_every 250 || exit 1
  done
done
