#!/bin/bash
# The prior pre-training runs on a Colab GPU with bf16, after run/colab_setup.sh.
#
# Locally, on an RTX 2060 in fp16, no setting at the table sizes the thesis needs produced
# a model that learns: at a learning rate of 1e-4 the linear-task R2 stayed near zero, and
# at 3e-4 runs either learned (50 rows, 3 features only) or blew up. bf16 needs no gradient
# scaler, and 24 to 40 GB hold a batch of 32 at 256 rows without checkpointing. The default
# is the pilot: A0B0 alone for one hour, then the learning check. The arms are worth
# running only if the pilot prints LEARNED.
#
#     bash run/prior_pretraining_colab.sh
#     ARMS="A0B0 A1B0" SEEDS="81 82" MAX_HOURS=2 bash run/prior_pretraining_colab.sh
#
# A run that hits MAX_HOURS keeps its work checkpoint and resumes when started again, but
# the clock counts the time already spent, so raise MAX_HOURS to continue it.
set -x
export MLFLOW_DISABLE_AGENT_HINT=1
PY=${PY:-python}
D=.cache/tfmp/dumps
DUMPS=${DUMPS:-$D/small_s5026_60000.h5,$D/small_s5027_60000.h5}
ARMS=${ARMS:-A0B0}
SEEDS=${SEEDS:-81}
STEPS=${STEPS:-200000}
MAX_HOURS=${MAX_HOURS:-1.0}
BATCH=${BATCH:-32}
LR=${LR:-3e-4}
WARMUP=${WARMUP:-0}
BUCKETS=${BUCKETS:-1000}
AMP=${AMP:-bf16}
CONTEXT=${CONTEXT:-200}
HELDOUT=${HELDOUT:-$D/heldout_256_200.h5}

for SEED in $SEEDS; do
  for ARM in $ARMS; do
    NAME=nanoTabPFN-$ARM-s$SEED
    if [ ! -f ".cache/tfmp/checkpoints/$NAME.pth" ]; then
      $PY -u experiments/h3_repair/prior_pretraining_train.py --dump "$DUMPS" \
        --arm $ARM --seed $SEED --steps $STEPS --batch_size $BATCH --accumulate 1 \
        --lr $LR --warmup $WARMUP --n_buckets $BUCKETS --edge_tables 2000 \
        --target_clip 10000 --grad_ckpt 0 --amp $AMP --max_hours $MAX_HOURS \
        --log_every 500 --ckpt_every 1000 || exit 1
    fi
    CK=".cache/tfmp/checkpoints/$NAME.pth"
    [ -f "$CK" ] || CK=".cache/tfmp/work/$NAME.pth"
    $PY experiments/h3_repair/prior_pretraining_check.py --checkpoint "$CK" \
      --context $CONTEXT --heldout "$HELDOUT"
  done
done
