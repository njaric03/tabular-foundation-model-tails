#!/bin/sh
# Stage 2 of the nanoTabPFN curriculum for every arm, with its learning check and gate.
#
# Runs from scratch never learned at 256 rows. A model that learned at 50 rows and 3
# features (A0B0-s42) was carried to 8 features (A0B0-s91, linear R2 +0.911) and then to
# 256 rows (A0B0-s92, R2 +0.950), and s92 reproduced the failure at N_TRAIN=200. Every arm
# here starts from the same s91, so the arms differ only in their last stage: the prior (A)
# and the target encoding (B) of the 256-row training. That is the design to describe.
#
# Every run gets s92's budget, 3755 steps of 16 tables, rather than a wall time, since
# parallel streams run slower. Seed 92 reads the shards small_s5026 and small_s5027, seed
# 93 the other two, so the second seed sees other tables, not the same ones in the same
# order. After each run: the learning check and the gate at 200 rows.
#
# The dumps are read from HDF5, not preloaded: two streams holding 2 GB each next to the
# TabICL evaluation left 1.4 GB of RAM free and stalled every process on this machine.
#
#     STREAM=1 sh run/prior_pretraining_curriculum.sh
#     STREAM=2 sh run/prior_pretraining_curriculum.sh
#
# Streams 3 and 4 split A1B0 into its two treatments, the heavy tail alone (ATB0) and the
# contaminated context alone (ACB0), on both seeds, one of each per stream. Each takes A1's
# draws on the same tables, so the three arms differ only in what is applied.
set -x
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MLFLOW_DISABLE_AGENT_HINT=1
PY=${PY:-venv-tfmp/Scripts/python.exe}
D=.cache/tfmp/dumps
INIT=.cache/tfmp/checkpoints/nanoTabPFN-A0B0-s91.pth

run() {
  ARM=$1 SEED=$2
  NAME=nanoTabPFN-$ARM-s$SEED
  if [ "$SEED" = 92 ]; then DUMPS=$D/small_s5026_60000.h5,$D/small_s5027_60000.h5
  else DUMPS=$D/small_s5028_60000.h5,$D/small_s5029_60000.h5; fi
  if [ ! -f ".cache/tfmp/checkpoints/$NAME.pth" ]; then
    $PY -u experiments/h3_repair/prior_pretraining_train.py --dump "$DUMPS" \
      --arm $ARM --seed $SEED --init $INIT --steps 3755 --batch_size 16 --accumulate 1 \
      --lr 3e-4 --warmup 0 --n_buckets 100 --edge_tables 2000 --target_clip 10000 \
      --grad_ckpt 1 --preload ${PRELOAD:-0} --log_every 200 --ckpt_every 500 || return 1
  fi
  $PY experiments/h3_repair/prior_pretraining_check.py --checkpoint \
    ".cache/tfmp/checkpoints/$NAME.pth" --context 200 --heldout $D/heldout_256_200.h5
  # One output file per stream: two processes never append to the same CSV (rule 1).
  TFMP_DEVICE=cuda MODELS=$NAME N_TRAIN=200 \
    OUTPUT=prior_pretraining_eval_curriculum_${STREAM:-1}.csv \
    $PY -u experiments/h3_repair/prior_pretraining_eval.py
}

case "${STREAM:-1}" in
  1) run A1B0 92; run A0B1 92 ;;
  2) run A0B0 93; run A1B0 93; run A1B1 92 ;;
  3) run ATB0 92; run ACB0 93 ;;
  4) run ACB0 92; run ATB0 93 ;;
esac
echo "STREAM_DONE"
