#!/bin/sh
# Pre-training of the eight nanoTabPFN arms: prior A0/A1 x encoding B0/B1 x seeds 1, 2.
# Every run gets the same 7000 steps of 8 tables from the same four dump shards in the same
# order, 1000 buckets, and differs only in the factor under test. The budget was fixed
# after the pilot in data/tfmp_artifacts.json: 5.3 tables per second on the RTX 2060, so
# about 3 hours per run and 24 in all. Seed 1 of every arm comes first, so a stop halfway
# still leaves one complete 2 x 2. A finished arm is skipped; an interrupted one resumes
# from .cache/tfmp/work/.
#
#     sh run/prior_pretraining.sh
set -x
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MLFLOW_DISABLE_AGENT_HINT=1
PY=${PY:-venv-tfmp/Scripts/python.exe}
D=.cache/tfmp/dumps
DUMPS=$D/base_s2026_14000.h5,$D/base_s2027_14000.h5,$D/base_s2028_14000.h5,$D/base_s2029_14000.h5

for SEED in 1 2; do
  for ARM in A0B0 A0B1 A1B0 A1B1; do
    if [ -f ".cache/tfmp/checkpoints/nanoTabPFN-$ARM-s$SEED.pth" ]; then
      continue
    fi
    $PY -u experiments/h3_repair/prior_pretraining_train.py --dump "$DUMPS" \
      --arm $ARM --seed $SEED --steps 7000 --batch_size 4 --accumulate 2 \
      --n_buckets 1000 --edge_tables 2000 --log_every 50 --ckpt_every 250 || exit 1
  done
done
