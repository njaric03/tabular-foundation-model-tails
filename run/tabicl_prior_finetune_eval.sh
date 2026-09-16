#!/bin/sh
# The four fine-tuned TabICLv2 arms on the cells where the base model loses 15.4% of
# Q(0.99): clip_context_real.py on freMTPL2sev, the same 20 subsamples, the same sizes and
# seed, on the CPU at 6 threads as the base model was measured. Predictions F1 to F4 are in
# experiments/h3_repair/tabicl_prior_finetune.py.
#
# The base TabICLv2 runs again here first. Its rows in clip_context_real.csv came from
# tabicl 2.1.1 in venv-tfm; the arms run in venv-tfmp with tabicl 2.2.0. Both default to
# the same checkpoint and outlier threshold, but the comparison is made against the base
# model measured in the same environment, and the two base runs against each other are the
# control that the environment did not move the number.
#
#     sh run/tabicl_prior_finetune_eval.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 MLFLOW_DISABLE_AGENT_HINT=1
PY=${PY:-venv-tfmp/Scripts/python.exe}
FT=TabICLv2,TabICLv2-FT-A0-s1,TabICLv2-FT-A1-s1,TabICLv2-FT-A0-s2,TabICLv2-FT-A1-s2

# Only the arms the predictions need, and clip_200 as the repair to compare with: 300 fits
# on the CPU instead of 1300.
MODELS=${MODELS:-$FT} DATASETS=freMTPL2sev VARIANTS=without_max,raw,clip_200 \
  OUTPUT=clip_context_real_tabicl_ft.csv \
  $PY -u experiments/h3_repair/clip_context_real.py
