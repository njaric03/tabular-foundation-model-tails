#!/bin/sh
# TabPFN-3.5, the package default since 15.9.2026, through the confirmation of 1.10.2026
# with raw, clip_200 and log at one and eight members (21.10.2026). On the RTX 2060 from
# venv-tfm9cu, the packages of venv-tfm9 with a CUDA build of torch. One GPU, so the four
# files run one after another (rule 1). A first fit on random data of the largest size the
# run meets (2001 context rows, 13 features, 5000 test rows, eight members) checks the GPU
# memory and stops the script if it fails. The predictions are in PREREGISTRATION.md, the
# verdicts from analysis/confirm_tabpfn35.py.
#
#     sh run/confirm_tabpfn35.sh
set -e
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TABPFN_DEVICE=cuda
PY=${PY:-venv-tfm9cu/Scripts/python.exe}
M=TabPFN-v3.5

$PY - <<'EOF'
import time

import numpy as np
import torch

from common import models

rng = np.random.default_rng(0)
X, y = rng.normal(size=(2001, 13)), np.exp(rng.normal(size=2001))
t = time.time()
models.quantiles("TabPFN-v3.5", X, y, rng.normal(size=(5000, 13)), seed=0,
                 levels=[0.5, 0.9, 0.99, 0.999], n_est=8)
print(f"memory check: {time.time() - t:.0f} s, "
      f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB", flush=True)
EOF

for N in 1 8; do
  MODELS=$M REPEATS=20 N_FIT=2000 N_TEST=1000 N_EST=$N CLIP_C=200 ARMS=raw,clip,log \
    OUTPUT=unit_error_confirm_tabpfn35_nest$N.csv \
    $PY -u experiments/h3_repair/unit_error_confirm.py
  MODELS=$M XI=0.7 SD_SHIFTS=1,20 CLIP_C=200 SEEDS=20 N_TRAIN=2000 N_TEST=5000 \
    N_EST=$N ARMS=raw,clip,log OUTPUT=legit_segment_tabpfn35_nest$N.csv \
    $PY -u experiments/h3_repair/legit_segment.py
done
