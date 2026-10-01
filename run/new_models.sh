#!/bin/sh
# The models released in September 2026 through the measurements that carry
# the thesis: natural extremes and capping on freMTPL2sev, the unit error on seven tables,
# one leverage row dosed on the maximum and on the sd shift, and shape against scale on
# the generator. One ensemble member, as everywhere. Causilo and TabPFN-3.5 live in
# venv-tfm9 (tabpfn 9.0.0), LimiX-2 in venv-limix on the GPU. Resumable; each step writes
# its own file, named by TAG. Measured: 1.5 h for Causilo on CPU; 2.5 h for LimiX-2 on an
# RTX 2060, plus about 7 h for BlogFeedback (280 features) in the unit-error step.
#
#     sh run/new_models.sh
#     MODELS=TabPFN-v3.5 TAG=tabpfn35 sh run/new_models.sh
#     PY=venv-limix/Scripts/python.exe MODELS=LimiX-2 TAG=limix2 sh run/new_models.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm9/Scripts/python.exe}
M=${MODELS:-Causilo}
TAG=${TAG:-causilo}

MODELS=$M N_EST=1 OUTPUT=clip_context_real_$TAG.csv \
  $PY -u experiments/h3_repair/clip_context_real.py

MODELS=$M XI=0.3,0.7 DOSE_MODE=sd DOSES=2,4,10,20,50 SEEDS=5 N_EST=1 \
  OUTPUT=influence_sd_$TAG.csv $PY -u experiments/h2_leverage/influence.py

MODELS=$M XI=0.3,0.7 DOSES=1,3,10,100 SEEDS=5 N_EST=1 OUTPUT=influence_$TAG.csv \
  $PY -u experiments/h2_leverage/influence.py

MODELS=$M N_EST=1 SEEDS=20 OUTPUT=shape_of_x_$TAG.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py

MODELS=$M N_EST=1 SEEDS=20 OUTPUT=scale_of_x_$TAG.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py

MODELS=$M N_EST=1 VARIANTS=raw,clip_50,clip_200,sd_cap OUTPUT=unit_error_real_$TAG.csv \
  $PY -u experiments/h3_repair/unit_error_real.py
