#!/bin/sh
# TabPFN-V3 measured again after the seeding fix of 25.8.2026. Before it, TabPFN got
# random_state=0 while the other models got the seed, so its ensemble permutations were
# frozen across seeds, which understates the spread and ties the point to one draw. The
# new rows go to `*_tabpfn_reseed*` files next to the old ones, and compare_reseeding.py
# prints the difference. The n_est=4 runs come first, since they carry the headline numbers.
#
#     sh run/tabpfn_reseed.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}
M=TabPFN-V3

MODELS=$M N_EST=4 SEEDS=20 OUTPUT=shape_of_x_tabpfn_reseed_20.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py
MODELS=$M N_EST=4 SEEDS=20 OUTPUT=scale_of_x_tabpfn_reseed_20.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py

MODELS=$M N_EST=1 SEEDS=20 OUTPUT=shape_of_x_tabpfn_reseed_nest1.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py
MODELS=$M N_EST=1 SEEDS=20 OUTPUT=scale_of_x_tabpfn_reseed_nest1.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py

MODELS=$M N_EST=1 SEEDS=5 OUTPUT=sample_complexity_tabpfn_reseed.csv \
  $PY -u experiments/h1_shape_vs_scale/sample_complexity.py
MODELS=$M SEEDS=5 OUTPUT=tail_families_tabpfn_reseed.csv \
  $PY -u experiments/h1_shape_vs_scale/tail_families.py
MODELS=$M SEEDS=10 OUTPUT1=xi_residual_tabpfn_reseed_part1.csv \
  OUTPUT2=xi_residual_tabpfn_reseed_part2.csv \
  $PY -u experiments/h1_shape_vs_scale/xi_residual.py
MODELS=$M N_EST=1 SEEDS=3 OUTPUT=dissociation_real_tabpfn_reseed.csv \
  $PY -u experiments/h1_shape_vs_scale/dissociation_real.py

$PY -u experiments/h1_shape_vs_scale/compare_reseeding.py
