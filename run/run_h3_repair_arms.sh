#!/bin/sh
# The four repair sweeps of the third part: the context arms and the output patch.
#
# WHY
# ---
# These four scripts produce the largest measurement in the repository --
# tail_splice.csv alone is 50400 rows -- and until now none of them had a driver.
# Every knob they take came from the environment of whatever shell ran them, and
# two of those knobs reach no column: CLIP_C here (the clip control's cap) and
# U_SHARES (the GPD threshold grid, which reaches the `arm` NAME but not a column
# of its own). Without this file the invocation behind the committed CSVs is not
# reconstructable from the repository, which is rule 2 read one level up: a knob
# that only the caller remembers is a knob that is lost.
#
# THE PARAMETERS, AND WHY THESE
# -----------------------------
# MODELS=TabPFN-V3,TabICLv2,GBM  The two foundation models that carry the results
#     plus the tree control that does not standardise the target. EXAONE and
#     TabDPT need other virtualenvs; they are a separate run, not a variant here.
# N_EST=1  The one free recommendation of the third part. The whole ensemble loss
#     happens at the first aggregation, so measuring a repair at N_EST > 1 would
#     confound the repair with a defect the repair is not about.
# CLIP_C=200  The clip control. 200 robust sd is the loose cap of the first run,
#     the one that was not measurably costly on a clean context; 50 is the tight
#     one, and it is a column in clip_context.py's own grid rather than here.
#     NOTE the name collides: clip_context.py reads CLIP_C as a comma-separated
#     GRID and tail_splice.py as ONE value, so they cannot share an export. They
#     are set per command below for that reason.
# U_SHARES=0.8,0.85,0.9,0.95  The GPD threshold as a sensitivity rather than a
#     pick. The grid is in the arm name, so a different grid writes different arms
#     and cannot resume over these rows.
# SEEDS=20 / REPEATS=10  Twenty seeds on the generator, where a seed is free; ten
#     repeats on the real tables, where each one is a fresh subsample and a fresh
#     model fit.
#
# COST
# ----
# About 30 hours on CPU for all four, tail_splice being most of it. Each command
# writes its own OUTPUT and is resumable, so they may be run one at a time and
# interrupted; rule 1 applies, one process per output file.
#
#     sh run/run_h3_repair_arms.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=venv-tfm/Scripts/python.exe

# The generator: every context arm against a dosed sd shift, 20 seeds.
MODEL=TabPFN-V3 XI=0.7,0.9 SD_SHIFTS=1,4,20,50 CLIP_C=20,50,200 SEEDS=20 \
  N_EST=1 N_TRAIN=2000 N_TEST=900 OUTPUT=clip_context.csv \
  $PY -u experiments/h3_repair/clip_context.py

# The same arms on the real tables, in the leverage bins of the prevalence survey.
MODELS=TabPFN-V3,TabICLv2,GBM CLIP_C=20,50,200 N_EST=1 \
  OUTPUT=clip_context_real.csv \
  $PY -u experiments/h3_repair/clip_context_real.py

# The same arms under a unit error, where the corrupted row is the top exceedance.
MODELS=TabPFN-V3,TabICLv2,GBM REPEATS=10 N_FIT=2000 N_TEST=1000 N_EST=1 \
  OUTPUT=unit_error_real.csv \
  $PY -u experiments/h3_repair/unit_error_real.py

# The output patch: conformal, with the GPD threshold measured as a grid and
# picked by two rules. CLIP_C is ONE value here, not the grid above.
MODELS=TabPFN-V3,TabICLv2,GBM REPEATS=10 N_FIT=1500 N_CALIB=500 N_TEST=1000 \
  N_EST=1 CLIP_C=200 U_SHARES=0.8,0.85,0.9,0.95 OUTPUT=tail_splice.csv \
  $PY -u experiments/h3_repair/tail_splice.py
