#!/bin/sh
# The dose response measured against the sd shift, in the range that actually occurs.
#
# WHY
# ---
# Two problems with the dose as it has been defined, both visible in the data
# that is already in the repository.
#
# 1. THE TREATMENT IS NOT FIXED. y0 = dose * max(y_train), and max(y_train) is
#    itself a heavy-tailed random variable. Recomputed from the seeds of
#    influence.csv, one nominal dose of 100 is:
#
#        xi = 0.3   sd shift 30.7 to 39.4
#        xi = 0.7   sd shift 50.3 to 97.1
#
#    a factor of two in the causal quantity, across seeds, at one nominal dose.
#    Part of the scatter that twelve self-corrections chased is scatter in the
#    treatment, not in the model. Dosing on the sd shift removes it.
#
# 2. THE GRID MISSES THE REAL RANGE. The prevalence survey measured the sd shift
#    on 99 public datasets: 5/99 above 2x, 2/99 above 4x, 1/99 above 10x, worst
#    case freMTPL2sev at 19.9x. The existing grid jumps from about 7 (dose 10) to
#    about 71 (dose 100), so the region 15 to 25 -- the realistic worst case --
#    has never been measured, while the headline number comes from 50 to 97,
#    which is two to five times beyond anything observed in 99 datasets.
#
# The targets below are read straight off that survey: 2 and 4 are the thresholds
# it reports, 10 is the 1-in-99 case, 20 is the worst real dataset, and 50 keeps
# one point in the old regime so the two dose definitions can be joined.
#
# WHAT IT SETTLES
# ---------------
# Whether the response is a smooth function of the sd shift, which the correlation
# in the existing rows already suggests -- rho(sd shift, |local Q(0.99) change|)
# is 0.85 for TabPFN-V3, 0.67 for TabICLv2 and 0.15 for GBM, the control -- and
# what the effect is at the leverage that public data actually contains. That
# number, not the one at dose 100, is what the abstract should quote.
#
# COST
# ----
# About 2 hours on CPU: three models, two tail indices, five targets, five seeds.
# Resumable, and `dose_mode` is part of the key, so these rows cannot be confused
# with the existing ones measured on the old definition.
#
#     sh run/run_influence_sd.sh
#
# TabPFN needs a Prior Labs account before its weights load; see README, Setup.
# Until that is done, run the rest and add TabPFN-V3 later, since every cell is keyed:
#
#     MODELS=TabICLv2,GBM sh run/run_influence_sd.sh
#
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}   # another environment: PY=... sh run/...

MODELS=${MODELS:-TabICLv2,TabPFN-V3,GBM} XI=0.3,0.7 \
  DOSE_MODE=sd DOSES=2,4,10,20,50 SEEDS=5 N_EST=1 \
  OUTPUT=influence_sd.csv \
  $PY -u experiments/h2_leverage/influence.py
