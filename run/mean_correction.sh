#!/usr/bin/env bash
# The mean correction across the model family, each model in its own environment. A wide
# table can kill a process silently, so a failure does not stop the driver; the script is
# resumable and a second pass fills the gaps.
set -u
cd "$(dirname "$0")/.."

# TabFM's weights once filled the disk; refuse to start below 5 GB free.
free_gb=$(df -k /c 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
if [ "${free_gb:-99}" -lt 5 ]; then
  echo "STOP: only ${free_gb} GB free on C:"
  exit 1
fi

PY=${PY:-venv-tfm/Scripts/python.exe}

run() { echo "########## $1 ##########"; shift; "$@" || echo "  (process failed, continuing)"; }

run GBM          env MODEL=GBM         $PY -u experiments/h3_repair/mean_correction.py
run TabICLv2     env MODEL=TabICLv2    $PY -u experiments/h3_repair/mean_correction.py
run TabPFN-V3    env MODEL=TabPFN-V3   $PY -u experiments/h3_repair/mean_correction.py
run TabPFN-v2.5  env MODEL=TabPFN-v2.5 $PY -u experiments/h3_repair/mean_correction.py
run TabPFN-v2.6  env MODEL=TabPFN-v2.6 $PY -u experiments/h3_repair/mean_correction.py
run TabDPT       env MODEL=TabDPT      ./venv-tabdpt/Scripts/python.exe -u experiments/h3_repair/mean_correction.py
run EXAONE       env MODEL=EXAONE      ./venv-tabfm/Scripts/python.exe -u experiments/h3_repair/mean_correction.py
# Last: 6.6 GB of weights and about two and a half minutes per fit.
run TabFM        env MODEL=TabFM       ./venv-tabfm/Scripts/python.exe -u experiments/h3_repair/mean_correction.py
echo "done"
