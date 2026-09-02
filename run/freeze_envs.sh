#!/bin/sh
# Freeze each virtualenv into requirements/, because the findings are claims about versions.
#
# WHY
# ---
# Section 2.1 of findings/NALAZI.md is a statement about what specific package
# versions do to the target variable: TabPFN 8.4.0, TabICL 2.1.1, TabDPT 1.2.0,
# TabFM 1.0.1, EXAONE-Tabular 1.0.0. The whole of H2 is the measured consequence.
# Nothing in the repository pins those versions, and three environments are in use
# with two different numpy majors between them.
#
# Every result row already carries its environment through results/provenance.csv,
# which common/append.py writes on the first write of a process. This is the other
# half: the exact set that has to be reinstalled to reproduce a run.
#
#     sh run/freeze_envs.sh
set -e
mkdir -p requirements

freeze() {
  name=$1; py=$2
  if [ -x "$py" ] || command -v "$py" >/dev/null 2>&1; then
    "$py" -m pip freeze > "requirements/$name.txt"
    echo "wrote requirements/$name.txt"
  else
    echo "skipped $name: no interpreter at $py"
  fi
}

freeze system           python
freeze venv-tfm         venv-tfm/Scripts/python.exe
freeze venv-tabfm       venv-tabfm/Scripts/python.exe
freeze venv-tabdpt      venv-tabdpt/Scripts/python.exe
freeze venv-graph       venv-graph/Scripts/python.exe
