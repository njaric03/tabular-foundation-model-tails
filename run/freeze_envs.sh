#!/bin/sh
# Freeze every virtualenv into requirements/<env>.txt. Part two is a claim about what
# specific package versions do to the target, and results/provenance.csv records the
# versions per row; this is the full set to reinstall. Missing environments are skipped.
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

freeze venv-tfm         venv-tfm/Scripts/python.exe
freeze venv-tabfm       venv-tabfm/Scripts/python.exe
freeze venv-tabdpt      venv-tabdpt/Scripts/python.exe
freeze venv-data        venv-data/Scripts/python.exe
