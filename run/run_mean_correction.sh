#!/usr/bin/env bash
# Korekcija sredine preko cele familije. Svaki model u svom okruzenju.
# `|| true` je bitan: siroki skupovi (Allstate ima 130 atributa) znaju da obore proces
# tihim padom, a bez toga to obori i ceo driver. Skripta je nastavljiva, pa se
# ponovnim pokretanjem popunjavaju rupe.
set -u
cd "$(dirname "$0")"
# Ne pokreci ako je disk skoro pun. Jednom je vec zavrsio na 100% zbog 6,6 GB tezina.
slobodno=$(df -k /c 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
if [ "${slobodno:-99}" -lt 5 ]; then
  echo "STOP: samo ${slobodno} GB slobodno na C:. Oslobodi prostor pa pusti ponovo."
  exit 1
fi
echo "slobodno na disku: ${slobodno} GB"

run() { echo "########## $1 ##########"; shift; "$@" || echo "  (proces pao, nastavljam)"; }

run GBM        env MODEL=GBM       python -u experiments/h3_repair/mean_correction.py
run TabICLv2   env MODEL=TabICLv2  python -u experiments/h3_repair/mean_correction.py
run TabPFN-V2  env MODEL=TabPFN-V2 TABPFN_MODEL_VERSION=v2 python -u experiments/h3_repair/mean_correction.py
run TabPFN-V3  env MODEL=TabPFN-V3 TABPFN_MODEL_VERSION=v3 python -u experiments/h3_repair/mean_correction.py
run TabDPT     env MODEL=TabDPT    ./venv-tabdpt/Scripts/python.exe -u experiments/h3_repair/mean_correction.py
echo "SVE GOTOVO"
