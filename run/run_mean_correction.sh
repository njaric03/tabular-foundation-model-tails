#!/usr/bin/env bash
# Korekcija sredine preko cele familije. Svaki model u svom okruzenju.
# `|| true` je bitan: siroki skupovi (Allstate ima 130 atributa) znaju da obore proces
# tihim padom, a bez toga to obori i ceo driver. Skripta je nastavljiva, pa se
# ponovnim pokretanjem popunjavaju rupe.
set -u
# ".." jer su putanje ispod relativne od korena repozitorijuma, kao u svakom
# drugom driveru; bez toga `cd` vodi u `run/`, gde `experiments/` ne postoji.
cd "$(dirname "$0")/.."
# Ne pokreci ako je disk skoro pun. Jednom je vec zavrsio na 100% zbog 6,6 GB tezina.
slobodno=$(df -k /c 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
if [ "${slobodno:-99}" -lt 5 ]; then
  echo "STOP: samo ${slobodno} GB slobodno na C:. Oslobodi prostor pa pusti ponovo."
  exit 1
fi
echo "slobodno na disku: ${slobodno} GB"

run() { echo "########## $1 ##########"; shift; "$@" || echo "  (proces pao, nastavljam)"; }

run GBM          env MODEL=GBM         python -u experiments/h3_repair/mean_correction.py
run TabICLv2     env MODEL=TabICLv2    python -u experiments/h3_repair/mean_correction.py
run TabPFN-V3    env MODEL=TabPFN-V3   python -u experiments/h3_repair/mean_correction.py
# TabPFN-V2 je ranije biran preko TABPFN_MODEL_VERSION, promenljive samog paketa,
# koju provenance nije belezio, pa ti redovi nisu imenovali checkpoint. Generacija
# sada ide kroz TABPFN_PATHS u `common/models.py`, pa se mere obe imenovano.
run TabPFN-v2.5  env MODEL=TabPFN-v2.5 python -u experiments/h3_repair/mean_correction.py
run TabPFN-v2.6  env MODEL=TabPFN-v2.6 python -u experiments/h3_repair/mean_correction.py
run TabDPT       env MODEL=TabDPT      ./venv-tabdpt/Scripts/python.exe -u experiments/h3_repair/mean_correction.py
run EXAONE       env MODEL=EXAONE      ./venv-tabfm/Scripts/python.exe -u experiments/h3_repair/mean_correction.py
# TabFM ide poslednji: 6,6 GB tezina i oko dva i po minuta po fitu, na smanjenom
# profilu koji skripta sama bira. Mean-only model, pa je za nalaz o sredini
# najrelevantniji, i do sada nije bio meren ni u jednom redu.
run TabFM        env MODEL=TabFM       ./venv-tabfm/Scripts/python.exe -u experiments/h3_repair/mean_correction.py
echo "SVE GOTOVO"
