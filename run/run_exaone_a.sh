#!/bin/sh
# Stream A: disocijacija na stvarnim skupovima.
#
# ISPRAVKA 25.8. uvece: prvi pokusaj je isao sa N_EST=4 i pao je na prvom skupu
# (OnlineNewsPopularity, 60 atributa) -- 2h44m bez ijednog zapisanog reda.
# Dva razloga, oba resena ovde:
#   1. Postojeci rezultat na 9 skupova (`findings/h1/dissociation_real.md`) je radjen sa
#      n_estimators = 1. N_EST=4 nije bio samo skup nego i NEUPOREDIV. N_EST=1 je i
#      tacan izbor i 4x jeftiniji.
#   2. Skupovi idu redom po broju atributa, od 6 do 130. EXAONE-ov CAST radi paznju i po
#      osi atributa (feature_attention_repeats=2), pa cena raste sa `d`. Ovako se rezultati
#      za uske skupove upisu u prvih par sati umesto da cekaju za najskupljim.
set -x
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY=venv-tabfm/Scripts/python.exe
REDOM="CPS1988,218_house_8L,houses,diamonds,particulate-matter-ukair-2017,house_16H,OnlineNewsPopularity,Buzzinsocialmedia_Twitter,superconduct,Allstate_Claims_Severity"
MODELS=EXAONE N_EST=1 SEEDS=3 DATASETS="$REDOM" OUTPUT=dissociation_real_exaone.csv \
  $PY -u experiments/h1_shape_vs_scale/dissociation_real.py
