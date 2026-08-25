#!/bin/sh
# Intervencija jednom tackom, EXAONE: pet seedova i zabelezen n_est.
#
# ZASTO. `influence_exaone.csv` je posle 27.8. imao dva nejednaka bloka. Blokovi xi=0,3 i
# xi=0,7 mereni su sa `N_EST=4` upisanim u kolonu i reprodukovali su se bit-identicno.
# Blok xi=0,9 je od 25.8., pre nego sto je `n_est` postao kolona, pa mu je to polje
# prazno: pokretanje je vodjeno kao N_EST=4, ali zapis to ne dokazuje. Iz tog bloka
# dolazi naslovna brojka nalaza (+671% na Q(0,99) pri dozi 100x), pa ne sme da stoji na
# nezabelezenom parametru.
#
# ZASTO PET SEEDOVA. Ostali modeli u `findings/h2/preprocessing_asymmetry.md` mereni su na pet
# seedova; EXAONE je bio na tri, pa uparivanje nije bilo posteno. `findings/h2/exaone.md` sam
# belezi da je pri dozi 100x raspon +2,1% do +759,6% preko tri seeda -- bimodalno, kao
# TabPFN. Na tri seeda medijana takvog raspona nije stabilna.
#
# STO SE NECE PONOVO MERITI. `KLJUC` u `experiments/h2_leverage/influence.py` ukljucuje `n_est`, pa se
# postojeca 24 reda na xi=0,3 i 0,7 sa n_est=4 preskacu. Ponovo se mere samo seedovi
# 10000 i 11000 na ta dva xi (16 redova) i ceo xi=0,9 na pet seedova (20 redova), jer
# tamo prazan n_est nije jednak cetvorci. Ukupno 36 novih redova.
#
# POSLE. Kad run prodje, dvanaest redova xi=0,9 sa praznim `n_est` se sklanja u arhivu:
# zamenjeni su istim merenjem sa zabelezenim parametrom.
#
#     sh run/run_exaone_influence_5seeds.sh
#
set -x
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
PY=venv-tabfm/Scripts/python.exe

MODELS=EXAONE XI=0.3,0.7,0.9 DOSES=1,5,10,100 SEEDS=5 N_EST=4 \
  OUTPUT=influence_exaone.csv \
  $PY -u experiments/h2_leverage/influence.py
