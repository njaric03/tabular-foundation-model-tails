#!/bin/sh
# Ponovno merenje TabPFN-V3 posle ispravke sejanja (25.8.2026).
#
# ZASTO
# -----
# `experiments/h1_shape_vs_scale/shape_of_x.py` je TabPFN-u davao `random_state=0` dok su TabICLv2 i EXAONE
# dobijali `seed`. Podaci jesu varirali po seedu, ali je TabPFN kroz svih 20 seedova
# vrteo ISTE permutacije ansambla. Posledice su dve:
#
#   1. rasipanje po seedovima potceneno samo za TabPFN -- fali cela komponenta varijanse;
#   2. tacka uslovljena jednim proizvoljnim izvlacenjem permutacija, a bas se tu meri
#      vincentizacija (`findings/h1/vincentization.md`: agregacija spusta implicirano xi za
#      0,17 pri xi=0,9).
#
# `predict_q` iz te skripte koristi jos pet merenja, pa je pogodjeno 278 TabPFN redova
# u 9 CSV-ova. Ovde se ponovo mere svi.
#
# REZIM
# -----
# Rezultati idu u NOVE fajlove sa sufiksom `-sejanje`. Stari se ne diraju, da se razlika
# moze izmeriti, a ne samo tvrditi. Poredjenje je na kraju.
#
# n_est=4 je prioritet: pri n_est=1 TabPFN i dalje bira jednu pretprocesnu konfiguraciju
# po `random_state`, ali nema usrednjavanja preko clanova, pa je efekat mnogo manji.
# Zato n_est=4 ide prvo -- ako se stane na pola, stalo se na pravom mestu.
#
# TabPFN i TabICL su u SISTEMSKOM python-u, ne u venv-u (venv-* nose torch za TabDPT,
# TabFM i EXAONE). Zato se ovde ne postavlja PY na venv.
#
# POKRETANJE
# ----------
#     sh run_tabpfn_reseed.sh
#
# Traje satima. Pusti tek kad EXAONE stream-ovi stanu, ili prihvati deljenje CPU-a.
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}   # another environment: PY=... sh run/...
M=TabPFN-V3

# ---- prioritet 1: naslovne brojke (n_est=4, 20 seedova) ---------------------
# `shape_of_x_20.csv` nosi udeo uhvacenog OBLIKA = 19%
MODELS=$M N_EST=4 SEEDS=20 OUTPUT=shape_of_x_tabpfn_reseed_20.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py

# `scale_of_x.csv` nosi udeo uhvacene SKALE; disocijacija je odnos to dvoje
MODELS=$M N_EST=4 SEEDS=20 OUTPUT=scale_of_x_tabpfn_reseed_20.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py

# ---- prioritet 2: n_est=1 tabela iz findings/NALAZI.md §3.1 --------------------------
MODELS=$M N_EST=1 SEEDS=20 OUTPUT=shape_of_x_tabpfn_reseed_nest1.csv \
  $PY -u experiments/h1_shape_vs_scale/shape_of_x.py

MODELS=$M N_EST=1 SEEDS=20 OUTPUT=scale_of_x_tabpfn_reseed_nest1.csv \
  $PY -u experiments/h1_shape_vs_scale/scale_of_x.py

# ---- prioritet 3: ostala merenja koja zovu `predict_q` ----------------------
MODELS=$M N_EST=1 SEEDS=5 OUTPUT=sample_complexity_tabpfn_reseed.csv \
  $PY -u experiments/h1_shape_vs_scale/sample_complexity.py

MODELS=$M SEEDS=5 OUTPUT=tail_families_tabpfn_reseed.csv \
  $PY -u experiments/h1_shape_vs_scale/tail_families.py

MODELS=$M SEEDS=10 OUTPUT1=xi_residual_tabpfn_reseed_part1.csv \
  OUTPUT2=xi_residual_tabpfn_reseed_part2.csv \
  $PY -u experiments/h1_shape_vs_scale/xi_residual.py

MODELS=$M N_EST=1 SEEDS=3 OUTPUT=dissociation_real_tabpfn_reseed.csv \
  $PY -u experiments/h1_shape_vs_scale/dissociation_real.py

# ---- koliko je ispravka pomerila brojke -------------------------------------
$PY -u experiments/h1_shape_vs_scale/compare_reseeding.py
