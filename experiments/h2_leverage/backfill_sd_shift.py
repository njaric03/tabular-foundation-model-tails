# -*- coding: utf-8 -*-
"""
Upisuje IZMERENU jacinu intervencije u vec izmerene redove, bez ijednog poziva modelu.

ZASTO
-----
Doza je do sada bila `y0 = doza * max(y_train)`, a `max(y_train)` je i sam
tesko-repna slucajna velicina. Ista nominalna doza zato nije ista intervencija
preko seedova:

    xi = 0.7, doza = 100  ->  pomeraj sd od 50.3 do 97.1 preko pet seedova

Dok to nije zapisano po redu, dve stvari se ne mogu uraditi. Prvo, ne moze se
razdvojiti rasipanje modela od rasipanja tretmana, a bas je rasipanje devet puta
obaralo nalaze (`findings/NALAZI.md` §6). Drugo, sintetika i stvarni podaci stoje
na razlicitim osama: prevalencija poluge je merena u pomeraju sd (`prevalence_data.csv`,
najgori od 99 skupova je 19.9), pa se dve polovine drugog dela nikada nisu videle
na jednoj slici.

STA RADI
--------
Za svaki red rekonstruise podatke iz (xi, seed, n_train) -- generator je
deterministican, pa se ne meri nista novo -- i upisuje:

    sd_shift      metrics.sd_shift(y_train + [y0]), ista funkcija kao u prevalenciji
    y0_over_max   y0 / max(y_train), da se stara doza i dalje cita
    dose_mode     "max", jer su svi postojeci redovi mereni tim pravilom

Nijedna izmerena vrednost se ne dira. Skripta je idempotentna: red koji vec ima
`sd_shift` se preskace.

POKRETANJE
----------
    python -u experiments/h2_leverage/backfill_sd_shift.py
    python -u experiments/h2_leverage/backfill_sd_shift.py --check   # samo ispis
"""
import sys

import numpy as np
import pandas as pd

from common import generator, metrics, paths, quiet

quiet.silence()

# (fajl, clip, da li ide dose_mode, parametri koji nedostaju kao kolone)
#
# clip mora da se poklopi sa skriptom koja je fajl proizvela, inace se
# rekonstruise drugi uzorak: influence.py zove gpd(..., clip=True), a
# mean_exaone.py bez clip-a.
#
# ZASTO SE `influence.csv` DOPUNJUJE I PARAMETRIMA. Taj fajl je stariji od
# trenutka kada su `position`, `n_train` i `n_est` postali kolone, pa se danas
# ne moze ni nastaviti: `append.done` odbija kljuc cija kolona ne postoji. Isti
# slucaj i isti postupak kao dopuna `n_est` od 29.8.2026 (`findings/NALAZI.md` §11):
# vrednost se upisuje samo kada je dokaziva. Ovde jeste, jer se SVI podrazumevani
# parametri skripte poklapaju sa sadrzajem fajla -- modeli TabICLv2/TabPFN-V3/GBM,
# xi 0.3 i 0.7, doze 1/3/10/100, pet seedova od 7000 -- sto znaci da je pokrenut
# `python -u influence.py` bez ijedne promenljive okruzenja. Nijedna izmerena
# vrednost se ne menja.
FILES = [
    ("h2_leverage/influence.csv", True, True,
     dict(position="centre", n_train=2000, n_est=1)),
    ("h2_leverage/influence_exaone.csv", True, True, {}),
    ("h2_leverage/mean_exaone.csv", False, False, {}),
]


def sd_shift_for(xi, seed, n_train, dose, clip):
    """Pomeraj sd koji je ta doza stvarno napravila, iz istog seeda."""
    rng = np.random.default_rng(int(seed))
    y = generator.gpd(int(n_train), rng, xi=float(xi), clip=clip).y
    y0 = float(dose) * float(y.max())
    return metrics.sd_shift(np.append(y, y0)), float(dose)


def main(check_only=False):
    for name, clip, with_mode, defaults in FILES:
        p = paths.result(name)
        if not p.exists():
            print(f"{name}: nema fajla, preskacem")
            continue
        d = pd.read_csv(p)
        for col, value in defaults.items():
            if col not in d.columns:
                d[col] = value
                print(f"{name}: kolona {col} dopisana kao {value!r} "
                      f"(podrazumevana vrednost skripte, vidi zaglavlje)")
        for col in ["sd_shift", "y0_over_max"] + (["dose_mode"] if with_mode else []):
            if col not in d.columns:
                d[col] = pd.NA

        filled = 0
        for i, r in d.iterrows():
            if pd.notna(d.at[i, "sd_shift"]):
                continue
            if any(pd.isna(r.get(c)) for c in ("xi", "seed", "n_train", "dose")):
                continue
            shift, ratio = sd_shift_for(r.xi, r.seed, r.n_train, r.dose, clip)
            d.at[i, "sd_shift"] = shift
            d.at[i, "y0_over_max"] = ratio
            filled += 1
        if with_mode:
            d.loc[d.dose_mode.isna(), "dose_mode"] = "max"

        print(f"{name}: {filled} redova dobilo sd_shift")
        g = d.dropna(subset=["sd_shift"]).groupby(["xi", "dose"]).sd_shift
        print(g.agg(["min", "median", "max"]).round(2).to_string())
        if not check_only:
            d.to_csv(p, index=False)
    if check_only:
        print("\n--check: nista nije upisano")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
