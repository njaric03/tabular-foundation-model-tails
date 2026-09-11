# -*- coding: utf-8 -*-
"""
Kontrola dostiznosti za nalaz "nijedan model ne prati xi(x)".

RUPA KOJA SE OVDE ZATVARA
-------------------------
`shape_of_x.py` meri da modeli hvataju 0.05-0.15 od promene xi koja je
0.28 -> 0.78. Zakljucak je bio: modeli ne prilagodjavaju oblik repa kroz kovarijate.

Ali taj zakljucak ima nepokrivenu pretpostavku: da se xi(x) pri n_train = 2000
UOPSTE moze proceniti. Ako ne moze, onda nalaz nije o modelima nego o podacima --
ista zamka koja je vec jednom pukla na nivou 0.999 (vidi `ocena teme (u `archive/`)`).

STA SE MERI
-----------
Orakl estimator koji zna sve sto model ne zna, pa se pita koliko ON izvuce:

  ORAKL-S  zna pravu skalu s(x) i pravu pripadnost tercilu; normalizuje y sa s(x)
           i fituje GPD po tercilu. To je GORNJA GRANICA dostiznosti -- nijedan
           model ne moze bolje, jer model skalu mora da nauci.
  ORAKL-R  zna samo tercile; skalu procenjuje log-linearnom regresijom iz podataka.
           Realisticnija granica.
  EMPIRIJA isti odnos kvantila Q(0.99)/Q(0.9) koji se trazi od modela, ali racunat
           iz SIROVIH y u tercilu. Pokazuje koliko mesanje skale samo po sebi kvari
           meru.

Sve troje se pusta po mrezi n_train, da se vidi gde granica pada.

KAKO CITATI
-----------
  ORAKL-S vrati nagib blizak 0.50  -> xi(x) JE dostizno; nalaz o modelima stoji
  ORAKL-S vrati nagib blizu nule   -> nije dostizno; nalaz je o podacima, ne modelima
  ORAKL-S dobar tek na velikom n   -> granica je locirana, i to je sam po sebi nalaz

Rezultati u `oracle_xi_of_x.csv`.

POKRETANJE
----------
    python -u oracle_xi_of_x.py
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import genpareto, norm

from common import generator, metrics, paths, quiet

quiet.silence()

# Generator je `common/generator.py`, isti objekat koji dobijaju i modeli. Ovde je
# do sada stajala kopija W, XI_LO/XI_HI i same formule. Ta kopija je bila
# opasnija nego drugde: ovo je kontrola koja odlucuje da li je H1 tvrdnja o
# modelima ili o podacima, a to poredjenje vazi samo dok orakl uzorkuje iz
# ISTOG generatora. Parity cuva `tests/test_generator.py`.
N_TERCILA = 3
HI, LO = 0.99, 0.9

N_GRID = [500, 1000, 2000, 5000, 20000, 100000]
# The seed of repetition r depends only on (r, n), so a larger count keeps the
# first twenty draws identical and only adds new ones.
N_PONAVLJANJA = int(os.environ.get("N_REPEATS", "20"))
POT_FRAC = 0.10
OUT = os.environ.get("OUTPUT", "oracle_xi_of_x.csv")


def make_data(n, rng):
    p = generator.gpd(n, rng, xi=generator.XI_OF_X)
    return p.X, p.y, p.s, p.xi


def skala_regresijom(X, y):
    """Log-linearna procena skale: isti postupak kao `reziduali` u probe skriptama."""
    ok = y > 0
    D = np.c_[np.ones(ok.sum()), X[ok]]
    b, *_ = np.linalg.lstsq(D, np.log(y[ok]), rcond=None)
    s_hat = np.exp(np.c_[np.ones(len(X)), X] @ b)
    return np.clip(s_hat, 1e-9, None)


def jedno_ponavljanje(n, rng):
    X, y, s, xi = make_data(n, rng)
    edges = np.quantile(X[:, 4], np.linspace(0, 1, N_TERCILA + 1))
    terc = np.clip(np.digitize(X[:, 4], edges[1:-1]), 0, N_TERCILA - 1)
    s_hat = skala_regresijom(X, y)

    out = {}
    for t in range(N_TERCILA):
        sel = terc == t
        out[f"true_{t}"] = float(xi[sel].mean())
        out[f"oracle_s_{t}"] = metrics.gpd_mle(y[sel] / s[sel])
        out[f"oracle_r_{t}"] = metrics.gpd_mle(y[sel] / s_hat[sel])
        q_lo, q_hi = np.quantile(y[sel], [LO, HI])
        out[f"empirical_{t}"] = metrics.xi_from_ratio(q_hi / q_lo) if q_lo > 1e-12 else np.nan
    return out


def nagib(d, pref):
    a, b = d.get(f"{pref}_0"), d.get(f"{pref}_{N_TERCILA-1}")
    return (b - a) if (a is not None and b is not None
                       and np.isfinite(a) and np.isfinite(b)) else np.nan


def main():
    rows, t0 = [], time.time()
    for n in N_GRID:
        for r in range(N_PONAVLJANJA):
            # The seed depends on n as well as on the repetition; with a seed that
            # ignored n, the draws across the n grid were nested, so the scatter
            # reported across n was not a clean standard error.
            d = jedno_ponavljanje(n, np.random.default_rng(10_000 + 97 * r + 7 * n))
            d.update(n=n, repeat=r)
            rows.append(d)
        df = pd.DataFrame([x for x in rows if x["n"] == n])
        pr = np.nanmean([nagib(x, "true") for x in rows if x["n"] == n])
        print(f"  n={n:>7}  pravi nagib {pr:+.3f} | "
              f"ORAKL-S {np.nanmean([nagib(x,'oracle_s') for x in rows if x['n']==n]):+.3f} "
              f"| ORAKL-R {np.nanmean([nagib(x,'oracle_r') for x in rows if x['n']==n]):+.3f} "
              f"| EMPIRIJA {np.nanmean([nagib(x,'empirical') for x in rows if x['n']==n]):+.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(paths.result(OUT), index=False)

    print("\n=== NAGIB (xi u 3. tercilu minus xi u 1.), prosek i rasipanje ===")
    print(f"{'n':>8}{'pravi':>9}{'ORAKL-S':>18}{'ORAKL-R':>18}{'EMPIRIJA':>18}"
          f"{'udeo S>0.25':>13}")
    for n in N_GRID:
        sub = [x for x in rows if x["n"] == n]
        def st(p):
            v = np.array([nagib(x, p) for x in sub], dtype=float)
            # ddof=1: this is scatter over repetitions, matching every other script.
            return np.nanmean(v), np.nanstd(v, ddof=1)
        pr, _ = st("true")
        (ms, ss), (mr, sr), (me, se) = st("oracle_s"), st("oracle_r"), st("empirical")
        udeo = np.nanmean(np.array([nagib(x, "oracle_s") for x in sub], dtype=float) > 0.25)
        print(f"{n:>8}{pr:>9.3f}{ms:>11.3f}+-{ss:<5.2f}{mr:>11.3f}+-{sr:<5.2f}"
              f"{me:>11.3f}+-{se:<5.2f}{udeo:>12.0%}")

    print("\n=== xi PO TERCILU (prosek preko ponavljanja) ===")
    for n in N_GRID:
        sub = pd.DataFrame([x for x in rows if x["n"] == n])
        red = " | ".join(
            f"T{t+1}: pravo {sub[f'true_{t}'].mean():.2f} "
            f"S {sub[f'oracle_s_{t}'].mean():.2f} "
            f"R {sub[f'oracle_r_{t}'].mean():.2f} "
            f"E {sub[f'empirical_{t}'].mean():.2f}"
            for t in range(N_TERCILA))
        print(f"  n={n:>7}  {red}")
    print(f"\nukupno {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
