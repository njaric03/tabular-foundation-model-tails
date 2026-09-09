# -*- coding: utf-8 -*-
"""
Disocijacija: prate li modeli SKALU s(x) dok OBLIK repa xi(x) skupljaju?

ZASTO
-----
`findings/h1/oracle.md` je pokazao da modeli hvataju 19-45% prave promene xi(x), i da to
nije problem podataka (orakl hvata 98%). Ostaje pitanje da li je taj deficit specifican
za OBLIK, ili modeli generalno slabo prate uslovnu strukturu.

Ako skalu prate skoro savrseno a oblik skupljaju, tvrdnja se izostrava iz
"modeli slabo prate rep" u:

    tabelarni fundacioni modeli su lokacijsko-skalni pratitelji;
    uslovni OBLIK repa im je skoro zamrznut.

To je jaca i preciznija recenica, i naslanja se na arhitekturu: kolonsko utapanje kodira
rang u raspodeli kolone (vidi `thesis/notes/pfn_theory.md`, odeljak 3), pa je nivo lako prenosiv,
a oblik nije.

MERA
----
Isti generator kao `shape_of_x.py`. Kljucno svojstvo tog generatora:

    s(x) zavisi od x0,x1,x2  (tezine 1.0, -0.7, 0.5)
    xi(x) zavisi od x4       (tezina u skali je NULA)

pa su nivo i oblik razdvojeni po konstrukciji.

  OBLIK  tercili po x4,      meri se implicirano xi iz Q(0.99)/Q(0.9)   [vec urajeno]
  SKALA  tercili po X@W,     meri se log-odnos predvidjene medijane

Za oba se racuna isti broj: koliki UDEO prave promene model uhvati.
Prava promena skale kroz tercile se zna analiticki, jer se s(x) zna.

Rezultati u `scale_of_x.csv`.

POKRETANJE
----------
    SEEDS=20 python -u scale_of_x.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd

from common import generator, metrics, models, paths, quiet

quiet.silence()

QS = [0.5, 0.9, 0.99]
N_EST = int(os.environ.get("N_EST", "4"))
N_TRAIN, N_TEST = 2000, 900

# generator i predikcije se uzimaju iz postojece skripte, da se ne dupliraju

BROJ_SEEDOVA = int(os.environ.get("SEEDS", "20"))
SEEDS = [7000 + 1000 * i for i in range(BROJ_SEEDOVA)]
MODELI = models.parse_list(os.environ.get("MODELS", "GBM,TabICLv2,TabPFN-V3"))
N_TERCILA = 3
OUT = os.environ.get("OUTPUT", "scale_of_x.csv")


def prava_medijana(s, xi):
    """Medijana GPD(xi, s) je s * ((0.5)^(-xi) - 1) / xi."""
    return s * metrics.gpd_quantile(0.5, xi)


def main():
    rows, t0 = [], time.time()
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        Xtr, ytr, _, _ = generator.gpd(N_TRAIN, rng, xi=generator.XI_OF_X)
        Xte, _, s_te, xi_te = generator.gpd(N_TEST, rng, xi=generator.XI_OF_X)

        # tercili po LINEARNOM PREDIKTORU SKALE, ne po x4
        lin = Xte @ generator.W / np.linalg.norm(generator.W)
        ivice = np.quantile(lin, np.linspace(0, 1, N_TERCILA + 1))
        terc = np.clip(np.digitize(lin, ivice[1:-1]), 0, N_TERCILA - 1)

        median_true = prava_medijana(s_te, xi_te)

        for ime in MODELI:
            q = models.quantiles(ime, Xtr, ytr, Xte, seed=seed, levels=QS, n_est=N_EST)
            median_model = q[:, QS.index(0.5)]
            for t in range(N_TERCILA):
                sel = terc == t
                rows.append(dict(
                    model=ime, seed=seed, tercile=t + 1, n_est=N_EST,
                    median_true=float(np.median(median_true[sel])),
                    median_model=float(np.median(median_model[sel])),
                    xi_mean=float(xi_te[sel].mean()),
                ))
            v = [f"{rows[-N_TERCILA + i]['median_model']:.2f}" for i in range(N_TERCILA)]
            p = [f"{rows[-N_TERCILA + i]['median_true']:.2f}" for i in range(N_TERCILA)]
            print(f"  {ime:10s} seed={seed}  medijana model {' / '.join(v)}"
                  f"   pravo {' / '.join(p)}  [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(paths.result(OUT), index=False)

    print("\n=== UDEO PRAVE PROMENE SKALE KOJI MODEL UHVATI ===")
    print("   (log-odnos medijane izmedju 3. i 1. tercila)\n")
    p = df.pivot_table(index=["model", "seed"], columns="tercile",
                       values=["median_model", "median_true"])
    nag_m = np.log(p[("median_model", 3)] / p[("median_model", 1)])
    nag_p = np.log(p[("median_true", 3)] / p[("median_true", 1)])
    udeo = (nag_m / nag_p).groupby(level=0)
    print(f"{'model':<12}{'nagib model':>13}{'nagib pravi':>13}{'UDEO':>9}{'+-':>8}")
    for ime in MODELI:
        m = nag_m.xs(ime, level=0)
        pr = nag_p.xs(ime, level=0)
        u = udeo.get_group(ime)
        print(f"{ime:<12}{m.mean():>13.3f}{pr.mean():>13.3f}{u.mean():>8.0%}{u.std():>8.2f}")
    # The shape share is the other half of the dissociation and depends on the
    # ensemble size, so the two must not be mixed. It is computed from
    # shape_of_x_*.csv by analysis/h1_shape_vs_scale.ipynb rather than pasted
    # here, so a re-run cannot print a stale reference beside a fresh number.
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
