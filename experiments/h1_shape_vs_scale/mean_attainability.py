# -*- coding: utf-8 -*-
"""
Da li se prava uslovna sredina uopste MOZE proceniti iz 2000 redova?

Ovo je kontrola koja je nedostajala uz brojku „model vraca 29 odsto prave uslovne
ocekivane vrednosti". Za GPD sa xi >= 0.5 varijansa je beskonacna, pa je i sam
uzoracki prosek los procenjivac sredine: dominiraju ga retke ogromne vrednosti
kojih u 2000 redova najcesce nema.

Ako uzoracki prosek trening skupa i sam vraca 55 odsto prave sredine pri xi = 0.9,
onda model koji vrati 57 odsto nije podbacio, na granici je informacije u podacima.
Model koji vrati 24 odsto jeste podbacio, i to je onda pravi nalaz.

Mere se dve referentne vrednosti, obe kao odnos prema pravoj uslovnoj sredini:

  r_uzorak   prost prosek trening targeta
  r_orakl    dobro specificiran EVT procenjivac: log-linearna skala, GPD MLE nad
             gornjih 10% reziduala, pa mean = telo + rep pod procenjenim GPD-om

Rezultati u `mean_attainability.csv`. Traje oko pola minuta.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import genpareto

from common import paths

W = np.array([1.0, -0.7, 0.5, 0.0, 0.0])
XI_RUN = [0.0, 0.3, 0.5, 0.7, 0.9]
N_TRAIN = 2000
N_REP_SAMPLE = 400        # replikacija za uzoracki prosek (jeftino)
N_REP_ORACLE = 120        # replikacija za EVT orakl (skuplje)


def make_data(xi, n, rng):
    X = rng.normal(size=(n, 5))
    s = np.exp(0.6 * (X @ W) / np.linalg.norm(W))
    U = rng.random(n)
    y = s * (-np.log(1 - U)) if xi == 0 else s * ((1 - U) ** (-xi) - 1) / xi
    return X, y, s


def oracle_mean(X, y, frac=0.10):
    """Dobro specificiran EVT procenjivac uslovne sredine. Nan ako je xi_hat >= 1."""
    D = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(D, np.log(y), rcond=None)
    scale_hat = np.exp(D @ b)
    z = y / scale_hat
    u = np.quantile(z, 1 - frac)
    shape, _, sigma = genpareto.fit(z[z > u] - u, floc=0)
    if shape >= 0.999:
        return np.nan
    body = z[z <= u]
    m_z = (body.sum() + (z > u).sum() * (u + sigma / (1 - shape))) / len(z)
    return float((m_z * scale_hat).mean())


def main():
    rows, t0 = [], time.time()
    for xi in XI_RUN:
        samp, orac = [], []
        for k in range(N_REP_SAMPLE):
            rng = np.random.default_rng(10000 + k)
            X, y, s = make_data(xi, N_TRAIN, rng)
            true = float((s / (1 - xi)).mean())
            samp.append(y.mean() / true)
            if k < N_REP_ORACLE:
                m = oracle_mean(X, y)
                if np.isfinite(m):
                    orac.append(m / true)
        samp, orac = np.array(samp), np.array(orac)
        rows.append(dict(
            xi=xi,
            r_sample_median=float(np.median(samp)),
            r_sample_q10=float(np.quantile(samp, 0.10)),
            r_sample_q90=float(np.quantile(samp, 0.90)),
            share_below_060=float(np.mean(samp < 0.60)),
            r_oracle_median=float(np.median(orac)) if len(orac) else np.nan,
            n_rep=N_REP_SAMPLE,
        ))
        print(f"  xi={xi}  r_uzorak={rows[-1]['r_sample_median']:.3f}  "
              f"r_orakl={rows[-1]['r_oracle_median']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(paths.result("mean_attainability.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n=== DOSTIZNOST SREDINE IZ 2000 REDOVA (odnos prema pravoj) ===")
    print(df.round(3).to_string(index=False))
    print(f"\nukupno {time.time()-t0:.0f}s -> mean_attainability.csv")


if __name__ == "__main__":
    main()
