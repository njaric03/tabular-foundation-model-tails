# -*- coding: utf-8 -*-
"""
Preziva li disocijacija nivo/oblik druge familije repa?

ZASTO
-----
`findings/h1/dissociation.md` je merio na JEDNOM generatoru: GPD sa log-linearnom skalom.
Mera je takodje GPD-centricna (implicirano xi se vraca iz odnosa kvantila preko GPD
formule). Ostaje ozbiljna mogucnost da efekat postoji samo kad su podaci bas GPD --
sto bi bilo kruzno. To je stavka 3 iz "sta bi oborilo temu" u `thesis/notes/evt_graft_plan.md`.

Ovde se ista mera pusta preko cetiri familije sa istim rasponom tezine repa.

FAMILIJE
--------
Sve imaju indeks repa xi(x) = 0.15 + 0.75 * Phi(x4) i istu skalu s(x), pa su
direktno uporedive. Razlikuje se OBLIK tela raspodele, ne rep.

    gpd       Y = s * ((1-U)^(-xi) - 1) / xi              (referenca, dosadasnji)
    frechet   Y = s * (-ln U)^(-xi)                        xi = 1/alpha
    burr      Y = s * ((1-U)^(-xi*c) - 1)^(1/c),  c = 2    xi = 1/(c*k)
    studentt  Y = s * |T_nu|,  nu = 1/xi                    xi = 1/nu

VAZNO ZA POSTENJE MERE
----------------------
Referentno "pravo" xi se NE uzima iz parametra familije, nego se racuna ISTOM merom
kojom se meri model: iz pravih uslovnih kvantila Q(0.99)/Q(0.9) te familije, provucenih
kroz istu `implied_xi` funkciju. Time GPD pretpostavka izlazi iz poredjenja -- meri se
koliko model prati PROMENU te mere, kakva god ona bila.

Rezultati u `tail_families.csv`.

POKRETANJE
----------
    python -u tail_families.py
    FAMILIJE=frechet,burr SEEDOVA=3 python -u tail_families.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from common import append, generator, metrics, models, paths, quiet

quiet.silence()

QS = [0.5, 0.9, 0.99]
HI, LO = 0.99, 0.9
N_EST = int(os.environ.get("N_EST", "4"))
N_TRAIN, N_TEST = 2000, 900


FAMILIJE = os.environ.get("FAMILIJE", "gpd,frechet,burr,studentt").split(",")
SEEDOVA = int(os.environ.get("SEEDS", "5"))
MODELI = models.parse_list(os.environ.get("MODELS", "GBM,TabICLv2,TabPFN-V3"))
SEEDS = [7000 + 1000 * i for i in range(SEEDOVA)]
N_TERCILA = 3
BURR_C = 2.0
# IZLAZ je podesiv kao i u ostalim skriptama: bez toga svaki paralelni stream pise u
# isti fajl, sto je vec jednom pokvarilo merenje.
OUT = os.environ.get("OUTPUT", "tail_families.csv")
KOLONE = ["n_est", "family", "model", "seed", "shape_share", "scale_share",
          "xi_mod_t1", "xi_mod_t3", "xi_ref_t1", "xi_ref_t3", "seconds", "reason"]
KEY = ["family", "model", "seed", "n_est"]


def kvantil(familija, a, xi, s):
    """Pravi uslovni kvantil nivoa `a`, po tacki (vektorski po xi i s)."""
    a = np.asarray(a, dtype=float)
    if familija == "gpd":
        core = ((1 - a) ** (-xi) - 1) / xi
    elif familija == "frechet":
        core = (-np.log(a)) ** (-xi)
    elif familija == "burr":
        core = ((1 - a) ** (-xi * BURR_C) - 1) ** (1.0 / BURR_C)
    elif familija == "studentt":
        core = student_t.ppf(0.5 + a / 2.0, df=1.0 / xi)   # |T_nu|
    else:
        raise ValueError(familija)
    return s * core


def uzorkuj(familija, U, xi, s):
    if familija == "gpd":
        return s * ((1 - U) ** (-xi) - 1) / xi
    if familija == "frechet":
        return s * (-np.log(U)) ** (-xi)
    if familija == "burr":
        return s * ((1 - U) ** (-xi * BURR_C) - 1) ** (1.0 / BURR_C)
    if familija == "studentt":
        return s * np.abs(student_t.ppf(U, df=1.0 / xi))
    raise ValueError(familija)


def podaci(familija, n, rng):
    """Isti dizajn kao glavni generator, druga familija repa nad njim.

    X, s(x) i xi(x) dolaze iz `generator.covariates`, pa vise ne postoje kao
    kopija; ovde se crta samo U i primenjuje kvantilna funkcija familije. Redosled
    izvlacenja je isti kao ranije (prvo X, pa U), pa se stari rezultati
    reprodukuju.
    """
    c = generator.covariates(n, rng, xi=generator.XI_OF_X)
    U = np.clip(rng.random(n), 1e-9, 1 - 1e-9)
    return c.X, uzorkuj(familija, U, c.xi, c.s), c.s, c.xi


def jedan(familija, ime, seed):
    rng = np.random.default_rng(seed)
    Xtr, ytr, _, _ = podaci(familija, N_TRAIN, rng)
    Xte, _, s_te, xi_te = podaci(familija, N_TEST, rng)
    q = models.quantiles(ime, Xtr, ytr, Xte, seed=seed, levels=QS, n_est=N_EST)

    # referenca: ista mera, ali na PRAVIM kvantilima te familije
    ref_hi = kvantil(familija, HI, xi_te, s_te)
    ref_lo = kvantil(familija, LO, xi_te, s_te)

    iv = np.quantile(Xte[:, 4], np.linspace(0, 1, N_TERCILA + 1))
    t_ob = np.clip(np.digitize(Xte[:, 4], iv[1:-1]), 0, N_TERCILA - 1)
    xi_mod, xi_ref = [], []
    for t in range(N_TERCILA):
        sel = t_ob == t
        num, den = q[sel, QS.index(HI)], q[sel, QS.index(LO)]
        ok = den > 1e-9
        xi_mod.append(metrics.xi_from_ratio(float(np.median(num[ok] / den[ok]))) if ok.any() else np.nan)
        xi_ref.append(metrics.xi_from_ratio(float(np.median(ref_hi[sel] / ref_lo[sel]))))
    udeo_ob = (xi_mod[2] - xi_mod[0]) / (xi_ref[2] - xi_ref[0])

    lin = Xte @ generator.W / np.linalg.norm(generator.W)
    iv = np.quantile(lin, np.linspace(0, 1, N_TERCILA + 1))
    t_sk = np.clip(np.digitize(lin, iv[1:-1]), 0, N_TERCILA - 1)
    med_pr = kvantil(familija, 0.5, xi_te, s_te)
    med_mo = q[:, QS.index(0.5)]
    a_m, b_m = np.median(med_mo[t_sk == 0]), np.median(med_mo[t_sk == N_TERCILA - 1])
    a_p, b_p = np.median(med_pr[t_sk == 0]), np.median(med_pr[t_sk == N_TERCILA - 1])
    udeo_sk = np.log(b_m / a_m) / np.log(b_p / a_p) if a_m > 0 and b_m > 0 else np.nan

    return dict(shape_share=udeo_ob, scale_share=udeo_sk,
                xi_mod_t1=xi_mod[0], xi_mod_t3=xi_mod[2],
                xi_ref_t1=xi_ref[0], xi_ref_t3=xi_ref[2], reason="")


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    t0 = time.time()
    for fam in FAMILIJE:
        for ime in MODELI:
            for seed in SEEDS:
                if append.key(dict(family=fam, model=ime, seed=seed), KEY[:3]) in gotovi:
                    continue
                t1 = time.time()
                try:
                    r = jedan(fam, ime, seed)
                except Exception as e:
                    r = dict(reason=f"{type(e).__name__}: {e}"[:100])
                r.update(family=fam, model=ime, seed=seed,
                         seconds=round(time.time() - t1, 1))
                append.write(OUT, r, KOLONE)
                if r.get("reason"):
                    print(f"  {fam:9s} {ime:10s} seed={seed} -- {r['reason']}", flush=True)
                else:
                    print(f"  {fam:9s} {ime:10s} seed={seed}  oblik {r['shape_share']:>6.0%}"
                          f"  skala {r['scale_share']:>6.0%}  "
                          f"(ref xi {r['xi_ref_t1']:.2f}->{r['xi_ref_t3']:.2f})"
                          f"  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); # Filter on whether the MEASUREMENT succeeded, not on notna() of the estimate:
    # xi_from_ratio returns NaN exactly on the heaviest tails (outside the brentq
    # bracket), so filtering on the estimate drops heavy tails preferentially.
    d = d[d.reason.fillna("") == ""] if "reason" in d.columns else d[d.shape_share.notna()]
    for kol, lab in [("shape_share", "OBLIK"), ("scale_share", "SKALA")]:
        print(f"\n=== {lab}: udeo prave promene, po familiji ===")
        p = d.pivot_table(index="family", columns="model", values=kol, aggfunc="mean")
        print(p.map(lambda v: f"{v:.0%}").to_string())
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
