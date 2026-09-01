# -*- coding: utf-8 -*-
"""
Empirijska pokrivenost predvidjenih kvantila: broj koji korisnik zaista dobije.

RUPA KOJA SE OVDE ZATVARA
-------------------------
Ceo rad polazi od toga da se prediktivna raspodela ovih modela cita neposredno,
za intervale i za procenu rizika. A do sada nijedno merenje u repozitorijumu ne
odgovara na najdirektnije pitanje koje iz toga sledi:

    koliki udeo stvarnih ishoda premasi predvidjeni Q(0,99)?

Sve postojece mere su posredne: implicirani indeks repa iz odnosa kvantila,
pinball, twCRPS. Sve tri kazu koliko je raspodela dobra u nekoj metrici, ali
nijedna ne kaze da li interval drzi nominalni nivo. Pokrivenost je jedina mera
koja se moze procitati bez ijednog estimatora izmedju: nominalno 1%, izmereno
toliko i toliko.

Uz to je ovo jedina tabela u kojoj se prvi i drugi deo rada srecu na istom
broju: ista pokrivenost se meri i pod dozom, pa se cena jedne poluge u kontekstu
izrazava u jedinici koja korisnika zanima, a ne u procentu pomeraja kvantila.

DIZAJN
------
Po skupu, modelu i seedu: fit na N_FIT redova, predikcija na N_TEST, i za svaki
nivo alpha se broji koliko test ishoda ga premasi.

    doza = 0    cist kontekst
    doza > 0    isti kontekst plus jedan red sa y = doza * max(y_fit),
                sto je isti zahvat kao u `h2_leverage/influence.py`

Belezi se i `sd_shift`, izmerena jacina zahvata, pa se pokrivenost moze crtati
protiv nje, na istoj osi na kojoj je merena i prevalencija poluge
(`prevalence_data.csv`, najgori od 99 skupova: 19,9).

KAKO CITATI
-----------
  cov99 ~ 0,99                    interval drzi nominalni nivo
  cov99 < 0,99                    previse ishoda iznad praga, rep potcenjen
  cov99 pada sa dozom             jedna poluga u kontekstu kosta pokrivenost
  n_above999 jednocifreno         nivo 0,999 na ovom n nije merljiv, i to se vidi

`n_above` se belezi uz svaku pokrivenost bas zato: na 6000 test redova nivo
0,999 ocekuje sest prekoracenja, pa je razlika izmedju 4 i 8 sum, a ne nalaz.

POKRETANJE
----------
    python -u experiments/h1_shape_vs_scale/coverage.py
    MODELS=TabICLv2 DATASETS=freMTPL2sev DOSES=0,100 python -u coverage.py
"""
import os
import time

import numpy as np
import pandas as pd

from common import append, datasets, generator, metrics, models, paths, quiet

quiet.silence()

LEVELS = [0.5, 0.9, 0.99, 0.999]
LEVEL_NAME = {0.5: "50", 0.9: "90", 0.99: "99", 0.999: "999"}
N_FIT, N_TEST = 3000, 6000
N_EST = int(os.environ.get("N_EST", "1"))
SEEDOVA = int(os.environ.get("SEEDS", "3"))
DOZE = [float(v) for v in os.environ.get("DOSES", "0,100").split(",")]
MODELI = os.environ.get("MODELS", "TabICLv2,TabPFN-V3,GBM,XGB").split(",")
# Isti devet skupova na kojima stoji disocijacija na stvarnim podacima, plus
# freMTPL2sev, jedini skup u pregledu koji ima i polugu i najtezi rep.
PODRAZUMEVANI = ["freMTPL2sev", "OnlineNewsPopularity", "diamonds",
                 "particulate-matter-ukair-2017", "Buzzinsocialmedia_Twitter",
                 "CPS1988", "218_house_8L", "superconduct", "houses", "house_16H"]
SKUPOVI = os.environ.get("DATASETS", ",".join(PODRAZUMEVANI)).split(",")
# "sint" kao ime skupa znaci generator sa poznatim xi, radi kontrole na podacima
# gde se prava pokrivenost zna po konstrukciji.
XI_SINT = float(os.environ.get("XI", "0.7"))
OUT = os.environ.get("OUTPUT", "coverage.csv")

KOLONE = (["dataset", "model", "seed", "dose", "n_est", "n_fit", "n_test", "sd_shift"]
          + [f"cov{LEVEL_NAME[a]}" for a in LEVELS]
          + [f"n_above{LEVEL_NAME[a]}" for a in LEVELS]
          + [f"pb{LEVEL_NAME[a]}" for a in LEVELS]
          + ["seconds", "reason"])
KEY = ["dataset", "model", "seed", "dose", "n_est"]


def uzorak(skup, seed, kes):
    """Fit i test deo jednog skupa; `sint` je generator sa poznatim xi."""
    if skup == "sint":
        rng = np.random.default_rng(seed)
        f = generator.gpd(N_FIT, rng, xi=XI_SINT, clip=True)
        t = generator.gpd(N_TEST, rng, xi=XI_SINT, clip=True)
        return f.X, f.y, t.X, t.y
    if skup not in kes:
        kes[skup] = datasets.load(skup, kes["_ids"], positive_only=True)
    X, y = kes[skup]
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"premalo redova ({len(y)})")
    idx = np.random.default_rng(seed).permutation(len(y))[: N_FIT + N_TEST]
    fi, ti = idx[:N_FIT], idx[N_FIT:]
    return X[fi], y[fi], X[ti], y[ti]


def jedan(skup, ime, doza, seed, kes):
    Xf, yf, Xte, yte = uzorak(skup, seed, kes)

    sd_shift = 1.0
    if doza > 0:
        # Isti zahvat kao u h2_leverage/influence.py: jedan red, y = doza * max.
        j = int(np.random.default_rng(seed).integers(len(yf)))
        y0 = doza * float(yf.max())
        sd_shift = metrics.sd_shift(np.append(yf, y0))
        Xf = np.vstack([Xf, Xf[[j]]])
        yf = np.concatenate([yf, [y0]])

    q = models.quantiles(ime, Xf, yf, Xte, seed=seed, levels=LEVELS, n_est=N_EST)

    r = dict(sd_shift=sd_shift, reason="")
    for i, a in enumerate(LEVELS):
        above = int((yte > q[:, i]).sum())
        r[f"cov{LEVEL_NAME[a]}"] = 1.0 - above / len(yte)
        r[f"n_above{LEVEL_NAME[a]}"] = above
        r[f"pb{LEVEL_NAME[a]}"] = metrics.pinball_at(q, yte, a, LEVELS)
    return r


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    kes = {"_ids": paths.load_json("sb_openml_ids.json")}
    t0 = time.time()
    for skup in SKUPOVI:
        for ime in MODELI:
            for doza in DOZE:
                for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                    k = dict(dataset=skup, model=ime, seed=seed, dose=doza, n_est=N_EST)
                    if append.key(k, KEY) in gotovi:
                        continue
                    t1 = time.time()
                    try:
                        r = jedan(skup, ime, doza, seed, kes)
                    except Exception as e:
                        r = dict(reason=f"{type(e).__name__}: {e}"[:110])
                    r.update(k, n_fit=N_FIT, n_test=N_TEST,
                             seconds=round(time.time() - t1, 1))
                    append.write(OUT, r, KOLONE)
                    poruka = r.get("reason") or (
                        f"cov 0.9 {r['cov90']:.3f} | 0.99 {r['cov99']:.4f} "
                        f"({r['n_above99']} iznad) | 0.999 {r['cov999']:.4f} "
                        f"({r['n_above999']} iznad)")
                    print(f"  {skup:32s} {ime:10s} d={doza:<5g} s={seed}  {poruka}"
                          f"  [{r['seconds']}s]", flush=True)
    report(t0)


def report(t0=None):
    d = pd.read_csv(paths.result(OUT))
    d = d[d.reason.isna() | (d.reason.astype(str).str.strip() == "")]
    if d.empty:
        return
    print("\n=== pokrivenost na cistom kontekstu (doza 0), nominalno u zagradi ===")
    for a in LEVELS:
        piv = d[d.dose == 0].pivot_table(index="dataset", columns="model",
                                         values=f"cov{LEVEL_NAME[a]}", aggfunc="median")
        print(f"\n  alpha = {a}")
        print("    " + piv.round(4).to_string().replace("\n", "\n    "))
    if (d.dose > 0).any():
        print("\n=== cena jedne poluge u kontekstu: pokrivenost na 0,99 ===")
        piv = d.pivot_table(index=["dataset", "model"], columns="dose",
                            values="cov99", aggfunc="median")
        print(piv.round(4).to_string())
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
