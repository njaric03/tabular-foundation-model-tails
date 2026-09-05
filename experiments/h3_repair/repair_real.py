# -*- coding: utf-8 -*-
"""
Winsorizacija na STVARNIM skupovima, sa i bez kvarnog zapisa.

ZASTO OVAKAV DIZAJN
-------------------
`influence_real.py` je pokazao da JEDNA STVARNA ekstremna tacka ne pomera
predikciju: u stvarnim podacima uklanjanje najvece vrednosti menja sd samo 1.13-1.36x,
dok je za efekat potrebno 4x i vise. Teski rep sam po sebi ne pravi siljak.

Siljak od 10-100x max pravi GRESKA U PODACIMA -- pogresna jedinica, pokvaren senzor,
duplirana evidencija. Zato se ovde na stvarne skupove NAMERNO ubacuje jedan kvaran zapis,
i meri se:

  KORIST   koliko winsorizacija smanji stetu od tog zapisa
  CENA     koliko winsorizacija kosta na CISTOM stvarnom skupu (bez ubacivanja)

Time se testira preporuka onako kako bi se stvarno koristila: ne znas unapred da li
tvoja baza ima kvaran red, pa winsorizujes uvek.

POKRETANJE
----------
    python -u repair_real.py
    SKUPOVI=freMTPL2sev DOZE=10 SEEDOVA=3 python -u repair_real.py
"""
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd

from common import append, datasets, models, paths, quiet

quiet.silence()


SKUPOVI = os.environ.get(
    "DATASETS", "freMTPL2sev,OnlineNewsPopularity,Allstate_Claims_Severity").split(",")
MODELI = models.parse_list(os.environ.get("MODELS", "TabICLv2,TabPFN-V3,GBM"))
DOZE = [float(v) for v in os.environ.get("DOSES", "0,10,100").split(",")]  # 0 = cist skup
SEEDOVA = int(os.environ.get("SEEDS", "5"))
N_EST = int(os.environ.get("N_EST", "1"))
N_FIT, N_TEST = 2000, 1500
NIVOI = [0.5, 0.9, 0.99, 0.999]
IME_NIVOA = {0.5: "pb50", 0.9: "pb90", 0.99: "pb99", 0.999: "pb999"}
WINSOR_UDEO = 0.005
OUT = os.environ.get("OUTPUT", "repair_real.csv")
KOLONE = ["n_est", "dataset", "model", "dose", "variant", "seed",
          "pb50", "pb90", "pb99", "pb999", "sd_shift", "seconds", "reason"]
KEY = ["dataset", "model", "dose", "variant", "seed", "n_est"]


def kvantili(ime, Xtr, ytr, Xte, seed):
    return models.quantiles(ime, Xtr, ytr, Xte, seed=seed, levels=NIVOI, n_est=N_EST)


def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def jedan(skup, ime, doza, varijanta, seed, kes):
    if skup not in kes:
        kes[skup] = datasets.load(skup, kes["_ids"])
    X, y = kes[skup]
    ok = np.isfinite(y) & (y > 0)
    X, y = X[ok], y[ok]
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"premalo redova ({len(y)})")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    fi, ti = idx[:N_FIT], idx[N_FIT:]
    Xf, yf, Xte, yte = X[fi], y[fi], X[ti], y[ti]

    sd_pre = float(yf.std())
    if doza > 0:
        # kvaran zapis: kopija nasumicnog reda sa targetom uvecanim `doza` puta
        j = int(rng.integers(len(yf)))
        Xf = np.vstack([Xf, Xf[[j]]])
        yf = np.concatenate([yf, [doza * yf.max()]])
    sd_shift = float(yf.std() / sd_pre)

    if varijanta == "winsor":
        kapa = np.quantile(yf, 1 - WINSOR_UDEO)
        yf = np.minimum(yf, kapa)

    q = kvantili(ime, Xf, yf, Xte, seed)
    pb = {IME_NIVOA[a]: pinball(yte, q[:, i], a) for i, a in enumerate(NIVOI)}
    return dict(sd_shift=sd_shift, reason="", **pb)


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    kes = {"_ids": paths.load_json("sb_openml_ids.json")}
    t0 = time.time()
    for skup in SKUPOVI:
        for ime in MODELI:
            for doza in DOZE:
                for varijanta in ["raw", "winsor"]:
                    for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                        if append.key(dict(dataset=skup, model=ime, dose=doza, variant=varijanta, seed=seed), KEY[:5]) in gotovi:
                            continue
                        t1 = time.time()
                        try:
                            r = jedan(skup, ime, doza, varijanta, seed, kes)
                        except Exception as e:
                            r = dict(reason=f"{type(e).__name__}: {e}"[:110])
                        r.update(dataset=skup, model=ime, dose=doza, variant=varijanta,
                                 seed=seed, seconds=round(time.time() - t1, 1))
                        append.write(OUT, r, KOLONE)
                        poruka = r.get("reason") or (
                            f"pb99 {r['pb99']:.4g} pb999 {r['pb999']:.4g} "
                            f"| sd x{r['sd_shift']:.2f}")
                        print(f"  {skup[:20]:<21} {ime:10s} d={doza:>5} {varijanta:7s} "
                              f"s={seed} {poruka} [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); d = d[d.pb99.notna()]
    print("\n=== pomeraj sd po dozi (medijana) ===")
    print(d.groupby("dose").sd_shift.median().round(2).to_string())
    for kol in ["pb99", "pb999"]:
        print(f"\n=== {kol}: winsor naspram sirovo (negativno = winsor bolji) ===")
        for (skup, ime), g in d.groupby(["dataset", "model"]):
            red = []
            for doza in DOZE:
                gg = g[g.dose == doza]
                s = gg[gg.variant == "raw"][kol].median()
                w = gg[gg.variant == "winsor"][kol].median()
                if np.isfinite(s) and s > 0:
                    red.append(f"doza {doza:>5.0f}: {100*(w-s)/s:+7.1f}%")
            print(f"  {skup[:22]:<23} {ime:<11} " + "  ".join(red))
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
