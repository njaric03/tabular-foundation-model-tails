# -*- coding: utf-8 -*-
"""
Odgovor SREDINE na jednu kontaminiranu tacku -- EXAONE, uporedivo sa TabFM-om.

ZASTO
-----
`influence.py` meri Q(0,5), a `findings/h2/preprocessing_asymmetry.md` za TabFM navodi
**+1572% na SREDINI**. To nisu iste velicine: pod teskim repom je sredina daleko iznad
medijane i pomera se drugacije. Da bi EXAONE bio uporediv sa tom brojkom, mora se meriti
istom merom i u istom rezimu.

Ovaj fajl je verzija `architecture_tabfm.py` sa EXAONE-om umesto TabFM-a. Generator,
doza, mera uticaja i obe referentne sredine su preuzeti IZ TOG FAJLA (import, ne kopija), pa
se brojke porede bez ograde.

REZIM
-----
Podrazumevano N_TRAIN=1000, N_TEST=200, N_EST=1 -- to su TabFM-ovi parametri, izabrani zbog
1,6 mlrd parametara na CPU-u. EXAONE ima 21,1M i podneo bi vise, ali se ovde namerno ostaje
u istom rezimu: cilj je poredjenje, ne najbolji moguci broj za EXAONE. Sire pokretanje ide
preko `N_TRAIN`/`N_TEST`/`N_EST`.

MERI SE
-------
  uticaj_sredina   medijana (mod2 - mod) / |mod|, gde je mod2 fit sa jednim dodatnim redom
                   (x0 = centar, y0 = 100 * max(y_train))
  odnos_prava      mean(model) / mean(prava sredina GPD)                = s/(1-xi)
  odnos_odsec      mean(model) / mean(E[Y | Y <= max(y_train)])

`odnos_odsec` je kontrola za odseceni nosac: model koji ne moze da predvidi iznad viđenog
maksimuma bi trebalo da stoji oko 1,0 prema odsecenoj sredini, a ispod 1,0 prema pravoj.

POKRETANJE
----------
    python -u mean_exaone.py
    XI=0.9 SEEDS=5 N_TRAIN=2000 python -u mean_exaone.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd

from common import append, generator, metrics, models, paths, quiet

quiet.silence()

from common.adapters import exaone as E

XI_LISTA = [float(v) for v in os.environ.get("XI", "0.5,0.7,0.9").split(",")]
SEEDOVA = int(os.environ.get("SEEDS", "3"))
N_TRAIN = int(os.environ.get("N_TRAIN", "1000"))
N_TEST = int(os.environ.get("N_TEST", "200"))
N_EST = int(os.environ.get("N_EST", "1"))
DOZA = float(os.environ.get("DOSE", "100"))
OUT = os.environ.get("OUTPUT", "mean_exaone.csv")
KLJUC = ["xi", "seed", "n_train", "n_est", "dose"]
KOLONE = ["xi", "model", "seed", "n_train", "n_est", "dose",
          # sd_shift je izmerena jacina intervencije; `dose` je samo nominalna, a
          # max(y_train) je i sam slucajan, pa ista doza nije isti tretman po seedu.
          "sd_shift",
          "ratio_true", "ratio_truncated", "mean_influence", "seconds", "reason"]


def sredina(Xa, ya, Xte, seed):
    return models.mean("EXAONE", Xa, ya, Xte, seed=seed, n_est=N_EST)


def main():
    gotovi = append.done(OUT, KLJUC)
    if gotovi:
        print(f"nastavljam, vec uradjeno {len(gotovi)}", flush=True)

    t0 = time.time()
    for xi in XI_LISTA:
        for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
            k = dict(xi=xi, seed=seed, n_train=N_TRAIN, n_est=N_EST, dose=DOZA)
            if append.key(k, KLJUC) in gotovi:
                continue
            t1 = time.time()
            try:
                rng = np.random.default_rng(seed)
                tr = generator.gpd(N_TRAIN, rng, xi=xi)
                Xtr, ytr = tr.X, tr.y
                te = generator.gpd(N_TEST, rng, xi=xi)
                Xte, s_te = te.X, te.s
                T = float(ytr.max())
                prava = s_te / (1 - xi)
                odsec = s_te * metrics.truncated_mean_std(T / s_te, xi)

                mod = sredina(Xtr, ytr, Xte, seed)
                x0 = np.zeros((1, Xtr.shape[1]))
                mod2 = sredina(np.vstack([Xtr, x0]),
                               np.concatenate([ytr, [DOZA * T]]), Xte, seed)
                uticaj = float(np.median((mod2 - mod) / np.maximum(np.abs(mod), 1e-9)))

                r = dict(ratio_true=float(np.mean(mod) / np.mean(prava)),
                         ratio_truncated=float(np.mean(mod) / np.mean(odsec)),
                         mean_influence=uticaj,
                         sd_shift=metrics.sd_shift(np.append(ytr, DOZA * T)),
                         reason="")
            except Exception as e:
                r = dict(reason=f"{type(e).__name__}: {e}"[:110])
            r.update(k, model="EXAONE",
                     seconds=round(time.time() - t1, 1))
            append.write(OUT, r, KOLONE)
            poruka = r.get("reason") or (
                f"model/prava {r['ratio_true']:.3f}  model/odsec {r['ratio_truncated']:.3f}  "
                f"uticaj sredine {r['mean_influence']:+.1%}")
            print(f"  xi={xi} seed={seed}  {poruka}  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT))
    d = d[d.reason.isna() | (d.reason == "")]
    print("\n=== EXAONE, sredina, medijana po seedovima ===")
    print(d.groupby("xi")[["ratio_true", "ratio_truncated", "mean_influence"]]
          .median().round(3).to_string())
    print("\nZa poredjenje (`findings/h2/preprocessing_asymmetry.md`, ista mera, doza 100x):")
    print("  TabFM  +1572% (po seedu +280% / +1572% / +7010% pri xi=0,9)")
    # Cross-model comparison lives in analysis/h2_leverage.ipynb, which reads
    # the result CSVs, instead of being pasted here as constants.
    print("  GBM      +0,9%")
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
