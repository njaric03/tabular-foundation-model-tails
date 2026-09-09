# -*- coding: utf-8 -*-
"""
Eksperiment 3: radi li popravka i na modelima koji daju samo SREDINU (TabDPT, TabFM)?

ZASTO SAMO WINSORIZACIJA
------------------------
`repair.py` testira log, winsor i rang na modelima sa kvantilima. Za modele
koji vracaju samo sredinu log i rang nisu cisti: inverzna transformacija sredine nije
egzaktna, jer E[g^{-1}(Z)] != g^{-1}(E[Z]) (Jensen). Winsorizacija tog problema nema --
ostaje u originalnim jedinicama, pa se nista ne vraca unazad.

MERI SE
-------
  KORIST   promena sredine kad se ubaci jedna tacka sa y = 100*max, sa i bez winsorizacije
  CENA     odnos prema odsecenoj sredini na CISTIM podacima, sa i bez

Popravka vredi samo ako smanji KORIST a ne pokvari CENU.

POKRETANJE
----------
    venv-tabdpt/Scripts/python.exe repair_mean.py
    MODEL=TabFM venv-tabfm/Scripts/python.exe repair_mean.py
"""
import os
import time

import numpy as np
import pandas as pd

from common import append, generator, metrics, models, paths, quiet

quiet.silence()

N_EST = int(os.environ.get("N_EST", "4"))

# W je u `common/generator.py`; ovde je stajala mrtva kopija.
MODEL = os.environ.get("MODEL", "TabDPT")
XI_LISTA = [float(v) for v in os.environ.get("XI", "0.7,0.9").split(",")]
SEEDOVA = int(os.environ.get("SEEDS", "5"))
N_TRAIN = int(os.environ.get("N_TRAIN", "2000"))
N_TEST = int(os.environ.get("N_TEST", "900"))
WINSOR_UDEO = 0.005
OUT = os.environ.get("OUTPUT", f"popravka-sredina-{MODEL}.csv")
KOLONE = ["n_train", "n_test", "xi", "model", "variant", "seed", "uticaj", "ratio_truncated", "seconds", "reason"]
KEY = ["xi", "variant", "seed", "n_train", "n_test"]

_ucitan = {}


def sredina(Xtr, ytr, Xte, seed):
    return models.mean(MODEL, Xtr, ytr, Xte, seed=seed, n_est=N_EST)


def obrada(varijanta, y):
    if varijanta == "raw":
        return y
    kapa = np.quantile(y, 1 - WINSOR_UDEO)
    return np.minimum(y, kapa)


def jedan(xi, varijanta, seed):
    rng = np.random.default_rng(seed)
    _p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
    Xtr, ytr, _ = _p.X, _p.y, _p.s
    _p = generator.gpd(N_TEST, rng, xi=xi, clip=True)
    Xte, _, s_te = _p.X, _p.y, _p.s
    T = float(ytr.max())
    odsec = s_te * metrics.truncated_mean_std(T / s_te, xi)

    # CENA: cisti podaci
    m0 = sredina(Xtr, obrada(varijanta, ytr), Xte, seed)
    odnos = float(np.mean(m0) / np.mean(odsec))

    # KORIST: jedna ubacena tacka
    x0 = np.zeros((1, Xtr.shape[1]))
    y_kontam = np.concatenate([ytr, [100 * T]])
    m1 = sredina(np.vstack([Xtr, x0]), obrada(varijanta, y_kontam), Xte, seed)
    uticaj = float(np.median((m1 - m0) / np.maximum(np.abs(m0), 1e-9)))

    return dict(uticaj=uticaj, ratio_truncated=odnos, reason="")


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    t0 = time.time()
    for xi in XI_LISTA:
        for varijanta in ["raw", "winsor"]:
            for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                if append.key(dict(xi=xi, variant=varijanta, seed=seed, n_train=N_TRAIN,
                                   n_test=N_TEST), KEY) in gotovi:
                    continue
                t1 = time.time()
                try:
                    r = jedan(xi, varijanta, seed)
                except Exception as e:
                    r = dict(reason=f"{type(e).__name__}: {e}"[:110])
                r.update(n_train=N_TRAIN, n_test=N_TEST,
                    xi=xi, model=MODEL, variant=varijanta, seed=seed,
                         seconds=round(time.time() - t1, 1))
                append.write(OUT, r, KOLONE)
                poruka = r.get("reason") or (
                    f"uticaj {r['uticaj']:+8.1%}  odnos_odsec {r['ratio_truncated']:.3f}")
                print(f"  xi={xi} {MODEL:8s} {varijanta:7s} s={seed}  {poruka}"
                      f"  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); d = d[d.uticaj.notna()]
    print(f"\n=== {MODEL} ===")
    print("KORIST: uticaj jedne ubacene tacke na sredinu (medijana)")
    print(d.pivot_table(index="xi", columns="variant", values="uticaj", aggfunc="median")
          .map(lambda v: f"{v:+.1%}").to_string())
    print("\nCENA: odnos prema odsecenoj sredini na CISTIM podacima (1.00 = tacno)")
    print(d.pivot_table(index="xi", columns="variant", values="ratio_truncated",
                        aggfunc="median").round(3).to_string())
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
