# -*- coding: utf-8 -*-
"""
Predvidja li ARHITEKTURA glave ponasanje u repu?

PREDVIDJANJE IZ KODA, PRE MERENJA
---------------------------------
Iz izvornog koda tri modela:

  TabPFN   `regressor.py`: raw_space_bardist_ = borders * y_train_std_ + y_train_mean_
           -> fiksna mreza korpi u z-prostoru, skalirana empirijskom sd
  TabDPT   `estimator.py`: edges = linspace(bin_min=-10, bin_max=+10, 2048)
           uz train_y = normalize_data(train_y)
           -> ISTO, i jos sa TVRDOM granicom nosaca na mean +- 10*std
  TabICL   999 kvantila, bez mreze korpi
           -> nema sta da izgubi rezoluciju skokovito

Odatle predvidjanje koje se moze oboriti:

  (1) TabPFN i TabDPT pokazuju ODSECEN NOSAC -- sredina im je bliza odsecenoj nego pravoj;
  (2) TabPFN i TabDPT gube rezoluciju kad se ubaci outlier, TabICL ne;
  (3) TabDPT ima TVRDU granicu: ne moze da predvidi iznad mean + 10*std, bez obzira na
      podatke. To je proverljivo analiticki.

Ako TabDPT prati TabPFN a ne TabICL, arhitektura predvidja ponasanje.

MERI SE (samo preko javnog API-ja, dakle sredina)
-------------------------------------------------
  odnos_prava    sredina modela / prava uslovna sredina
  odnos_odsec    sredina modela / E[Y | Y <= max(y_train)]
  uticaj         promena sredine kad se ubaci jedna tacka sa y = 100*max
  granica        koliki deo prave sredine uopste staje u nosac mean +- 10*std

POKRETANJE (TabDPT ima svoj venv)
---------------------------------
    venv-tabdpt/Scripts/python.exe architecture_tabdpt.py
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from common import append, generator, metrics, models, paths

warnings.filterwarnings("ignore")

W = np.array([1.0, -0.7, 0.5, 0.0, 0.0])
XI_LISTA = [float(v) for v in os.environ.get("XI", "0.5,0.7,0.9").split(",")]
SEEDOVA = int(os.environ.get("SEEDS", "3"))
N_TRAIN, N_TEST = 2000, 900
OUT = os.environ.get("OUTPUT", "architecture_tabdpt.csv")
KOLONE = ["xi", "model", "seed", "ratio_true", "ratio_truncated", "mean_influence",
          "bound_coverage", "seconds", "reason"]
KEY = ["xi", "model", "seed"]


def sredina_tabdpt(Xtr, ytr, Xte, seed):
    return models.mean("TabDPT", Xtr, ytr, Xte, seed=seed)


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    t0 = time.time()
    for xi in XI_LISTA:
        for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
            if append.key(dict(xi=xi, model="TabDPT", seed=seed), KEY[:3]) in gotovi:
                continue
            t1 = time.time()
            try:
                rng = np.random.default_rng(seed)
                _p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
                Xtr, ytr, _ = _p.X, _p.y, _p.s
                _p = generator.gpd(N_TEST, rng, xi=xi, clip=True)
                Xte, yte, s_te = _p.X, _p.y, _p.s
                T = float(ytr.max())
                prava = s_te / (1 - xi)
                odsec = s_te * metrics.truncated_mean_std(T / s_te, xi)

                mod = sredina_tabdpt(Xtr, ytr, Xte, seed)

                # tvrda granica nosaca: mean + 10*std trening targeta
                gornja = ytr.mean() + 10.0 * ytr.std()
                pokrivenost = float(np.mean(prava <= gornja))

                # uticaj jedne ekstremne tacke na sredinu
                x0 = np.zeros((1, Xtr.shape[1]))
                mod2 = sredina_tabdpt(np.vstack([Xtr, x0]),
                                      np.concatenate([ytr, [100 * T]]), Xte, seed)
                uticaj = float(np.median((mod2 - mod) / np.maximum(np.abs(mod), 1e-9)))

                r = dict(ratio_true=float(np.mean(mod) / np.mean(prava)),
                         ratio_truncated=float(np.mean(mod) / np.mean(odsec)),
                         mean_influence=uticaj, bound_coverage=pokrivenost,
                         reason="")
            except Exception as e:
                r = dict(reason=f"{type(e).__name__}: {e}"[:110])
            r.update(xi=xi, model="TabDPT", seed=seed,
                     seconds=round(time.time() - t1, 1))
            append.write(OUT, r, KOLONE)
            poruka = r.get("reason") or (
                f"model/prava {r['ratio_true']:.3f}  model/odsecena {r['ratio_truncated']:.3f}  "
                f"uticaj {r['mean_influence']:+.1%}  nosac pokriva {r['bound_coverage']:.0%}")
            print(f"  xi={xi} TabDPT seed={seed}  {poruka}  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); d = d[d.ratio_true.notna()]
    print("\n=== TabDPT ===")
    print(d.groupby("xi")[["ratio_true", "ratio_truncated", "mean_influence",
                           "bound_coverage"]].median().round(3).to_string())
    # Cross-model comparison lives in analysis/h2_leverage.ipynb, which reads
    # the result CSVs, instead of being pasted here as constants.
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
