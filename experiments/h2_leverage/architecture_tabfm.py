# -*- coding: utf-8 -*-
"""
TabFM kao peti model: odsecen nosac i funkcija uticaja.

ZASTO BAS OVA DVA TESTA
-----------------------
`TabFMRegressor.predict(X)` vraca samo sredinu, pa se implicirano xi i disocijacija
nivo/oblik ne mogu meriti. Ali dva testa traze samo sredinu:

  ODSECEN NOSAC   model / prava sredina  naspram  model / E[Y | Y <= max(y_train)]
  UTICAJ          promena sredine kad se ubaci jedna tacka sa y = 100*max

PREDVIDJANJE, ZAPISANO PRE MERENJA
----------------------------------
`TabFMRegressor.__init__` ima `outlier_threshold: float = 4.0` -- dakle EKSPLICITNO
odsecanje outliera u pretprocesiranju, cega nema ni kod TabPFN-a ni kod TabDPT-a.

Odatle:
  (1) TabFM treba da bude ROBUSTAN na ubacenu tacku -- za razliku od TabDPT-a kome
      sredina menja znak i TabPFN-a koji je bimodalan;
  (2) ali bas zbog odsecanja treba da PODBACI na pravoj sredini pod teskim repom,
      jer odseca upravo one vrednosti koje sredinu i cine.

Ako oba prodju, taksonomija dobija cetvrtu kategoriju: eksplicitno klipovanje kupuje
robusnost po ceni nivoa.

POKRETANJE
----------
    venv-tabfm/Scripts/python.exe architecture_tabfm.py
"""
import os
import time

import numpy as np
import pandas as pd

from common import append, generator, metrics, paths, quiet

quiet.silence()

# W je u `common/generator.py`; ovde je stajala mrtva kopija.
XI_LISTA = [float(v) for v in os.environ.get("XI", "0.5,0.7,0.9").split(",")]
SEEDOVA = int(os.environ.get("SEEDS", "3"))
# TabFM je ~1,6 mlrd parametara; na CPU-u se mora skromno (isto kao experiments/h2_leverage/architecture_tabfm.py)
N_TRAIN = int(os.environ.get("N_TRAIN", "1000"))
N_TEST = int(os.environ.get("N_TEST", "200"))
N_EST = int(os.environ.get("N_EST", "1"))
OUT = os.environ.get("OUTPUT", "architecture_tabfm.csv")
KOLONE = ["n_est", "n_train", "n_test", "xi", "model", "seed", "ratio_true", "ratio_truncated", "mean_influence",
          "seconds", "reason"]
KEY = ["xi", "seed", "n_est", "n_train", "n_test"]


def main():
    import torch
    import tabfm
    from tabfm import TabFMRegressor

    ckpt = os.path.expanduser("~/tabfm-regression")
    if not os.path.isfile(os.path.join(ckpt, "model.safetensors")):
        raise SystemExit(f"nedostaje {ckpt}/model.safetensors")
    t0 = time.time()
    print("ucitavam TabFM tezine (6,6 GB, bf16)...", flush=True)
    model = tabfm.tabfm_v1_0_0_pytorch.load(
        model_type="regression", checkpoint_path=ckpt, device="cpu",
        dtype=torch.bfloat16)
    print(f"  gotovo [{time.time()-t0:.0f}s]", flush=True)

    def sredina(Xa, ya, Xte, seed):
        r = TabFMRegressor(model, n_estimators=N_EST, random_state=seed)
        r.fit(Xa, ya)
        return np.asarray(r.predict(Xte), dtype=float)

    gotovi = append.done(OUT, KEY)

    if gotovi:

        print(f"resuming, {len(gotovi)} cells already measured", flush=True)

    for xi in XI_LISTA:
        for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
            if append.key(dict(xi=xi, seed=seed), KEY[:2]) in gotovi:
                continue
            t1 = time.time()
            try:
                rng = np.random.default_rng(seed)
                _p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
                Xtr, ytr, _ = _p.X, _p.y, _p.s
                _p = generator.gpd(N_TEST, rng, xi=xi, clip=True)
                Xte, _, s_te = _p.X, _p.y, _p.s
                T = float(ytr.max())
                prava = s_te / (1 - xi)
                odsec = s_te * metrics.truncated_mean_std(T / s_te, xi)

                mod = sredina(Xtr, ytr, Xte, seed)
                x0 = np.zeros((1, Xtr.shape[1]))
                mod2 = sredina(np.vstack([Xtr, x0]),
                               np.concatenate([ytr, [100 * T]]), Xte, seed)
                uticaj = float(np.median((mod2 - mod) / np.maximum(np.abs(mod), 1e-9)))

                r = dict(ratio_true=float(np.mean(mod) / np.mean(prava)),
                         ratio_truncated=float(np.mean(mod) / np.mean(odsec)),
                         mean_influence=uticaj, reason="")
            except Exception as e:
                r = dict(reason=f"{type(e).__name__}: {e}"[:110])
            r.update(n_est=N_EST, n_train=N_TRAIN, n_test=N_TEST,
                     xi=xi, model="TabFM", seed=seed,
                     seconds=round(time.time() - t1, 1))
            append.write(OUT, r, KOLONE)
            poruka = r.get("reason") or (
                f"model/prava {r['ratio_true']:.3f}  model/odsecena {r['ratio_truncated']:.3f}  "
                f"uticaj {r['mean_influence']:+.1%}")
            print(f"  xi={xi} TabFM seed={seed}  {poruka}  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); d = d[d.ratio_true.notna()]
    print(f"\n=== TabFM, {len(d)} pokretanja ===")
    print(d.groupby("xi")[["ratio_true", "ratio_truncated", "mean_influence"]]
          .median().round(3).to_string())
    print("\nporedjenje pri xi=0.9 (medijane, n_est=1):")
    # Cross-model comparison lives in analysis/h2_leverage.ipynb, which reads
    # the result CSVs, instead of being pasted here as constants.
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
