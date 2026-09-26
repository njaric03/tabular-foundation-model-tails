"""TabPFN-3's raw-space bin boundaries for one synthetic task, for the grid figure.

Rebuilds the contexts of `experiments/h3_repair/clip_context.py` for one task whose
border counts sit near the medians over all 40, fits TabPFN-3 on each and keeps the
boundaries that fall within the clean context's target range. Each count is checked
against the historical table. No quantiles are predicted.

venv-tfm/Scripts/python.exe -m analysis.recheck_grid_example
"""
import numpy as np
import pandas as pd

from common import generator, leverage, metrics, models, paths, quiet, transforms

quiet.silence()

XI, SEED, N_TRAIN = 0.7, 10000, 2000
CELLS = [('raw', 0.0, 1.0), ('raw', 0.0, 4.0), ('raw', 0.0, 20.0), ('raw', 0.0, 50.0),
         ('clip', 200.0, 50.0)]
OUT = paths.ROOT / 'results' / 'recheck_20260925' / 'grid_example.csv'


def main():
    rng = np.random.default_rng(SEED)
    train = generator.gpd(N_TRAIN, rng, xi=XI, clip=True)
    lo, hi = train.y.min(), train.y.max()
    old = pd.read_csv(paths.ROOT / 'results' / 'h3_repair' / 'clip_context.csv')
    old = old[(old.xi == XI) & (old.seed == SEED) & (old.n_est == 1)
              & (old.reason.fillna('') == '')]
    rows = [pd.DataFrame(dict(variant='context', clip_c=0.0, sd_shift_target=1.0,
                              value=train.y))]
    for variant, c, target in CELLS:
        X, y = train.X, train.y
        if target > 1.0:
            X, y = leverage.add_row(train.X, train.y, metrics.y0_for_sd_shift(train.y, target))
        y_fit, _, _ = transforms.transform(variant, c, y)
        m = models.tabpfn_regressor('TabPFN-V3', SEED, 1)
        m.fit(X, y_fit)
        borders = m.raw_space_bardist_.borders.detach().double().numpy()
        inside = borders[(borders >= lo) & (borders <= hi)]
        ref = old[(old.variant == variant) & (old.clip_c == c) & (old.sd_shift_target == target)]
        assert len(ref) == 1 and len(inside) == ref.borders_in_data.iloc[0], (variant, target)
        print(variant, target, len(inside), flush=True)
        rows.append(pd.DataFrame(dict(variant=variant, clip_c=c, sd_shift_target=target,
                                      value=inside)))
    pd.concat(rows).to_csv(OUT, index=False)
    print(f'written {OUT}')


if __name__ == '__main__':
    main()
