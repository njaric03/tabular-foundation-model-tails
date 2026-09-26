"""Recheck the previously omitted tree controls on one recorded insurance context."""
import importlib.metadata
import json

import numpy as np
import pandas as pd

from analysis.recheck import OUT, cached_insurance
from common import metrics, models, transforms


def main():
    X, y = cached_insurance()
    with np.load(OUT / 'TabPFN-v2.5_natural_predictions.npz') as saved:
        idx = saved['indices']
        assert np.array_equal(y[idx[2000:]], saved['y_test'])
    Xf, yf, Xte, yte = X[idx[:2000]], y[idx[:2000]], X[idx[2000:]], y[idx[2000:]]
    historical = pd.read_csv('results/h3_repair/clip_context_real.csv')
    rows = []
    for model in ('XGB', 'CB'):
        for arm in ('raw', 'clip'):
            y_fit = yf if arm == 'raw' else transforms.transform('clip', 200., yf)[0]
            q = models.quantiles(model, Xf, y_fit, Xte, seed=7000,
                                 levels=[.5, .9, .99, .999], n_est=1)
            assert q.shape == (1000, 4) and np.isfinite(q).all()
            ref = historical[(historical.model == model) & (historical.bin == '4.0-inf')
                             & (historical.repeat == 0) & (historical.arm == arm)
                             & (historical.clip_c == (200 if arm == 'clip' else 0))]
            assert len(ref) == 1
            got = metrics.pinball(yte, q[:, 2], .99)
            old = float(ref.iloc[0].pb99)
            row = dict(model=model, arm=arm, bin='4.0-inf', repeat=0,
                       n_fit=2000, n_test=1000, seed=7000,
                       pb99=got, historical_pb99=old, relative_gap=abs(got/old-1))
            rows.append(row)
            print(json.dumps(row), flush=True)
    record = dict(packages={p: importlib.metadata.version(p)
                            for p in ('xgboost', 'catboost', 'numpy', 'scikit-learn')},
                  scope='One recorded high-leverage insurance context, repeat 0.',
                  results=rows)
    (OUT / 'baseline_capping_check.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
