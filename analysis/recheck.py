"""Bounded pre-submission replication; never overwrites historical result tables.

Run as a module in the model's own environment, e.g.
python -m analysis.recheck natural --model TabDPT
python -m analysis.recheck synthetic
python -m analysis.recheck exaone
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time

import numpy as np
import pandas as pd

from common import datasets, leverage, models, paths

OUT = paths.ROOT / 'results' / 'recheck_20260925'
OUT.mkdir(exist_ok=True)


def emit(value):
    print(json.dumps(value, default=str), flush=True)


def environment():
    import torch
    torch.set_num_threads(4)
    versions = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'torch', 'tabpfn', 'tabdpt',
                 'tabicl', 'exaonetabular'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), packages=versions,
                torch_threads=torch.get_num_threads(), device='cpu')


def cached_insurance():
    from sklearn.datasets import fetch_openml
    sev = fetch_openml(data_id=41215, as_frame=True, parser='auto').data
    freq = fetch_openml(data_id=41214, as_frame=True, parser='auto').data
    df = sev.groupby('IDpol', as_index=False).ClaimAmount.sum().merge(
        freq.drop(columns=['ClaimNb']), on='IDpol', how='inner')
    y = df.pop('ClaimAmount').to_numpy(float)
    X, y = datasets.prepare(df.drop(columns=['IDpol']), y)
    known = paths.load_json(datasets.FINGERPRINTS)['freMTPL2sev']
    assert datasets.fingerprint(X, y) == known
    positive = y > 0
    return X[positive], y[positive]


def compare(got, reference, fields):
    return {name: dict(measured=float(got[name]), historical=float(reference[name]),
                       absolute_gap=float(abs(got[name] - reference[name])),
                       relative_gap=float(abs(got[name] - reference[name]) /
                                          max(abs(reference[name]), 1e-12)))
            for name in fields}


def natural(model):
    from experiments.h3_repair.clip_context_real import arm_row
    X, y = cached_insurance()
    found = leverage.subsamples(y, np.random.default_rng(31337), 2000, 1000, 5, 4000)
    idx, shift, lev = found[3][0]
    Xf, yf, Xte, yte = X[idx[:2000]], y[idx[:2000]], X[idx[2000:]], y[idx[2000:]]
    keep = np.arange(2000) != np.argmax(yf)
    fit_diagnostics = []
    original = np.linalg.lstsq

    def capture(a, b, *args, **kwargs):
        answer = original(a, b, *args, **kwargs)
        if a.ndim == 2 and a.shape == (len(Xte), 2):
            coef = answer[0]
            residual = np.max(np.abs(a @ coef - b)) / max(np.std(b), 1e-12)
            fit_diagnostics.append(dict(scale=float(coef[0]), offset=float(coef[1]),
                                        residual_over_prediction_sd=float(residual),
                                        rank=int(answer[2])))
        return answer

    if model == 'TabDPT':
        np.linalg.lstsq = capture
    predictions = {}
    try:
        for arm in ('without_max', 'raw', 'raw_repeat'):
            started = time.time()
            xx, yy = (Xf[keep], yf[keep]) if arm == 'without_max' else (Xf, yf)
            emit(dict(stage='predict', model=model, arm=arm))
            q = models.quantiles(model, xx, yy, Xte, 7000, [.5, .9, .99, .999], 1)
            assert q.shape == (1000, 4) and np.isfinite(q).all()
            assert (np.diff(q, axis=1) >= -1e-6).all()
            predictions[arm] = q
            emit(dict(stage='predicted', arm=arm, seconds=round(time.time()-started, 2)))
    finally:
        np.linalg.lstsq = original
    old = pd.read_csv(paths.result('clip_context_real.csv'))
    old = old[(old.model == model) & (old.bin == '4.0-inf') & (old.repeat == 0)]
    comparisons = {}
    for arm in ('without_max', 'raw'):
        got = arm_row(predictions[arm], predictions['without_max'], yte)
        ref = old[old.arm == arm].iloc[0]
        comparisons[arm] = compare(got, ref, ['d_q99', 'pb99', 'pb999'])
    repeat_gap = float(np.max(np.abs(predictions['raw'] - predictions['raw_repeat'])))
    np.savez_compressed(OUT / (model + '_natural_predictions.npz'), indices=idx,
                        y_test=yte, **predictions)
    return dict(model=model, bin='4.0-inf', repeat=0, context_sd_shift=shift,
                repeat_max_abs_gap=repeat_gap, affine_diagnostics=fit_diagnostics,
                comparisons=comparisons,
                indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest())


def synthetic():
    from experiments.h3_repair.clip_context import measure
    old = pd.read_csv(paths.result('clip_context.csv'))
    results = []
    for shift in (1., 20., 50.):
        for variant, cap in (('raw', 0.), ('clip', 200.)):
            cell = dict(model='TabPFN-V3', xi=.7, seed=7000, n_train=2000,
                        n_test=900, n_est=1, sd_shift_target=shift,
                        variant=variant, clip_c=cap)
            emit(dict(stage='predict', **cell))
            got = measure(cell)
            ref = old
            for k, value in cell.items():
                ref = ref[ref[k] == value]
            assert len(ref) == 1
            result = dict(cell=cell, comparison=compare(got, ref.iloc[0],
                          ['q99', 'q999', 'pb99', 'pb999', 'borders_in_data']))
            results.append(result)
            emit(result)
    return results


def exaone():
    from common.adapters import exaone as adapter
    rng = np.random.default_rng(7291)
    X = rng.normal(size=(256, 5))
    y = 2 + np.exp(.5 * X[:, 0] + rng.normal(size=256))
    Xte = rng.normal(size=(48, 5))
    levels = [.5, .9, .99, .999]
    m = adapter.create(seed=7000, n_est=1)
    m.fit(X, y)
    q = adapter.quantiles(m, Xte, levels)
    repeat = adapter.quantiles(m, Xte, levels)
    before = os.environ.get('PER_LEVEL')
    os.environ['PER_LEVEL'] = '1'
    try:
        per_level = adapter.quantiles(m, Xte, levels)
    finally:
        if before is None:
            os.environ.pop('PER_LEVEL', None)
        else:
            os.environ['PER_LEVEL'] = before
    m2 = adapter.create(seed=7000, n_est=1)
    m2.fit(X, 7*y + 11)
    rescaled = (adapter.quantiles(m2, Xte, levels)-11)/7
    assert q.shape == (48, 4) and np.isfinite(q).all()
    assert (np.diff(q, axis=1) >= -1e-6).all()
    return dict(shape=list(q.shape), monotone=True, finite=True,
                repeat_max_abs_gap=float(np.max(np.abs(q-repeat))),
                fast_vs_per_level_max_abs_gap=float(np.max(np.abs(q-per_level))),
                affine_max_relative_gap=float(np.max(np.abs(q-rescaled)) /
                                              max(np.max(np.abs(q)), 1e-12)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['natural', 'synthetic', 'exaone'])
    parser.add_argument('--model', default='TabPFN-v2.5')
    args = parser.parse_args()
    record = dict(kind=args.kind, environment=environment())
    emit(record)
    name = args.kind + ('_' + args.model if args.kind == 'natural' else '')
    try:
        record['results'] = (natural(args.model) if args.kind == 'natural' else
                             synthetic() if args.kind == 'synthetic' else exaone())
        record['execution'] = 'completed'
    except Exception as exc:
        record.update(execution='failed', error=repr(exc))
        raise
    finally:
        (OUT / (name + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
        emit(record)
