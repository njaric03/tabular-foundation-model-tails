"""Ten paired tasks to assess whether historical drift reverses the main effect.

Uses two tail indices and the first five original seeds. It is a bounded robustness
check, not a replacement for the historical 40-task figures.
"""
import json

from analysis.recheck import OUT, emit, environment
from experiments.h3_repair.clip_context import measure

PATH = OUT / 'v3_direction_check.json'
record = dict(environment=environment(), results=[])
for xi in (.7, .9):
    for seed in (7000, 8000, 9000, 10000, 11000):
        for shift, variant, cap in ((1., 'raw', 0.), (50., 'raw', 0.), (50., 'clip', 200.)):
            cell = dict(model='TabPFN-V3', xi=xi, seed=seed, n_train=2000,
                        n_test=900, n_est=1, sd_shift_target=shift,
                        variant=variant, clip_c=cap)
            result = dict(cell=cell, measured=measure(cell))
            record['results'].append(result)
            emit(dict(xi=xi, seed=seed, shift=shift, variant=variant,
                      pb99=result['measured']['pb99'], pb999=result['measured']['pb999']))
            PATH.write_text(json.dumps(record, indent=2), encoding='utf-8')
record['execution'] = 'completed'
PATH.write_text(json.dumps(record, indent=2), encoding='utf-8')
