"""Separate repeatability and thread-count sensitivity from historical drift."""
import json

import torch

from analysis.recheck import OUT, emit, environment
from experiments.h3_repair.clip_context import measure

record = dict(environment=environment(), results=[])
for threads in (4, 1):
    torch.set_num_threads(threads)
    for repeat in range(2):
        cell = dict(model='TabPFN-V3', xi=.7, seed=7000, n_train=2000,
                    n_test=900, n_est=1, sd_shift_target=20., variant='raw', clip_c=0.)
        result = dict(threads=threads, repeat=repeat, measured=measure(cell))
        record['results'].append(result)
        emit(result)
(OUT/'v3_numeric_check.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
