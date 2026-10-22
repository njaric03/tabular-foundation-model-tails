"""The model-by-experiment matrix of the thesis (outline 3.1), read from results/.

For every result file with a `model` column, which models it measures and with how many
ensemble members. Writes the matrix to thesis/figures/matrica_modela.csv, which is not
tracked, and prints it. Pre-training and fine-tuning arms are left out, as the thesis does.

    python -m analysis.model_matrix
"""
from pathlib import Path

import pandas as pd

from common.models import SYNONYMS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'thesis/figures'
SKIP = ('nanoTabPFN', 'TabICLv2-FT')


def main():
    records = []
    for path in sorted((ROOT / 'results').glob('*/*.csv')):
        try:
            d = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if 'model' not in d:
            continue
        if 'reason' in d:
            d = d[d.reason.fillna('') == '']
        members = d.groupby('model').n_est.unique() if 'n_est' in d else None
        for m in d.model.dropna().unique():
            if str(m).startswith(SKIP):
                continue
            n = members[m] if members is not None else []
            # Older files carry a synonym, `XGBoost` in prevalence_models.csv.
            records.append(dict(part=path.parent.name, file=path.stem,
                                model=SYNONYMS.get(m, m),
                                members='/'.join(str(int(x)) for x in sorted(n)
                                                 if pd.notna(x))))
    data = pd.DataFrame(records)
    matrix = data.pivot_table(index='model', columns='part', values='file', aggfunc='count',
                              fill_value=0)
    matrix['total'] = matrix.sum(axis=1)
    matrix = matrix.sort_values('total', ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / 'matrica_modela.csv', index=False)
    print('result files per model and part of the thesis')
    print(matrix.to_string())
    eight = data[data.members.str.contains('8')]
    print('\nfiles with eight members:', ', '.join(sorted(set(eight.file))))


if __name__ == '__main__':
    main()
