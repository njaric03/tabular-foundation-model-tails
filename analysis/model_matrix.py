"""The model-by-experiment matrix of the thesis (outline 3.1), read from results/.

For every result file with a `model` column, which models it measures and with how many
ensemble members. Writes the matrix to thesis/figures/matrica_modela.csv and the body of
table 3.1 to tabela_modela.tex, neither tracked, and prints both. Pre-training and
fine-tuning arms are left out, as the thesis does.

    python -m analysis.model_matrix
"""
from pathlib import Path

import pandas as pd

from common.models import SYNONYMS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'thesis/figures'
SKIP = ('nanoTabPFN', 'TabICLv2-FT')
# Table 3.1 of the thesis: row label, models, role (outline 3.1). The package version is
# read from the provenance rows of the model's result files; trees have no members.
TABLE = [
    ('TabPFN-3', ['TabPFN-V3'], 'glavni model'),
    ('TabPFN-3.5', ['TabPFN-v3.5'], 'novija verzija'),
    ('TabICLv2', ['TabICLv2'], 'glavni model'),
    ('GBM', ['GBM'], 'kontrola na stablima'),
    ('XGBoost, CatBoost', ['XGB', 'CB'], 'dopunske kontrole'),
    ('TabPFN-2.5, TabPFN-2.6', ['TabPFN-v2.5', 'TabPFN-v2.6'], 'dopunska provera'),
    ('EXAONE-Tabular', ['EXAONE'], 'dopunska provera'),
    ('TabDPT', ['TabDPT'], 'dopunska provera'),
    ('LimiX-2', ['LimiX-2'], 'dopunska provera'),
    ('Causilo', ['Causilo'], 'dopunska provera'),
]
PACKAGE = {'TabPFN-V3': 'tabpfn', 'TabPFN-v3.5': 'tabpfn', 'TabPFN-v2.5': 'tabpfn',
           'TabPFN-v2.6': 'tabpfn', 'TabICLv2': 'tabicl', 'GBM': 'scikit_learn',
           'XGB': 'xgboost', 'CB': 'catboost', 'EXAONE': 'exaone_tabular',
           'TabDPT': 'tabdpt', 'LimiX-2': 'limix', 'Causilo': 'causilo'}
TREES = ('GBM', 'XGB', 'CB')


def latex(data):
    """The body of table 3.1, written to tabela_modela.tex for \\input."""
    prov = pd.concat([pd.read_csv(ROOT / 'results' / name, low_memory=False)
                      for name in ('provenance.csv', 'provenance_tfmp.csv')])
    prov['stem'] = prov.output.astype(str).str.split('/').str[-1].str.replace(
        r'\.csv$', '', regex=True)
    lines = [r'\begin{tabularx}{\textwidth}{@{}l X l l@{}}', r'\toprule',
             r'Model & Paket i verzija & Članova ansambla & Uloga \\', r'\midrule']
    for label, models, role in TABLE:
        packages, members = [], set()
        for m in models:
            g = data[data.model == m]
            pkg = PACKAGE[m]
            versions = sorted(prov[prov.stem.isin(g.file)][f'v_{pkg}'].dropna().astype(str)
                              .unique())
            text = pkg.replace('_', '-') + ' ' + ', '.join(versions)
            if text not in packages:
                packages.append(text)
            members |= {int(x) for v in g.members.dropna() for x in str(v).split('/') if x}
        n = '---' if set(models) <= set(TREES) else ', '.join(map(str, sorted(members)))
        lines.append(f'{label} & {"; ".join(packages)} & {n} & {role} \\\\')
    lines += [r'\bottomrule', r'\end{tabularx}']
    (OUT / 'tabela_modela.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


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
    latex(data)


if __name__ == '__main__':
    main()
