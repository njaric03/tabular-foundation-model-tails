"""Vector figures and their plotted data for the research supplement.

Run from the repository root: python -m analysis.supplement_figures
All plotted points come from the historical result tables; no predictions are rerun.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/h3_repair'
BINS_LEV = ['2.0-4.0', '4.0-inf']
TFM = ['TabPFN-v2.5', 'TabPFN-v2.6', 'TabPFN-V3', 'TabICLv2', 'EXAONE', 'TabDPT']
ORDER = TFM + ['GBM', 'XGB', 'CB']
OUT = ROOT / 'supplement'
FIG = OUT / 'figures'
DATA = OUT / 'plot_data'
README_FIG = ROOT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)
INK, SLATE, BLUE, GREY = '#222222', '#35546E', '#137C8B', '#9A9A9A'
# Light to dark with the size of the added target; the darkest is also 50x in the capping figure.
ORANGES = ['#EBC3AE', '#D98B63', '#BA5A31']
LEVELS, TAGS = [.5, .9, .99, .999], ['50', '90', '99', '999']
PERCENTILES = ['Median', '90th', '99th', '99.9th']
LABELS = {'TabPFN-v2.5': 'TabPFN-2.5', 'TabPFN-v2.6': 'TabPFN-2.6',
          'TabPFN-V3': 'TabPFN-3', 'TabICLv2': 'TabICLv2', 'EXAONE': 'EXAONE-Tabular',
          'TabDPT': 'TabDPT', 'GBM': 'Scikit-learn GBM', 'XGB': 'XGBoost',
          'CB': 'CatBoost'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                     'mathtext.fontset': 'stix', 'font.size': 11,
                     'axes.labelsize': 11, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#AAAAAA', 'axes.linewidth': .6,
                     'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
                     'legend.frameon': False, 'legend.fontsize': 10,
                     'legend.title_fontsize': 10,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def load_natural():
    frames = [pd.read_csv(RESULTS / name) for name in
              ('clip_context_real.csv', 'clip_context_real_v25v26.csv')]
    d = pd.concat(frames)
    d = d[d.reason.fillna('') == ''].copy()
    key = ['dataset', 'model', 'bin', 'repeat', 'arm', 'clip_c', 'n_est', 'n_fit', 'n_test']
    d = d.drop_duplicates(key, keep='last')
    d['arm_c'] = np.where(d.arm == 'clip', 'clip_' + d.clip_c.map('{:g}'.format), d.arm)
    return d


def save(fig, name):
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', pad_inches=.07)
    fig.savefig(FIG / (name + '.png'), dpi=180, bbox_inches='tight', pad_inches=.07)
    # The README shows the PNGs; a packaged copy of this script has no figures folder.
    if README_FIG.is_dir():
        fig.savefig(README_FIG / (name + '.png'), dpi=180, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)


def dots(ax, frame, value, n, left_label, right_label, accent):
    """One row per model: each context as a dot, the median as a diamond."""
    for i, model in enumerate(ORDER):
        values = frame[frame.model == model][value].to_numpy()
        assert len(values) == n and np.isfinite(values).all()
        colour = accent if model in TFM else '#777777'
        ax.scatter(values, i + np.linspace(-.16, .16, n), s=22, color=colour,
                   alpha=.45 if model in TFM else .65, linewidths=0)
        ax.scatter([np.median(values)], [i], s=54, marker='D', color=colour,
                   edgecolors='white', linewidths=.65, zorder=4)
    ax.set_yticks(range(len(ORDER)), [LABELS[m] for m in ORDER])
    ax.set_ylim(len(ORDER) - .55, -.55)
    ax.grid(axis='x', color='#E8E8E8', linewidth=.5, zorder=0)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.axhline(len(TFM) - .5, color='#DDDDDD', lw=.65)
    ax.text(0, 1.03, left_label, transform=ax.transAxes, fontsize=10, color=GREY)
    ax.text(1, 1.03, right_label, ha='right', transform=ax.transAxes, fontsize=10,
            color=GREY)


def natural():
    d = load_natural()
    x = d[(d.arm_c == 'raw') & (d.bin == '4.0-inf')].copy()
    x['ratio'] = 1 + x.d_q99
    x[['model', 'repeat', 'sd_shift', 'ratio']].to_csv(DATA/'natural_sensitivity.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.5, 1.85), layout='constrained')
    dots(ax, x, 'ratio', 5, 'Lower predictions', 'Higher predictions', SLATE)
    ax.axvline(1, color=INK, lw=.8, zorder=0)
    ax.set_xscale('log')
    ax.set_xlim(.3, 3.3)
    ax.set_xticks([.5, 1, 2, 3], ['0.5×', '1×', '2×', '3×'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Predicted 99th percentile with the largest claim, relative to without it')
    save(fig, 'natural_sensitivity')

    idx = ['dataset', 'model', 'bin', 'repeat', 'n_est', 'n_fit', 'n_test']
    raw = d[d.arm_c == 'raw'].set_index(idx).pb99
    cap = d[(d.arm_c == 'clip_200') & d.bin.isin(BINS_LEV)].copy().set_index(idx)
    assert raw.index.is_unique and cap.index.is_unique
    cap['relative_change'] = cap.pb99 / raw.reindex(cap.index) - 1
    cap = cap.reset_index()
    cap[['model', 'bin', 'repeat', 'relative_change']].to_csv(DATA/'natural_capping.csv',
                                                             index=False)
    fig, ax = plt.subplots(figsize=(6.5, 1.85), layout='constrained')
    dots(ax, cap, 'relative_change', 10, 'Lower loss', 'Higher loss', BLUE)
    ax.axvline(0, color=INK, lw=.8, zorder=0)
    ax.set_xlim(-1, .3)
    ax.set_xticks([-.75, -.5, -.25, 0, .25])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel('Change in 99th-percentile pinball loss after capping')
    save(fig, 'natural_capping')


def bars(ax, groups, labels, colours, legend_title):
    """Median loss change per percentile, one bar per setting, each labelled."""
    width = .8 / len(groups)
    for k, (values, label, colour) in enumerate(zip(groups, labels, colours)):
        x = np.arange(4) + (k - (len(groups) - 1) / 2) * width
        ax.bar(x, values, width=width * .92, color=colour, label=label, zorder=2)
        for xx, v in zip(x, values):
            # Bars within half a percent of zero are invisible; the label carries them.
            text = '0%' if abs(v) < .005 else f'{v:+.0%}'
            ax.text(xx, max(v, 0) + .02, text, ha='center', va='bottom', fontsize=9.5)
    ax.axhline(0, color=INK, lw=.8, zorder=3)
    ax.set_xticks(range(4), PERCENTILES)
    ax.set_xlabel('Predicted percentile')
    ax.set_ylabel('Change in pinball loss')
    ax.set_ylim(-.04, .95)
    ax.set_yticks([0, .25, .5, .75])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis='y', color='#E8E8E8', lw=.5, zorder=0)
    ax.tick_params(axis='x', length=0)
    ax.legend(title=legend_title, loc='upper left', ncol=len(groups),
              alignment='left', handlelength=1.2, columnspacing=1.2)


def synthetic():
    """Loss change against the clean context, per task and percentile, then medians."""
    d = pd.read_csv(RESULTS / 'clip_context.csv')
    d = d[(d.reason.fillna('') == '')
          & ((d.variant == 'raw') | ((d.variant == 'clip') & (d.clip_c == 200)))].copy()
    keys = ['xi', 'seed', 'n_train', 'n_test']
    cols = ['pb' + t for t in TAGS]
    clean = d[(d.variant == 'raw') & (d.sd_shift_target == 1)]
    assert not clean.duplicated(keys).any() and (clean[cols] > 0).all().all()
    d = d.merge(clean[keys + cols], on=keys, validate='many_to_one', suffixes=('', '_clean'))
    assert not d.duplicated(keys + ['variant', 'sd_shift_target']).any()
    records = []
    for tag, level in zip(TAGS, LEVELS):
        part = d[keys + ['variant', 'sd_shift_target']].copy()
        part['quantile'] = level
        part['loss'] = d['pb' + tag]
        part['clean_loss'] = d['pb' + tag + '_clean']
        part['relative_change'] = part.loss / part.clean_loss - 1
        records.append(part)
    long = pd.concat(records, ignore_index=True)
    assert np.isfinite(long.relative_change).all()
    long.to_csv(DATA / 'synthetic.csv', index=False)
    med = long.pivot_table(index=['variant', 'sd_shift_target'], columns='quantile',
                           values='relative_change', aggfunc='median')[LEVELS]

    fig, ax = plt.subplots(figsize=(6.5, 1.7), layout='constrained')
    bars(ax, [med.loc[('raw', s)].to_numpy() for s in (4, 20, 50)],
         ['4×', '20×', '50×'], ORANGES, 'One added target multiplies the context SD by')
    save(fig, 'synthetic_error')

    # The added target always exceeds the cap, so the capped context is the same at every
    # shift; 50x is shown.
    assert np.allclose(med.loc[('clip', 4)], med.loc[('clip', 50)])
    fig, ax = plt.subplots(figsize=(6.5, 1.7), layout='constrained')
    bars(ax, [med.loc[('raw', 50)].to_numpy(), med.loc[('clip', 50)].to_numpy()],
         ['Without capping', 'With capping'], [ORANGES[-1], BLUE],
         'Context SD multiplied by 50')
    save(fig, 'synthetic_capping')


if __name__ == '__main__':
    synthetic()
    natural()
    print(f'Figures and plotted data written to {OUT}')
