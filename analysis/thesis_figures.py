"""Main figures of parts two and three of the thesis, with Serbian labels.

Run from the repository root: python -m analysis.thesis_figures
Reads the confirmation outputs of 21 and 22 October 2026 and writes PDF, PNG and the
plotted values to thesis/figures/, which is not tracked; the dose figure is also drawn
with English labels into figures/ for the README. A figure whose files are not
there yet is skipped. TabPFN-3.5 ran on a GPU and every other model on CPU, so a row of
3.5 is compared with the others by its medians only.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator, PercentFormatter

from analysis.supplement_figures import INK, GREY

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/h3_repair'
OUT = ROOT / 'thesis/figures'
# Two hues, validated together on white with the dataviz validator (all pairs): the cap
# and the log are the two repairs and keep these colours in every figure. raw is ink.
CAP, LOG, RAW = '#2a78d6', '#eb6834', '#555555'
ARMS = {'raw': ('bez izmene', RAW), 'clip': ('odsecanje', CAP), 'log': ('logaritam', LOG)}
# (label, file in the unit-error confirmation, file in the segment run, model name)
CONFIGS = [
    ('TabPFN-3, 1 član', 'unit_error_confirm_log.csv', 'legit_segment_log.csv', 'TabPFN-V3'),
    ('TabPFN-3, 8 članova', 'unit_error_confirm_nest8.csv', 'legit_segment_nest8.csv',
     'TabPFN-V3'),
    ('TabPFN-3.5, 1 član', 'unit_error_confirm_tabpfn35_nest1.csv',
     'legit_segment_tabpfn35_nest1.csv', 'TabPFN-v3.5'),
    ('TabPFN-3.5, 8 članova', 'unit_error_confirm_tabpfn35_nest8.csv',
     'legit_segment_tabpfn35_nest8.csv', 'TabPFN-v3.5'),
    ('TabICLv2', 'unit_error_confirm_log.csv', 'legit_segment_log.csv', 'TabICLv2'),
    ('GBM (kontrola)', 'unit_error_confirm_log.csv', 'legit_segment_log.csv', 'GBM'),
]
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                     'mathtext.fontset': 'stix', 'font.size': 11,
                     'axes.labelsize': 11, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#AAAAAA', 'axes.linewidth': .6,
                     'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
                     'legend.frameon': False, 'legend.fontsize': 10,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    d = pd.read_csv(path)
    return d[d.reason.fillna('') == ''].copy()


def save(fig, name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight', pad_inches=.07)
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight', pad_inches=.07)
    data.to_csv(OUT / (name + '.csv'), index=False)
    plt.close(fig)
    print(f'wrote {OUT / name}.pdf')


def rows(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color='#E8E8E8', linewidth=.5, zorder=0)
    # The tree control sits apart from the foundation models.
    ax.axhline(len(labels) - 1.5, color='#DDDDDD', lw=.65)


def unit_error():
    """Change in 99th-percentile loss of each repair against raw, visible unit error."""
    records = []
    for label, unit_file, _, model in CONFIGS:
        d = load(unit_file)
        if d is None:
            print(f'unit_error: {unit_file} is missing, figure skipped')
            return
        d = d[(d.model == model) & (d.context == 'visible')]
        w = d.pivot_table(index=['dataset', 'repeat'], columns='arm', values='pb99')
        for arm in ('clip', 'log'):
            rel = (w[arm] / w['raw'] - 1).groupby(level='dataset').median()
            records += [dict(config=label, arm=arm, dataset=k, change=v) for k, v in rel.items()]
    data = pd.DataFrame(records)
    labels = [c[0] for c in CONFIGS]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True, sharex=True,
                             layout='constrained')
    for ax, arm in zip(axes, ('clip', 'log')):
        name, colour = ARMS[arm]
        for i, label in enumerate(labels):
            v = data[(data.config == label) & (data.arm == arm)].change.to_numpy()
            assert len(v) == 5 and np.isfinite(v).all()
            ax.scatter(v, i + np.linspace(-.15, .15, 5), s=20, color=colour, alpha=.5,
                       linewidths=0, zorder=3)
            ax.scatter([np.median(v)], [i], s=52, marker='D', color=colour,
                       edgecolors='white', linewidths=.65, zorder=4)
        ax.axvline(0, color=INK, lw=.8, zorder=2)
        rows(ax, labels)
        ax.set_title(name, fontsize=11, loc='left', color=INK)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].text(0, -.2, 'niži gubitak  ←', transform=axes[0].transAxes, fontsize=9.5,
                 color=GREY)
    fig.supxlabel('Promena gubitka na 0,99 prema neizmenjenom kontekstu, '
                  'greška jedinice ×100', fontsize=10.5)
    save(fig, 'greska_jedinice_popravke', data)


def segment_tail():
    """Predicted over true Q(0.999), outside and inside a legitimate high-scale segment."""
    records = []
    for label, _, seg_file, model in CONFIGS:
        d = load(seg_file)
        if d is None:
            print(f'segment_tail: {seg_file} is missing, figure skipped')
            return
        d = d[(d.model == model) & (d.seg_scale == 20.0) & (d.sd_shift_target == 1.0)]
        for arm in ARMS:
            for part in ('out', 'seg'):
                v = d[d.variant == arm][f'qratio999_{part}'].to_numpy()
                assert len(v) == 20
                records.append(dict(config=label, arm=arm, part=part, median=np.median(v),
                                    q25=np.quantile(v, .25), q75=np.quantile(v, .75)))
    data = pd.DataFrame(records)
    labels = [c[0] for c in CONFIGS]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), sharey=True, sharex=True,
                             layout='constrained')
    offsets = dict(zip(ARMS, (-.22, 0, .22)))
    for ax, part, title in zip(axes, ('out', 'seg'), ('van segmenta', 'u segmentu')):
        for i, label in enumerate(labels):
            for arm, (name, colour) in ARMS.items():
                r = data[(data.config == label) & (data.arm == arm) & (data.part == part)]
                y = i + offsets[arm]
                ax.plot([r.q25.item(), r.q75.item()], [y, y], color=colour, lw=2,
                        solid_capstyle='round', zorder=3)
                ax.scatter([r['median'].item()], [y], s=34, marker='D', color=colour,
                           edgecolors='white', linewidths=.6, zorder=4,
                           label=name if i == 0 and part == 'out' else None)
        ax.axvline(1, color=INK, lw=.8, zorder=2)
        rows(ax, labels)
        ax.set_title(title, fontsize=11, loc='left', color=INK)
        ax.set_xscale('log')
        ax.set_xlim(.05, 4)
        ax.set_xticks([.1, .25, .5, 1, 2], ['0,1×', '0,25×', '0,5×', '1×', '2×'])
        ax.xaxis.set_minor_locator(NullLocator())
    fig.legend(loc='outside upper center', ncol=3, handletextpad=.3, columnspacing=1.4)
    fig.supxlabel('Predviđeni Q(0,999) prema pravom, čist kontekst, segment ×20 '
                  '(medijana i interkvartilni raspon, 20 semena)', fontsize=10.5)
    save(fig, 'segment_rep', data)


# The dose figure is also drawn in English for the README.
DOSE_TEXT = {
    'sr': dict(one='1 član', eight='8 članova (podrazumevano)', level='nivo {}',
               x='Povećanje standardne devijacije konteksta jednim dodatim redom '
                 '(medijana, 40 zadataka)',
               y='Rast očekivanog gubitka prema čistom kontekstu'),
    'en': dict(one='1 member', eight='8 members (default)', level='{} level',
               x='Context sd multiplied by one added row (median over 40 tasks)',
               y='Rise in expected pinball loss over the clean context'),
}


def dose(lang='sr'):
    """Median rise of raw's expected loss over the clean context, by the sd shift."""
    text = DOSE_TEXT[lang]
    files = {('TabPFN-3', 1): 'clip_context_dose_v3_nest1.csv',
             ('TabPFN-3', 8): 'clip_context_dose_v3_nest8.csv',
             ('TabPFN-3.5', 1): 'clip_context_dose_tabpfn35_nest1.csv',
             ('TabPFN-3.5', 8): 'clip_context_dose_tabpfn35_nest8.csv'}
    shifts = [1.0, 4.0, 20.0, 50.0]
    records = []
    for (model, n), name in files.items():
        d = load(name)
        if d is None:
            print(f'dose: {name} is missing, figure skipped')
            return
        d = d[d.variant == 'raw']
        for tag, level in (('99', .99), ('999', .999)):
            w = d.pivot_table(index=['xi', 'seed'], columns='sd_shift_target',
                              values=f'epb{tag}')
            if list(w.columns) != shifts or w.isna().any().any():
                print(f'dose: {name} is not complete yet, figure skipped')
                return
            for s in shifts:
                rel = w[s] / w[1.0] - 1
                records.append(dict(model=model, members=n, level=level, shift=s,
                                    median=rel.median(), tasks=len(rel)))
    data = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True, sharey='row',
                             layout='constrained')
    styles = {1: dict(color=INK, ls='-', marker='o', label=text['one']),
              8: dict(color=GREY, ls='--', marker='s', label=text['eight'])}
    for r, level in enumerate((.99, .999)):
        for c, model in enumerate(('TabPFN-3', 'TabPFN-3.5')):
            ax = axes[r, c]
            for n, style in styles.items():
                p = data[(data.model == model) & (data.members == n) & (data.level == level)]
                ax.plot(range(len(shifts)), p['median'], lw=2, ms=6,
                        label=style['label'] if r == 0 and c == 0 else None,
                        **{k: v for k, v in style.items() if k != 'label'})
            ax.axhline(0, color=INK, lw=.8, zorder=0)
            ax.grid(axis='y', color='#E8E8E8', lw=.5, zorder=0)
            # Ticks on whole percents, so the formatter never rounds 2.5% to 2%.
            ax.set_yticks([0, .02, .04, .06] if level == .99 else [0, .5, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            if r == 0:
                ax.set_title(model, fontsize=11, loc='left', color=INK)
            if c == 0:
                shown = str(level).replace('.', ',') if lang == 'sr' else str(level)
                ax.set_ylabel(text['level'].format(shown))
        axes[1, 0].set_xticks(range(len(shifts)), ['1×', '4×', '20×', '50×'])
    fig.legend(loc='outside upper center', ncol=2)
    fig.supxlabel(text['x'], fontsize=10.5)
    fig.supylabel(text['y'], fontsize=10.5)
    if lang == 'sr':
        save(fig, 'doza_ansambl', data)
    else:
        fig.savefig(ROOT / 'figures/dose_by_configuration.png', dpi=180,
                    bbox_inches='tight', pad_inches=.07)
        plt.close(fig)
        print(f'wrote {ROOT / "figures/dose_by_configuration.png"}')


if __name__ == '__main__':
    unit_error()
    segment_tail()
    dose()
    dose('en')
