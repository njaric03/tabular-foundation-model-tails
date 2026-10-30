"""Main figures of parts two and three of the thesis, with Serbian labels.

Run from the repository root: python -m analysis.thesis_figures
Reads the confirmation outputs of 21 and 22 October 2026, the target-transform test of
28 October and the conditional cap of 29 October, and writes PDF, PNG and the plotted
values to thesis/figures/, which is not tracked; the dose figure is also drawn with
English labels into figures/ for the README. A figure whose files are not there yet is
skipped. TabPFN-3.5 ran on a GPU and every other model on CPU, so a row of 3.5 is
compared with the others by its medians only.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator, PercentFormatter

from analysis.dose_ensemble import generator_law
from analysis.supplement_figures import GREY, INK, LABELS, ORDER, TFM, load_natural
from analysis.tabpfn_y import leverage_measures, loss_measures

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/h3_repair'
OUT = ROOT / 'thesis/figures'
# Three hues, validated together on white with the dataviz validator (all pairs): the
# cap, the log and the conditional cap keep these colours in every figure. The aqua is
# one step darker than the palette's light slot, which falls below 3:1 on white. raw is
# ink.
CAP, LOG, CCLIP, RAW = '#2a78d6', '#eb6834', '#199e70', '#555555'
ARMS = {'raw': ('bez izmene', RAW), 'clip': ('odsecanje', CAP), 'log': ('logaritam', LOG),
        'cclip': ('uslovno odsecanje', CCLIP)}
# The conditional cap ran at one member only, on the seeds of the segment run.
CCLIP_SEG = 'legit_segment_cclip.csv'
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
    cclip = load(CCLIP_SEG)
    for label, _, seg_file, model in CONFIGS:
        d = load(seg_file)
        if d is None:
            print(f'segment_tail: {seg_file} is missing, figure skipped')
            return
        d = d[(d.model == model) & (d.seg_scale == 20.0) & (d.sd_shift_target == 1.0)]
        if cclip is not None and (d.n_est == 1).all():
            c = cclip[(cclip.model == model) & (cclip.variant == 'cclip')
                      & (cclip.seg_scale == 20.0) & (cclip.sd_shift_target == 1.0)]
            assert c.empty or set(c.seed) == set(d.seed)
            d = pd.concat([d, c])
        for arm in ARMS:
            for part in ('out', 'seg'):
                v = d[d.variant == arm][f'qratio999_{part}'].to_numpy()
                if arm == 'cclip' and len(v) == 0:
                    continue
                assert len(v) == 20
                records.append(dict(config=label, arm=arm, part=part, median=np.median(v),
                                    q25=np.quantile(v, .25), q75=np.quantile(v, .75)))
    data = pd.DataFrame(records)
    labels = [c[0] for c in CONFIGS]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6), sharey=True, sharex=True,
                             layout='constrained')
    offsets = dict(zip(ARMS, (-.27, -.09, .09, .27)))
    for ax, part, title in zip(axes, ('out', 'seg'), ('van segmenta', 'u segmentu')):
        for i, label in enumerate(labels):
            for arm, (name, colour) in ARMS.items():
                r = data[(data.config == label) & (data.arm == arm) & (data.part == part)]
                if r.empty:
                    continue
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
    fig.legend(loc='outside upper center', ncol=4, handletextpad=.3, columnspacing=1.4)
    fig.supxlabel('Predviđeni Q(0,999) prema pravom, čist kontekst, segment ×20 '
                  '(medijana i interkvartilni raspon, 20 semena)', fontsize=10.5)
    save(fig, 'segment_rep', data)


def target_transform():
    """TabPFN-3 at a shift or dose of 20: one member, eight at the default and eight with
    every member on the untransformed target (rules Y0 to Y3 of 28.10.2026), one dot per
    seed. Q and R come from the dose design (20 seeds), G from the leverage sweep (5)."""
    loss, lev = loss_measures(), leverage_measures()
    if len(loss) < 3 or len(lev) < 3:
        print('target_transform: a file of 28.10.2026 is missing, figure skipped')
        return
    arms = [('1', '1 član'), ('8', '8 članova'), ('8n', '8 članova,\nbez transformacije cilja')]
    panels = [('Q', 'Q(0,99) na test redovima'), ('R', 'očekivani gubitak na 0,99'),
              ('G', 'globalna promena Q(0,99)')]
    records = []
    for a, _ in arms:
        for m, v in (('Q', loss[a]['Q']), ('R', loss[a]['R']), ('G', lev[a][20.0])):
            records += [dict(arm=a, measure=m, value=x) for x in v.to_numpy()]
    data = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6), sharey=True, layout='constrained')
    for ax, (m, title) in zip(axes, panels):
        for i, (a, _) in enumerate(arms):
            v = data[(data.arm == a) & (data.measure == m)].value.to_numpy()
            assert len(v) and np.isfinite(v).all()
            ax.scatter(v, i + np.linspace(-.15, .15, len(v)), s=16, color=INK, alpha=.45,
                       linewidths=0, zorder=3)
            ax.scatter([np.median(v)], [i], s=44, marker='D', color=INK, edgecolors='white',
                       linewidths=.6, zorder=4)
        ax.axvline(0, color=INK, lw=.8, zorder=2)
        ax.set_yticks(range(len(arms)), [t for _, t in arms])
        ax.set_ylim(len(arms) - .5, -.5)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color='#E8E8E8', linewidth=.5, zorder=0)
        ax.set_title(title, fontsize=10.5, loc='left', color=INK)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.supxlabel('Promena prema čistom kontekstu pri pomeraju 20, TabPFN-3 '
                  '(tačka po semenu, romb je medijana)', fontsize=10.5)
    save(fig, 'transformacija_cilja', data)


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
        # The expected loss under the generator's capped law, the reference from 22.10.
        d = generator_law(d[d.variant == 'raw'])
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


def natural():
    """freMTPL2sev: the predicted Q(0.99) with the largest claim against without it, and
    the change in 0.99 loss after capping, one row per model, one dot per subsample."""
    d = load_natural()
    ratio = d[(d.arm_c == 'raw') & (d.bin == '4.0-inf')].assign(value=lambda x: 1 + x.d_q99)
    idx = ['dataset', 'model', 'bin', 'repeat', 'n_est', 'n_fit', 'n_test']
    raw = d[d.arm_c == 'raw'].set_index(idx).pb99
    cap = d[(d.arm_c == 'clip_200') & d.bin.isin(['2.0-4.0', '4.0-inf'])].set_index(idx)
    cap = cap.assign(value=cap.pb99 / raw.reindex(cap.index) - 1).reset_index()
    data = pd.concat([ratio.assign(panel='ratio'), cap.assign(panel='cap')])[
        ['panel', 'model', 'bin', 'repeat', 'value']]
    models = [m for m in ORDER if m in set(data.model)]
    labels = [LABELS[m] + ('' if m in TFM else ' (kontrola)') for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.9), sharey=True, layout='constrained')
    for ax, panel, colour, title in zip(axes, ('ratio', 'cap'), (INK, CAP),
                                        ('najveća šteta zadržana prema uklonjenoj',
                                         'odsecanje prema bez izmene')):
        for i, m in enumerate(models):
            v = data[(data.panel == panel) & (data.model == m)].value.to_numpy()
            c = colour if m in TFM else GREY
            ax.scatter(v, i + np.linspace(-.15, .15, len(v)), s=16, color=c, alpha=.5,
                       linewidths=0, zorder=3)
            ax.scatter([np.median(v)], [i], s=44, marker='D', color=c, edgecolors='white',
                       linewidths=.6, zorder=4)
        ax.set_yticks(range(len(models)), labels)
        ax.set_ylim(len(models) - .5, -.5)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color='#E8E8E8', linewidth=.5, zorder=0)
        ax.axhline(len([m for m in models if m in TFM]) - .5, color='#DDDDDD', lw=.65)
        ax.set_title(title, fontsize=10.5, loc='left', color=INK)
    axes[0].axvline(1, color=INK, lw=.8, zorder=2)
    axes[0].set_xscale('log')
    axes[0].set_xlim(.3, 3.3)
    axes[0].set_xticks([.5, 1, 2, 3], ['0,5×', '1×', '2×', '3×'])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set_xlabel('Q(0,99), pomeraj ≥4\n5 poduzoraka', fontsize=10)
    axes[1].axvline(0, color=INK, lw=.8, zorder=2)
    axes[1].set_xlim(-1, .3)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].set_xlabel('gubitak na 0,99, pomeraj ≥2\n10 poduzoraka', fontsize=10)
    save(fig, 'prirodni_ekstremi', data)


if __name__ == '__main__':
    unit_error()
    segment_tail()
    target_transform()
    natural()
    dose()
    dose('en')
