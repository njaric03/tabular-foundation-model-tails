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
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, PercentFormatter
from matplotlib.transforms import blended_transform_factory

from common import tables
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
# The ensemble size where no repair is drawn: red for one member, the configuration that
# moves, and grey for the default eight, which stays flat. The red is away from the three
# repairs' hues; trees stay grey as controls.
ONE, EIGHT = '#c92a2a', '#8a8a8a'
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
                     'axes.labelsize': 11, 'axes.unicode_minus': True,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888888', 'axes.linewidth': .6,
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


def rows(ax, labels, control=True):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color='#E4E4E4', linewidth=.5, zorder=0)
    # The tree control sits apart from the foundation models.
    if control:
        ax.axhline(len(labels) - 1.5, color='#CCCCCC', lw=.65)


# One mark system for every figure: a repeat (seed, table or subsample) is a small mark,
# a circle at one member and a square at four or eight, as the lines of the dose figure;
# the median over repeats is a larger diamond in a darker shade of the same colour.
def shape(label):
    return 's' if any(f'{n} član' in label for n in (4, 8)) else 'o'


def dark(colour, f=.7):
    return tuple(f * c for c in to_rgb(colour))


def repeats(ax, v, i, colour, marker='o'):
    ax.scatter(v, i + np.linspace(-.17, .17, len(v)), s=22 if marker == 'o' else 18,
               marker=marker, color=colour, alpha=.6, linewidths=0, zorder=3)


def median(ax, x, i, colour, s=64):
    ax.scatter([x], [i], s=s, marker='D', color=dark(colour), edgecolors='white',
               linewidths=.8, zorder=4)


def mark(marker, colour, label, ms=6.5, mew=.6):
    """A legend entry for one kind of mark."""
    return Line2D([], [], ls='', marker=marker, color=colour, ms=ms, mec=colour if
                  marker == '|' else 'white', mew=mew, label=label)


def key(fig, handles, ncol=None):
    fig.legend(handles=handles, loc='outside upper center', ncol=ncol or len(handles),
               handletextpad=.3, columnspacing=1.5)


# Part one: scale and shape pair only at the same member count. scale_of_x.csv ran at four
# members, and at one member the scale exists for TabPFN-V3 only (the reseed file).
H1 = ROOT / 'results/h1_shape_vs_scale'
H1_ROWS = [  # label, scale file, shape file, model, colour
    ('TabPFN-3, 1 član', 'scale_of_x_tabpfn_reseed_nest1.csv', 'shape_of_x_nest1.csv',
     'TabPFN-V3', ONE),
    ('TabPFN-3, 4 člana', 'scale_of_x.csv', 'shape_of_x_20.csv', 'TabPFN-V3', EIGHT),
    ('TabPFN-3.5, 1 član', 'scale_of_x_tabpfn35.csv', 'shape_of_x_tabpfn35.csv',
     'TabPFN-v3.5', ONE),
    ('TabICLv2, 4 člana', 'scale_of_x.csv', 'shape_of_x_20.csv', 'TabICLv2', EIGHT),
    ('GBM (kontrola)', 'scale_of_x.csv', 'shape_of_x_20.csv', 'GBM', GREY),
]


def scale_shape():
    """Share of the true change in the conditional scale and in the tail shape that the
    prediction follows: one dot per seed and a bar for the share pooled over seeds, as in
    the claim ledger, so the diamond keeps meaning a median. The EVT oracle on the same 2000
    rows is marked over the shape panel."""
    records = []
    for label, sc_file, sh_file, model, _ in H1_ROWS:
        sc = tables.ok_rows(pd.read_csv(H1 / sc_file))
        sh = tables.ok_rows(pd.read_csv(H1 / sh_file))
        sc, sh = sc[sc.model == model], sh[sh.model == model]
        assert sc.n_est.nunique() == 1 and set(sc.n_est) == set(sh.n_est)
        runs = tables.share_by_run(sh, sc)
        pooled = dict(scale=tables.scale_share_by_model(sc).share[model],
                      shape=tables.slope_by_model(sh).share[model])
        for part in ('scale', 'shape'):
            records += [dict(row=label, part=part, kind='seed', value=v)
                        for v in runs[f'{part}_share']]
            records.append(dict(row=label, part=part, kind='pooled', value=pooled[part]))
    o = pd.read_csv(H1 / 'oracle_xi_of_x.csv')
    g = o[o.n == 2000]
    oracle = (g.oracle_r_2 - g.oracle_r_0).mean() / (g.true_2 - g.true_0).mean()
    data = pd.DataFrame(records)
    labels = [r[0] for r in H1_ROWS]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), sharey=True, sharex=True,
                             layout='constrained')
    for ax, part, title in zip(axes, ('scale', 'shape'), ('skala', 'oblik repa')):
        for i, (label, *_, colour) in enumerate(H1_ROWS):
            d = data[(data.row == label) & (data.part == part)]
            v = d[d.kind == 'seed'].value.to_numpy()
            assert len(v) == 20 and np.isfinite(v).all()
            repeats(ax, v, i, colour, shape(label))
            ax.scatter(d[d.kind == 'pooled'].value, [i], s=230, marker='|', color=dark(colour),
                       linewidths=2.6, zorder=4)
        ax.axvline(1, color=INK, lw=.9, zorder=2)
        ax.axvline(0, color='#BBBBBB', lw=.6, zorder=1)
        rows(ax, labels)
        ax.set_title(title, fontsize=11, loc='left', color=INK)
        ax.set_xlim(-.6, 1.2)
        ax.set_xticks([-.5, 0, .5, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    # The oracle as a mark on the top edge of the shape panel, at its share.
    top = blended_transform_factory(axes[1].transData, axes[1].transAxes)
    axes[1].scatter([oracle], [1], s=46, marker='v', color=INK, transform=top, clip_on=False,
                    zorder=5)
    key(fig, [mark('o', ONE, 'seme, 1 član'), mark('s', EIGHT, 'seme, 4 člana', ms=6),
              mark('o', GREY, 'seme, GBM'),
              mark('|', INK, 'sva semena zajedno', ms=11, mew=2.2),
              mark('v', INK, f'EVT orakl, 2000 redova: {oracle:.0%}')], ncol=3)
    fig.supxlabel('Udeo prave promene koji predviđanje prati', fontsize=10.5)
    save(fig, 'skala_oblik', data.assign(oracle=oracle))


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
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), sharey=True, sharex=True,
                             layout='constrained')
    for ax, arm in zip(axes, ('clip', 'log')):
        name, colour = ARMS[arm]
        for i, label in enumerate(labels):
            v = data[(data.config == label) & (data.arm == arm)].change.to_numpy()
            assert len(v) == 5 and np.isfinite(v).all()
            repeats(ax, v, i, colour, shape(label))
            median(ax, np.median(v), i, colour)
        ax.axvline(0, color=INK, lw=.9, zorder=2)
        rows(ax, labels)
        ax.set_title(name, fontsize=11, loc='left', color=INK)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].text(0, -.2, '← niži gubitak', transform=axes[0].transAxes, fontsize=10,
                 color=INK)
    key(fig, [mark('o', RAW, 'tabela, 1 član'), mark('s', RAW, 'tabela, 8 članova', ms=6),
              mark('D', dark(RAW), 'medijana preko tabela')])
    fig.supxlabel('Promena gubitka na 0,99 prema izvornom modelu, '
                  'vidljiva greška jedinice ×100', fontsize=10.5)
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
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.9), sharey=True, sharex=True,
                             layout='constrained')
    offsets = dict(zip(ARMS, (-.3, -.1, .1, .3)))
    for ax, part, title in zip(axes, ('out', 'seg'), ('van segmenta', 'u segmentu')):
        for i, label in enumerate(labels):
            for arm, (name, colour) in ARMS.items():
                r = data[(data.config == label) & (data.arm == arm) & (data.part == part)]
                if r.empty:
                    continue
                y = i + offsets[arm]
                ax.plot([r.q25.item(), r.q75.item()], [y, y], color=colour, lw=2.6,
                        solid_capstyle='round', zorder=3)
                median(ax, r['median'].item(), y, colour, s=40)
        # The exact quantile is the reference every interval is read against.
        ax.axvline(1, color=INK, lw=1.4, zorder=2)
        ax.text(1.07, 1.01, '1× = tačan kvantil', fontsize=10, color=INK, va='bottom',
                transform=blended_transform_factory(ax.transData, ax.transAxes))
        rows(ax, labels)
        ax.set_ylim(len(labels) - .5, -.6)
        ax.set_title(title, fontsize=11, loc='left', color=INK)
        ax.set_xscale('log')
        ax.set_xlim(.05, 5)
        ax.set_xticks([.1, .5, 1, 2, 4], ['0,1×', '0,5×', '1×', '2×', '4×'])
        ax.xaxis.set_minor_locator(NullLocator())
    # Room between the panels for the reference label of the left one.
    fig.get_layout_engine().set(wspace=.08)
    key(fig, [Line2D([], [], color=colour, lw=2.6, marker='D', ms=6, mfc=dark(colour),
                     mec='white', mew=.6, label=name) for name, colour in ARMS.values()])
    fig.supxlabel('Predviđeni Q(0,999) prema pravom, čist kontekst, segment ×20\n'
                  'romb je medijana, linija interkvartilni raspon preko 20 semena',
                  fontsize=10.5)
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
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 3.1), sharey=True, layout='constrained')
    colours = {'1': ONE, '8': EIGHT, '8n': EIGHT}
    for ax, (m, title) in zip(axes, panels):
        for i, (a, label) in enumerate(arms):
            v = data[(data.arm == a) & (data.measure == m)].value.to_numpy()
            assert len(v) and np.isfinite(v).all()
            repeats(ax, v, i, colours[a], shape(label))
            median(ax, np.median(v), i, colours[a], s=56)
        ax.axvline(0, color=INK, lw=.9, zorder=2)
        rows(ax, [t for _, t in arms], control=False)
        ax.set_title(title, fontsize=10.5, loc='left', color=INK)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    key(fig, [mark('o', ONE, 'seme, 1 član'), mark('s', EIGHT, 'seme, 8 članova', ms=6),
              mark('D', dark(GREY), 'medijana preko semena')])
    fig.supxlabel('Promena prema čistom kontekstu pri pomeraju 20, TabPFN-3', fontsize=10.5)
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
    styles = {1: dict(color=ONE, ls='-', marker='o', label=text['one']),
              8: dict(color=EIGHT, ls='--', marker='s', label=text['eight'])}
    for r, level in enumerate((.99, .999)):
        for c, model in enumerate(('TabPFN-3', 'TabPFN-3.5')):
            ax = axes[r, c]
            for n, style in styles.items():
                p = data[(data.model == model) & (data.members == n) & (data.level == level)]
                ax.plot(range(len(shifts)), p['median'], lw=2, ms=6,
                        label=style['label'] if r == 0 and c == 0 else None,
                        **{k: v for k, v in style.items() if k != 'label'})
            ax.axhline(0, color=INK, lw=.9, zorder=0)
            ax.grid(axis='y', color='#E4E4E4', lw=.5, zorder=0)
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
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.3), sharey=True, layout='constrained')
    # Every model here ran at one member. Red stands for one member only where the count
    # is compared, so the left panel is the raw model in dark grey and the right the cap.
    for ax, panel, colour, title in zip(axes, ('ratio', 'cap'), (RAW, CAP),
                                        ('najveća šteta zadržana prema uklonjenoj',
                                         'odsecanje prema bez izmene')):
        for i, m in enumerate(models):
            v = data[(data.panel == panel) & (data.model == m)].value.to_numpy()
            c = colour if m in TFM else GREY
            repeats(ax, v, i, c)
            median(ax, np.median(v), i, c, s=56)
        rows(ax, labels, control=False)
        ax.axhline(len([m for m in models if m in TFM]) - .5, color='#CCCCCC', lw=.65)
        ax.set_title(title, fontsize=10.5, loc='left', color=INK)
    axes[0].axvline(1, color=INK, lw=.9, zorder=2)
    axes[0].set_xscale('log')
    axes[0].set_xlim(.3, 3.3)
    axes[0].set_xticks([.5, 1, 2, 3], ['0,5×', '1×', '2×', '3×'])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set_xlabel('Q(0,99), pomeraj ≥4\n5 poduzoraka', fontsize=10)
    axes[1].axvline(0, color=INK, lw=.9, zorder=2)
    axes[1].set_xlim(-1, .3)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].set_xlabel('gubitak na 0,99, pomeraj ≥2\n10 poduzoraka', fontsize=10)
    key(fig, [mark('o', RAW, 'poduzorak, 1 član'),
              mark('D', dark(RAW), 'medijana preko poduzoraka')])
    save(fig, 'prirodni_ekstremi', data)


if __name__ == '__main__':
    scale_shape()
    unit_error()
    segment_tail()
    target_transform()
    natural()
    dose()
    dose('en')
