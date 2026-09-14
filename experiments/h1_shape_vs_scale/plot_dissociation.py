# -*- coding: utf-8 -*-
"""The thesis figure of the dissociation, written to figures/dissociation.{pdf,png}.

Per model a dumbbell between two shares of the true change: the scale s(x), from
terciles of the linear predictor, and the tail shape xi(x), from terciles of x4. The
length of the bar is the dissociation. For models also measured at n_est=1 an arrow
runs from the shape at n_est=4 to the shape at n_est=1, since Vincentization lowers the
implied xi and the ensemble alone widens the gap (TabPFN 42% to 19%). The whiskers are
the standard error of the mean over seeds. The labels are in Serbian, the language of
the thesis.

    python -u experiments/h1_shape_vs_scale/plot_dissociation.py
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import paths  # noqa: E402

# One hue at two steps, checked for monotone lightness and contrast against the surface.
# Model identity is carried by the row, not by colour.
COLOUR_SCALE = "#86b6ef"     # the half that is already good, pushed back
COLOUR_SHAPE = "#1c5cab"     # the finding, emphasised
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

CONTROLS = {"GBM", "XGB"}
ORDER = ["GBM", "TabICLv2", "EXAONE", "TabPFN-V3", "XGB"]

SHAPE_N4 = ["shape_of_x_20.csv", "shape_of_x_exaone_20.csv"]
SHAPE_N1 = ["shape_of_x_nest1.csv", "shape_of_x_exaone_nest1.csv"]
SCALE_N4 = ["scale_of_x.csv", "scale_of_x_exaone_20.csv"]


def read(names: list[str]) -> pd.DataFrame:
    missing = [n for n in names if not paths.result(n).exists()]
    if missing:
        raise FileNotFoundError(f"missing results: {missing}")
    return pd.concat([pd.read_csv(paths.result(n)) for n in names], ignore_index=True)


def shape_share(d: pd.DataFrame) -> pd.Series:
    """Implied-xi slope across terciles over the true slope, per (model, seed)."""
    p = d.pivot_table(index=["model", "seed"], columns="tercile",
                      values=["xi_implied", "xi_true"])
    return ((p[("xi_implied", 3)] - p[("xi_implied", 1)]) /
            (p[("xi_true", 3)] - p[("xi_true", 1)]))


def scale_share(d: pd.DataFrame) -> pd.Series:
    """Log median ratio of the outer terciles over the true one, per (model, seed)."""
    p = d.pivot_table(index=["model", "seed"], columns="tercile",
                      values=["median_model", "median_true"])
    return (np.log(p[("median_model", 3)] / p[("median_model", 1)]) /
            np.log(p[("median_true", 3)] / p[("median_true", 1)]))


def summarise(s: pd.Series) -> pd.DataFrame:
    """Per model: mean, sd over seeds, standard error of the mean, number of seeds."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    g = s.groupby(level=0)
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(),
                         "se": g.std() / np.sqrt(g.count()), "n": g.count()})


def main() -> int:
    shape4 = summarise(shape_share(read(SHAPE_N4)))
    shape1 = summarise(shape_share(read(SHAPE_N1)))
    scale4 = summarise(scale_share(read(SCALE_N4)))
    names = [m for m in ORDER if m in shape4.index and m in scale4.index]

    print("=== share of the true change (n_est=4, 20 seeds) ===\n")
    print(f"{'model':<12}{'scale':>8}{'se':>7}{'shape':>9}{'se':>7}{'sd':>8}{'shape n_est=1':>15}")
    for m in names:
        n1 = f"{shape1.loc[m, 'mean']:.0%}" if m in shape1.index else "--"
        print(f"{m:<12}{scale4.loc[m, 'mean']:>8.0%}{scale4.loc[m, 'se']:>7.2f}"
              f"{shape4.loc[m, 'mean']:>9.0%}{shape4.loc[m, 'se']:>7.2f}"
              f"{shape4.loc[m, 'sd']:>8.2f}{n1:>15}")

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "xtick.color": INK_3, "ytick.color": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig, ax = plt.subplots(figsize=(8.0, 0.82 * len(names) + 2.5))
    rows = list(range(len(names)))[::-1]

    for y, m in zip(rows, names):
        xs, xo = scale4.loc[m, "mean"], shape4.loc[m, "mean"]
        ax.plot([xo, xs], [y, y], color=COLOUR_SCALE, lw=3.2, solid_capstyle="round",
                zorder=2, alpha=0.55)
        for x, se, colour in ((xs, scale4.loc[m, "se"], COLOUR_SCALE),
                              (xo, shape4.loc[m, "se"], COLOUR_SHAPE)):
            ax.plot([x - se, x + se], [y, y], color=colour, lw=1.2, alpha=0.9, zorder=3)
        if m in shape1.index and abs(shape1.loc[m, "mean"] - xo) > 0.02:
            x1 = shape1.loc[m, "mean"]
            ax.annotate("", xy=(x1, y), xytext=(xo, y),
                        arrowprops=dict(arrowstyle="-|>", color=INK_3, lw=1.0,
                                        alpha=0.55, shrinkA=8, shrinkB=1))
            ax.plot(x1, y, marker="o", ms=7, mfc=SURFACE, mec=INK_3, mew=1.3, zorder=4)
            ax.annotate(f"{x1:.0%} pri n_est=1", (x1, y), textcoords="offset points",
                        xytext=(0, 13), fontsize=8.2, color=INK_3, ha="center")
        ax.plot(xo, y, marker="o", ms=11, color=COLOUR_SHAPE, mec=SURFACE, mew=2, zorder=5)
        ax.plot(xs, y, marker="o", ms=11, color=COLOUR_SCALE, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"{xo:.0%}", (xo, y), textcoords="offset points", xytext=(-11, -3.5),
                    fontsize=9.5, color=COLOUR_SHAPE, ha="right", weight="bold")
        ax.annotate(f"{xs:.0%}", (xs, y), textcoords="offset points", xytext=(11, -3.5),
                    fontsize=9.5, color=INK_2, ha="left")

    ax.set_yticks(rows)
    ax.set_yticklabels([m + ("  (etalon)" if m in CONTROLS else "") for m in names],
                       fontsize=10.5)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.axvline(1.0, color=BASELINE, lw=1.0, ls="--", zorder=1)
    ax.annotate("savršeno praćenje", (1.0, len(names) - 0.45), fontsize=8.5,
                color=INK_3, ha="center", va="bottom")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel("udeo prave promene koji model uhvati", color=INK_2, labelpad=8)
    fig.suptitle("Modeli prate nivo, oblik repa im je zamrznut",
                 x=0.012, y=0.985, ha="left", fontsize=13, color=INK)
    fig.text(0.012, 0.925,
             "svaka spona je jaz između praćenja SKALE s(x) i praćenja OBLIKA repa ξ(x); "
             "20 seedova, n_est=4, crte = standardna greška sredine",
             ha="left", fontsize=8.8, color=INK_2)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=COLOUR_SHAPE, mec=SURFACE, mew=1.5,
               ms=9, label="OBLIK repa  ξ(x)"),
        Line2D([], [], marker="o", ls="", color=COLOUR_SCALE, mec=SURFACE, mew=1.5,
               ms=9, label="SKALA  s(x)"),
        Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK_3, mew=1.3,
               ms=7, label="oblik pri n_est=1 (bez vincentizacije)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
        fontsize=9, handletextpad=0.5, columnspacing=1.8)

    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    for ext in ("pdf", "png"):
        out = paths.figure(f"dissociation.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
