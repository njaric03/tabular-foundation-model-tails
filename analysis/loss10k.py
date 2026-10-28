# -*- coding: utf-8 -*-
"""Verdicts of the rules of 27.10.2026: the repairs' loss at 10 000 rows, raw against
clip_200 and log for TabPFN-V3 at one and eight members, on the expected pinball under
the generator's law.

Reads the two outputs of `run/robustness_2710.sh loss10k`; a file that is not there yet
is skipped, and a partial one is judged on the seeds it holds. A task is one seed, ten
in all, at XI=0.7.

    python -m analysis.loss10k
"""
from analysis.confirm_ensemble_log import verdict
from analysis.dose_ensemble import arm, generator_law, load, rise, wide

FILES = {1: "clip_context_n10k_nest1.csv", 8: "clip_context_n10k_nest8.csv"}


def main():
    for n, f in FILES.items():
        d = load(f)
        if d is None:
            continue
        g = generator_law(d)
        w99, w999 = wide(g, "epb99"), wide(g, "epb999")
        full = [(s, a) for s in (1.0, 20.0) for a in ("raw", "clip", "log")]
        if any(c not in w99.columns for c in full):
            print(f"\n{n} member(s): no seed has every arm yet")
            continue
        w99, w999 = w99.dropna(subset=full), w999.dropna(subset=full)
        k = len(w99)
        print(f"\n{n} member(s), {k} complete seeds")

        r = arm(w99, 20.0, "clip", "raw")
        verdict("P1", (r < 0).sum() >= 8, f"clip_200 below raw at shift 20 in "
                                          f"{(r < 0).sum()} of {k}, median {r.median():+.1%}")
        c = arm(w99, 1.0, "clip", "raw")
        verdict("P2", abs(c.median()) <= 0.02, f"clean context: median {c.median():+.2%}")
        r, c = arm(w99, 20.0, "log", "raw"), arm(w99, 1.0, "log", "raw")
        verdict("P3", (r < 0).sum() >= 8 and c.median() <= 0.02,
                f"log below raw at shift 20 in {(r < 0).sum()} of {k}, median "
                f"{r.median():+.1%}; clean context: median {c.median():+.2%}")

        # Reported without a rule.
        for a in ("clip", "log"):
            r, c = arm(w99, 20.0, a, "raw"), arm(w999, 20.0, a, "raw")
            print(f"     {a:4s} over raw at shift 20: 0.99 range {r.min():+.1%} to "
                  f"{r.max():+.1%}; 0.999 {(c < 0).sum()} of {k} below, median "
                  f"{c.median():+.1%}")
        print(f"     raw at shift 20 over the clean context: 0.99 "
              f"{rise(w99, 20.0).median():+.1%}, 0.999 {rise(w999, 20.0).median():+.1%}")
        q = wide(d, "q99").reindex(w99.index)
        q = q[(20.0, "raw")] / q[(1.0, "raw")] - 1
        print(f"     raw Q(0.99), shift 20 over 1: median {q.median():+.1%}, "
              f"range {q.min():+.1%} to {q.max():+.1%}")
        m = d[(d.variant == "clip") & (d.sd_shift_target == 1.0)].n_clipped
        print(f"     values the cap binds on a clean context: median {m.median():g}, "
              f"range {m.min():g} to {m.max():g}")


if __name__ == "__main__":
    main()
