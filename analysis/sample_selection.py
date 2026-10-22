"""Does the selection of leverage subsamples starve their test rows of large claims?

`leverage.subsamples` accepts a random permutation when its context has a large sd shift,
so the claims that cause the shift sit in the context and cannot be test rows. A test set
without the population's largest claims would hide the cost of capping exactly where it
falls. This redraws the subsamples of `clip_context_real.py` (seed 31337, same sizes and
bins), counts the large claims in each test set against the clean bin and against
sampling at random, and relates the cap's gain at 0.99 to those counts. No model is run.

    python -m analysis.sample_selection
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import datasets, leverage, paths, tables

N_FIT, N_TEST, N_PER_BIN, MAX_ATTEMPTS = 2000, 1000, 5, 4000
MODELS = ("TabPFN-V3", "TabICLv2", "GBM")


def main():
    _, y = datasets.load("freMTPL2sev")
    y = y[np.isfinite(y) & (y > 0)]
    cut = {lv: float(np.quantile(y, lv)) for lv in (0.99, 0.999)}
    top10 = set(np.argsort(y)[-10:])
    found = leverage.subsamples(y, np.random.default_rng(31337), N_FIT, N_TEST, N_PER_BIN,
                                MAX_ATTEMPTS)
    rows = []
    for k, subs in found.items():
        for j, (idx, shift, _) in enumerate(subs):
            ctx, te = idx[:N_FIT], idx[N_FIT:]
            rows.append(dict(bin=leverage.bin_name(k), repeat=j, sd_shift=shift,
                             test_max_over_q999=float(y[te].max() / cut[0.999]),
                             test_above_q99=int((y[te] > cut[0.99]).sum()),
                             test_above_q999=int((y[te] > cut[0.999]).sum()),
                             top10_in_context=len(top10 & set(ctx.tolist())),
                             top10_in_test=len(top10 & set(te.tolist()))))
    s = pd.DataFrame(rows)
    print(f"freMTPL2sev, {len(y)} positive claims; at random a test set of {N_TEST} holds "
          f"{N_TEST * 0.01:.0f} claims above Q(0.99) and {N_TEST * 0.001:.0f} above Q(0.999)")
    print(s.groupby("bin")[["test_above_q99", "test_above_q999", "test_max_over_q999",
                            "top10_in_context", "top10_in_test"]].agg(["median", "min", "max"])
          .round(2).to_string())

    d = tables.ok_rows(pd.read_csv(paths.result("h3_repair/clip_context_real.csv")))
    d = d[(d.dataset == "freMTPL2sev") & d.model.isin(MODELS)]
    w = d.pivot_table(index=["model", "bin", "repeat"], columns=["arm", "clip_c"], values="pb99")
    gain = (w[("clip", 200.0)] / w[("raw", 0.0)] - 1).rename("cap_gain").reset_index()
    g = gain.merge(s, on=["bin", "repeat"])
    g = g[g.bin.isin(["2.0-4.0", "4.0-inf"])]
    print("\ncap's change in pinball at 0.99 against the test rows' large claims, sd shift >= 2")
    for m, part in g.groupby("model"):
        rho, p = spearmanr(part.cap_gain, part.test_above_q99)
        print(f"  {m:9s} n={len(part)}  median {part.cap_gain.median():+.1%}  Spearman with "
              f"test claims above Q(0.99): {rho:+.2f} (p {p:.2f})")
        rich = part.test_above_q99 >= part.test_above_q99.median()
        print(f"            richer half {part[rich].cap_gain.median():+.1%}, "
              f"poorer half {part[~rich].cap_gain.median():+.1%}")


if __name__ == "__main__":
    main()
