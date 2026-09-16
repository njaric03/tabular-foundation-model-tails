# -*- coding: utf-8 -*-
"""Tables of the prior dumps, in a fixed order, with arm A1 applied and the target encoded.

    tables = Tables(".cache/tfmp/dumps/a.h5,.cache/tfmp/dumps/b.h5", args)
    X, y, k, split, info = tables.raw(t)       # one table, A1 applied if the arm is A1
    X, y_enc, k, info = tables.encoded(t, split)

`args` carries `arm` ("A0B0", "A1B1", ...), `seed`, `preload` (GB), `target_clip` and the
A1 settings (`p_tail`, `p_contam`, `xi_lo`, `xi_hi`, `w_sd`, `shift_lo`, `shift_hi`). The
A1 draws of table t depend only on (seed, t), so two arms of one seed see the same tables
and the same draws. Shared by the nanoTabPFN pre-training and the TabICLv2 fine-tuning.
"""
from __future__ import annotations

import time

from common.adapters.nanotabpfn import encode, encode_stats
from common.prior_arms import apply_a1


class Tables:
    """Tables of one or more dumps, in the order given, with arm A1 applied and the target
    encoded. Table t is the t-th table of the concatenation, wrapping at the end."""

    def __init__(self, paths, args):
        import h5py
        import numpy as np
        self.files = [h5py.File(p, "r") for p in paths.split(",")]
        self.offsets, n = [], 0
        for f in self.files:
            self.offsets.append(n)
            n += f["y"].shape[0]
        self.n = n
        self.max_features = max(f["X"].shape[2] for f in self.files)
        self.max_split = max(int(f["train_test_split_index"][:].max()) for f in self.files)
        # Reading a table out of HDF5 costs more than the step it feeds: at 50 rows the GPU
        # held 98 MB and ran at 37 tables per second, the same rate as at 256 rows. Dumps
        # that fit in memory are read once.
        self.mem = None
        size = sum(f["X"].size * 4 + f["y"].size * 4 for f in self.files)
        if args.preload and size <= args.preload * 2 ** 30:
            t0 = time.time()
            self.mem = dict(
                X=np.concatenate([f["X"][:] for f in self.files]),
                y=np.concatenate([f["y"][:] for f in self.files]),
                num_features=np.concatenate([f["num_features"][:] for f in self.files]),
                split=np.concatenate([f["train_test_split_index"][:] for f in self.files]))
            print(f"preloaded {n} tables, {size / 2 ** 30:.2f} GB, in "
                  f"{time.time() - t0:.0f}s", flush=True)
        self.args = args
        self.a1 = args.arm.startswith("A1")
        self.encoding = "robust" if args.arm.endswith("B1") else "meansd"

    def locate(self, t):
        """(file, index within it) of table t of the concatenation."""
        t %= self.n
        for f, start in zip(reversed(self.files), reversed(self.offsets)):
            if t >= start:
                return f, t - start

    def raw(self, t):
        import numpy as np
        t %= self.n
        if self.mem is not None:
            k = int(self.mem["num_features"][t])
            split = int(self.mem["split"][t])
            X = self.mem["X"][t, :, :k]
            y = self.mem["y"][t].astype(np.float64)
        else:
            f, i = self.locate(t)
            k = int(f["num_features"][i])
            split = int(f["train_test_split_index"][i])
            X = f["X"][i, :, :k].astype(np.float32)
            y = f["y"][i].astype(np.float64)
        info = {}
        if self.a1:
            a = self.args
            rng = np.random.default_rng([a.seed, t])
            y, info = apply_a1(X, y, k, split, rng, p_tail=a.p_tail, p_contam=a.p_contam,
                               xi_lo=a.xi_lo, xi_hi=a.xi_hi, w_sd=a.w_sd,
                               lo=a.shift_lo, hi=a.shift_hi)
        return X, y, k, split, info

    def encoded(self, t, split):
        """(X, y encoded by the context statistics, k) with the context cut at `split`."""
        X, y, k, own_split, info = self.raw(t)
        centre, scale = encode_stats(y[:split], self.encoding)
        return X, encode(y, centre, scale, self.args.target_clip), k, info
