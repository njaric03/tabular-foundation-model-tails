# -*- coding: utf-8 -*-
"""Generate the base prior dump for the pre-training experiment, on TabICL's SCM prior.

`python -m tfmplayground.external_priors --lib tabicl` cannot produce this dump, for two
reasons found in the installed code:

  - its `TabICLPriorDataLoader` never passes `regression=True`. tabicl 2.1.1 has no
    regression prior at all, and in 2.2.0 `max_classes=0` without the flag either raises
    in `np.random.randint(2, 1)` or labels the table with two classes;
  - it keeps the feature count of the first table of a batch for every table in it,
    while tabicl samples the count per group of `batch_size_per_gp` tables.

Every table is written with its own feature count and split, under the keys
`PriorDumpDataLoader` reads. Rows are fixed at `--seq_len` for every table, because
`make_global_bucket_edges` and the dump loader would otherwise read the zero padding of a
shorter table as targets.

tabicl's `Reg2Cls` clamps every regression target at four sd from its mean, both taken
over the values inside a first four-sd cut, before it standardises it
(`outlier_removing(y, threshold=4)`). On the 14000 tables of seed 2026, 3.2% still reach
beyond four plain sd (up to 22.6, as often below as above), where the inner sd is wide,
but leverage is all but absent: the sd shift of a context is 4 or more in 2 tables and 20
or more in none, against 2% of 99 public datasets at 4. That is arm A0 as generated, not a
choice made here. The 64-table pilot, capped at exactly 4.00, had understated the tail.

    venv-tfmp/Scripts/python.exe experiments/h3_repair/prior_pretraining_dump.py \\
        --save_path .cache/tfmp/dumps/base_1024x32.h5 --num_tables 20000
"""
import argparse
import json
import random
import time
from pathlib import Path


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--save_path", required=True)
    p.add_argument("--num_tables", type=int, required=True)
    p.add_argument("--gen_batch", type=int, default=16,
                   help="tables per call to tabicl; they share seq_len and train size")
    p.add_argument("--batch_size_per_gp", type=int, default=4,
                   help="tables per group sharing SCM hyperparameters and feature count")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--min_features", type=int, default=2)
    p.add_argument("--max_features", type=int, default=32)
    p.add_argument("--min_train_size", type=int, default=512)
    p.add_argument("--max_train_size", type=int, default=896)
    p.add_argument("--prior_type", default="mix_scm")
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--np_seed", type=int, default=2026)
    p.add_argument("--torch_seed", type=int, default=2026)
    return p.parse_args()


def main():
    args = parse()
    from importlib.metadata import version

    import h5py
    import numpy as np
    import torch
    from tabicl.prior import PriorDataset

    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)
    random.seed(args.torch_seed)

    # min_seq_len=None is how tabicl fixes the length: its sampler draws
    # np.random.randint(min, max), whose upper bound is exclusive, so min == max raises.
    prior = PriorDataset(regression=True, batch_size=args.gen_batch,
                         batch_size_per_gp=args.batch_size_per_gp,
                         min_features=args.min_features, max_features=args.max_features,
                         min_seq_len=None, max_seq_len=args.seq_len,
                         min_train_size=args.min_train_size,
                         max_train_size=args.max_train_size,
                         prior_type=args.prior_type, n_jobs=args.n_jobs, device="cpu")

    out = Path(args.save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    L, F, n = args.seq_len, args.max_features, args.num_tables
    t0, written, skipped = time.time(), 0, 0
    with h5py.File(out, "w") as f:
        dX = f.create_dataset("X", shape=(n, L, F), dtype="f4", chunks=(1, L, F),
                              compression="lzf")
        dy = f.create_dataset("y", shape=(n, L), dtype="f4", chunks=(1, L))
        dnf = f.create_dataset("num_features", shape=(n,), dtype="i4")
        dnd = f.create_dataset("num_datapoints", shape=(n,), dtype="i4")
        dsplit = f.create_dataset("train_test_split_index", shape=(n,), dtype="i4")
        f.create_dataset("original_batch_size", data=np.array([args.gen_batch]))
        f.create_dataset("problem_type", data="regression", dtype=h5py.string_dtype())
        f.attrs["config"] = json.dumps(vars(args))
        # tabicl 2.2.0 has no __version__ attribute.
        f.attrs["tabicl_version"] = version("tabicl")

        while written < n:
            X, y, d, seq_lens, train_sizes = prior.get_batch()
            X, y = X.numpy(), y.numpy()
            for i in range(len(y)):
                if written >= n:
                    break
                if int(seq_lens[i]) != L or X.shape[1] != L:
                    raise RuntimeError(f"table of {int(seq_lens[i])} rows, expected {L}")
                if not (np.isfinite(X[i]).all() and np.isfinite(y[i]).all()) \
                        or np.std(y[i]) == 0:
                    skipped += 1
                    continue
                k = int(d[i])
                dX[written] = 0.0
                dX[written, :, :k] = X[i, :, :k]
                dy[written] = y[i]
                dnf[written], dnd[written] = k, L
                dsplit[written] = int(train_sizes[i])
                written += 1
            if written % (20 * args.gen_batch) < args.gen_batch:
                rate = written / max(time.time() - t0, 1e-9)
                print(f"{written}/{n} tables, {skipped} skipped, {rate:.1f}/s", flush=True)
        f.attrs["skipped_non_finite_or_constant"] = skipped
    print(f"done: {written} tables in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
