# -*- coding: utf-8 -*-
"""Has a pre-trained nanoTabPFN learned anything? The gate before any arm is compared.

Two readings of the predicted median, both as R2 against the truth:

  - a plain linear task, y = 2 x0 - x1 + N(0, 0.3^2), three features, `--context` rows of
    context and 200 test rows, five seeds. The released TFM-Playground regressor gets
    +0.46 at 35 rows; every model of the first two local settings got about -0.01, and
    that is what this check exists to catch;
  - optionally, held-out tables of the model's own prior (`--heldout`), where it is also
    compared with the same head told only the marginal of the context.

A model that stays near zero on the linear task predicts a constant, and no statement
about its tail under leverage means anything.

    python experiments/h3_repair/prior_pretraining_check.py \\
        --checkpoint .cache/tfmp/checkpoints/nanoTabPFN-A0B0-s81.pth --context 200 \\
        --heldout .cache/tfmp/dumps/heldout_256_200.h5
"""
import argparse


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--context", type=int, default=200, help="context rows of the linear task")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--heldout", default="", help="a dump of held-out prior tables")
    p.add_argument("--tables", type=int, default=200, help="held-out tables to read")
    p.add_argument("--device", default="", help="cuda or cpu; default cuda when available")
    return p.parse_args()


def linear_r2(path, context, seeds, dev):
    import numpy as np

    from common.adapters import nanotabpfn
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(context + 200, 3))
        y = 2.0 * X[:, 0] - X[:, 1] + 0.3 * rng.normal(size=len(X))
        q, _ = nanotabpfn.predict(path, X[:context], y[:context], X[context:], [0.5], dev=dev)
        out.append(1 - np.mean((y[context:] - q[:, 0]) ** 2) / np.var(y[context:]))
    return np.array(out)


def heldout(path, dump, n_tables, dev):
    import h5py
    import numpy as np
    import torch

    from common.adapters.nanotabpfn import encode, encode_stats, load
    model, dist, meta = load(path, dev)
    edges = dist.borders.double().cpu().numpy()
    f = h5py.File(dump, "r")
    gain, r2 = [], []
    for t in range(min(n_tables, f["y"].shape[0])):
        k = int(f["num_features"][t])
        sp = int(f["train_test_split_index"][t])
        y = f["y"][t].astype(float)
        c, s = encode_stats(y[:sp], meta["encoding"])
        ye = encode(y, c, s, meta["target_clip"])
        with torch.no_grad():
            x = torch.tensor(f["X"][t, :, :k].astype(np.float32), device=dev)[None]
            yt = torch.tensor(ye[:sp], dtype=torch.float32, device=dev)[None]
            logits = model((x, yt), train_test_split_index=sp).squeeze(0).float()
            target = torch.tensor(ye[sp:], dtype=torch.float32, device=dev)[:, None]
            nll = float(dist(logits[:, None, :], target).mean())
            idx = np.clip(np.searchsorted(edges, ye[:sp]) - 1, 0, len(edges) - 2)
            counts = np.bincount(idx, minlength=len(edges) - 1) + 0.5
            marg = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32,
                                device=dev).repeat(len(target), 1)
            gain.append(float(dist(marg[:, None, :], target).mean()) - nll)
            med = dist.icdf(logits, 0.5).double().cpu().numpy() * s + c
        r2.append(1 - np.mean((y[sp:] - med) ** 2) / max(np.var(y[sp:]), 1e-12))
    return np.array(gain), np.array(r2)


def main():
    args = parse()
    import numpy as np
    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    r = linear_r2(args.checkpoint, args.context, args.seeds, dev)
    name = args.checkpoint.replace("\\", "/").split("/")[-1]
    print(f"{name}: linear task, {args.context} context rows, R2 {r.mean():+.3f} "
          f"({', '.join(f'{v:+.2f}' for v in r)})", flush=True)
    if args.heldout:
        g, h = heldout(args.checkpoint, args.heldout, args.tables, dev)
        print(f"{name}: {len(h)} held-out prior tables, R2 median {np.median(h):+.3f}, above 0 "
              f"in {int((h > 0).sum())}/{len(h)}; NLL gain over the context marginal "
              f"{np.median(g):+.3f} nats", flush=True)
    verdict = "LEARNED" if r.mean() > 0.3 else "DID NOT LEARN"
    print(f"{name}: {verdict} (gate: linear R2 above 0.3)", flush=True)


if __name__ == "__main__":
    main()
