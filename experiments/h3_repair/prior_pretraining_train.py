# -*- coding: utf-8 -*-
"""Pre-train one arm of nanoTabPFN on the base dump: prior (A0, A1) x encoding (B0, B1).

The loop is TFM-Playground's `tfmplayground/train.py` (lines 64 to 99 at commit 7b37681):
schedule-free AdamW, gradient norm clipped at 1, the context target standardised per
table and the test targets with the same statistics, `FullSupportBarDistribution` as the
loss. It is repeated here rather than edited in the ignored clone, so that the change the
experiment makes is versioned. What differs from the original:

  - the target encoding is `common.adapters.nanotabpfn.encode_stats`, the same function
    the adapter applies at inference: `meansd` (B0) or `robust` (B1);
  - arm A1 applies `common.prior_arms.apply_a1` to every table before encoding, with a
    random stream fixed by the table index and the pre-training seed, so A1B0 and A1B1
    of one seed see identical tables;
  - the bucket edges are computed from the encoded **test** targets of the arm's own
    first `--edge_tables` tables, the values the loss is evaluated on, with pfns'
    `get_bucket_limits`. TFM-Playground's `make_global_bucket_edges` standardises all rows
    of a table by statistics of all rows, which the loss never uses;
  - fp16 autocast with a gradient scaler by default, because the RTX 2060 has no bf16;
    `--amp bf16` on a card that has it drops the scaler. The loss is computed in float32;
  - the encoded target is capped at +-`--target_clip` (1e4) in the context, the loss and
    the edges. Without it the first B1 run met encoded values of 2e9 on contexts whose
    robust scale is a millionth of the sd, and its loss was NaN from step 50;
  - a micro-batch of several tables uses the smallest split among them, so every table
    keeps at least that many context rows;
  - a resumable checkpoint every `--ckpt_every` steps, and a CSV of the loss.

    venv-tfmp/Scripts/python.exe experiments/h3_repair/prior_pretraining_train.py \\
        --dump .cache/tfmp/dumps/base_1024x32.h5 --arm A1B0 --seed 1 --steps 20000
"""
import argparse
import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TFMP = ROOT / ".cache" / "tfmp"


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--dump", required=True)
    p.add_argument("--arm", required=True, choices=["A0B0", "A0B1", "A1B0", "A1B1"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, required=True, help="optimiser steps")
    p.add_argument("--batch_size", type=int, default=1, help="tables per micro-batch")
    p.add_argument("--accumulate", type=int, default=8, help="micro-batches per step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=0,
                   help="steps over which the learning rate is warmed up; at 3e-4 without "
                        "it, half of the runs blew up in the first thousand steps")
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--embedding", type=int, default=192)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--n_buckets", type=int, default=100)
    p.add_argument("--edge_tables", type=int, default=2000)
    p.add_argument("--p_tail", type=float, default=0.5)
    p.add_argument("--p_contam", type=float, default=0.3)
    p.add_argument("--xi_lo", type=float, default=0.0)
    p.add_argument("--xi_hi", type=float, default=1.0)
    p.add_argument("--w_sd", type=float, default=2.0)
    p.add_argument("--shift_lo", type=float, default=4.0)
    p.add_argument("--shift_hi", type=float, default=50.0)
    p.add_argument("--init", default="",
                   help="start from the weights and bucket edges of this checkpoint instead "
                        "of from scratch: a curriculum from a model that already learned")
    p.add_argument("--preload", type=float, default=4.0,
                   help="read dumps into memory when they fit in this many GB; 0 to stream")
    p.add_argument("--target_clip", type=float, default=1e4,
                   help="cap on the encoded target in the context, the loss and the edges")
    p.add_argument("--amp", default="fp16",
                   help="fp16 (with a gradient scaler), bf16 (without one), or none; "
                        "1 and 0 are read as fp16 and none, as the earlier runs wrote them")
    p.add_argument("--sdpa", type=int, default=1,
                   help="call attention with need_weights=False, 1 or 0")
    p.add_argument("--grad_ckpt", type=int, default=1,
                   help="recompute each transformer block in the backward pass, 1 or 0")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--max_hours", type=float, default=0.0,
                   help="stop cleanly after this wall time; 0 means no limit")
    return p.parse_args()


def fit_in_memory(model, sdpa: bool, grad_ckpt: bool):
    """Two changes that leave the function the model computes as it is.

    `nn.MultiheadAttention` returns averaged attention weights by default, and to do so
    runs the math path, which keeps every attention matrix for the backward pass. The
    layers discard the weights (`[0]`), so asking for none only changes the kernel. And
    recomputing each block in the backward pass trades time for activation memory. On a
    6 GB card the model with neither peaked at 7.9 GB, spilled into system memory, and ran
    at 0.1 tables per second.
    """
    import functools

    import torch
    from torch.utils.checkpoint import checkpoint

    if sdpa:
        for m in model.modules():
            if isinstance(m, torch.nn.MultiheadAttention):
                m.forward = functools.partial(m.forward, need_weights=False)
    if grad_ckpt:
        for block in model.transformer_blocks:
            inner = block.forward

            def forward(src, train_test_split_index, num_mem_chunks=1, _inner=inner):
                if not torch.is_grad_enabled():
                    return _inner(src, train_test_split_index, num_mem_chunks)
                return checkpoint(_inner, src, train_test_split_index, num_mem_chunks,
                                  use_reentrant=False)
            block.forward = forward
    return model


def bucket_edges(tables, n_tables, n_buckets):
    import numpy as np
    import torch
    from pfns.bar_distribution import get_bucket_limits
    ys = []
    for t in range(min(n_tables, tables.n)):
        _, _, _, split, _ = tables.raw(t)
        _, ye, _, _ = tables.encoded(t, split)
        ys.append(ye[split:])
    ys = torch.tensor(np.concatenate(ys), dtype=torch.float32)
    return get_bucket_limits(n_buckets, ys=ys)


def main():
    args = parse()
    import numpy as np
    import schedulefree
    import torch
    from pfns.bar_distribution import FullSupportBarDistribution
    from tfmplayground.models.nanotabpfn import NanoTabPFNModel

    from common.prior_dumps import Tables

    name = f"nanoTabPFN-{args.arm}-s{args.seed}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in ("checkpoints", "work", "logs"):
        (TFMP / d).mkdir(parents=True, exist_ok=True)
    final, work = TFMP / "checkpoints" / f"{name}.pth", TFMP / "work" / f"{name}.pth"
    log_path = TFMP / "logs" / f"{name}.csv"

    torch.manual_seed(args.seed)
    tables = Tables(args.dump, args)
    model = NanoTabPFNModel(embedding_size=args.embedding, num_attention_heads=args.heads,
                            mlp_hidden_size=args.hidden, num_layers=args.layers,
                            num_outputs=args.n_buckets).to(dev)
    fit_in_memory(model, bool(args.sdpa), bool(args.grad_ckpt))
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr,
                                               weight_decay=0.0,
                                               warmup_steps=args.warmup)
    amp = {"1": "fp16", "0": "none"}.get(str(args.amp), str(args.amp))
    if amp not in ("fp16", "bf16", "none"):
        raise SystemExit(f"--amp {args.amp!r}: expected fp16, bf16 or none")
    # Without including_emulation=False torch reports bf16 on a Turing card (RTX 2060,
    # T4) and the run silently trains in emulated bf16.
    if amp == "bf16" and dev == "cuda" and \
            not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit(f"--amp bf16 on {torch.cuda.get_device_name(0)}, which has no bf16")
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    use_autocast = amp != "none" and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16" and dev == "cuda")

    step, pointer, seconds = 0, 0, 0.0
    if work.exists():
        state = torch.load(work, map_location="cpu", weights_only=False)
        if state["config"] != vars(args) | {"steps": state["config"]["steps"],
                                             "max_hours": state["config"]["max_hours"]}:
            raise SystemExit(f"{work} was written with another configuration; move it aside")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        edges = state["bucket_edges"]
        step, pointer, seconds = state["step"], state["pointer"], state["seconds"]
        print(f"resuming {name} at step {step}", flush=True)
    elif args.init:
        # A curriculum: the weights and the edges of a model that learned on smaller
        # tables. Keeping its edges keeps the meaning of the decoder's outputs; the
        # optimiser starts fresh.
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"])
        edges = init["bucket_edges"]
        if len(edges) - 1 != args.n_buckets:
            raise SystemExit(f"--init has {len(edges) - 1} buckets, --n_buckets is "
                             f"{args.n_buckets}")
        print(f"initialised from {args.init}, step {init.get('step')}, edges "
              f"[{edges[0]:.3g}, {edges[-1]:.3g}]", flush=True)
    else:
        t_edges = time.time()
        edges = bucket_edges(tables, args.edge_tables, args.n_buckets)
        print(f"edges from {min(args.edge_tables, tables.n)} tables in "
              f"{time.time() - t_edges:.0f}s: [{edges[0]:.3g}, {edges[-1]:.3g}]", flush=True)
    dist = FullSupportBarDistribution(edges.to(dev))

    def save(path, final_weights):
        model.eval()
        optimizer.eval()
        state = dict(architecture=dict(num_layers=args.layers, embedding_size=args.embedding,
                                       num_attention_heads=args.heads,
                                       mlp_hidden_size=args.hidden,
                                       num_outputs=args.n_buckets),
                     model=model.state_dict(), bucket_edges=edges.cpu(),
                     encoding=tables.encoding, target_clip=args.target_clip,
                     max_context=tables.max_split,
                     max_features=tables.max_features, config=vars(args), step=step,
                     pointer=pointer, seconds=seconds)
        if not final_weights:
            state.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        model.train()
        optimizer.train()

    new_log = not log_path.exists()
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["step", "tables", "loss", "seconds", "tables_per_s", "max_mem_mb",
                         "share_tail", "share_contam", "skipped"])

    model.train()
    optimizer.train()
    t0, window_loss, window_n, n_tail, n_contam, skipped = time.time() - seconds, 0.0, 0, 0, 0, 0
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    while step < args.steps:
        for _ in range(args.accumulate):
            ids = [pointer + i for i in range(args.batch_size)]
            pointer += args.batch_size
            raws = [tables.raw(t) for t in ids]
            split = min(r[3] for r in raws)
            k = max(r[2] for r in raws)
            xs, ys = [], []
            for t in ids:
                X, ye, kt, info = tables.encoded(t, split)
                n_tail += info.get("tail", False)
                n_contam += info.get("contam", False)
                xs.append(np.pad(X, ((0, 0), (0, k - kt))))
                ys.append(ye)
            x = torch.tensor(np.stack(xs), device=dev)
            y = torch.tensor(np.stack(ys), dtype=torch.float32, device=dev)
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                skipped += 1
                continue
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
                out = model((x, y[:, :split]), train_test_split_index=split)
            loss = dist(out.float(), y[:, split:]).mean() / args.accumulate
            scaler.scale(loss).backward()
            window_loss += loss.detach().item() * args.accumulate
            window_n += 1
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        seconds = time.time() - t0

        if step % args.log_every == 0 or step == args.steps:
            mem = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
            n_tab = pointer
            writer.writerow([step, n_tab, round(window_loss / max(window_n, 1), 5),
                             round(seconds, 1), round(n_tab / max(seconds, 1e-9), 3),
                             round(mem), round(n_tail / max(n_tab, 1), 4),
                             round(n_contam / max(n_tab, 1), 4), skipped])
            log.flush()
            print(f"{name} step {step}/{args.steps} loss {window_loss / max(window_n, 1):.4f} "
                  f"{n_tab / max(seconds, 1e-9):.2f} tables/s mem {mem:.0f}MB", flush=True)
            window_loss, window_n = 0.0, 0
        if step % args.ckpt_every == 0:
            save(work, final_weights=False)
        if args.max_hours and seconds > args.max_hours * 3600:
            save(work, final_weights=False)
            print(f"stopped at {args.max_hours} h, step {step}; rerun to resume", flush=True)
            return

    save(final, final_weights=True)
    save(work, final_weights=False)
    log.close()
    print(json.dumps(dict(name=name, steps=step, tables=pointer, seconds=round(seconds))),
          flush=True)


if __name__ == "__main__":
    main()
