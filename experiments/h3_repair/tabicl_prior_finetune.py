# -*- coding: utf-8 -*-
"""Continue TabICLv2's pre-training under a heavy-tailed, contaminated prior.

WHY
---
On freMTPL2sev TabICLv2 loses 15.4% of Q(0.99) on subsamples whose largest claim shifts
the sd by 4 or more (`clip_context_real.py`, raw against without_max). It has no grid: its
StandardScaler squashes the body under an inflated sd and the network conditions on a
nearly degenerate context. EXAONE standardises the same way and loses nothing, which
points at what the network has learned rather than at the pipeline. Whether that is so
cannot be read off the production models; it can be tested by changing what TabICLv2
learns from and nothing else.

A small model pre-trained from scratch was the first attempt
(`prior_pretraining_eval.py`) and never learned at usable table sizes on this GPU.
TabICLv2 already predicts, has 28.5 M parameters, and its continued pre-training fits
here: 1668 MB and 0.35 s per step at 1024 rows (`data/tfmp_artifacts.json`).

WHAT IS TRAINED
---------------
Starting from `tabicl-regressor-v2-20260212.ckpt`, two arms that see the same tables in
the same order, with the same optimiser, steps and seed:

    FT-A0   the 1024-row dumps as generated, TabICL's own mix_scm regression prior
    FT-A1   the same tables under `common.prior_arms.apply_a1`: half with a GPD tail
            whose shape follows one feature, 30% with one context row rescaled to an sd
            shift in [4, 50]; test targets never contaminated

The target is standardised by the mean and sd of the context, as TabICLRegressor's
StandardScaler does before prediction (ddof=1 here against its ddof=0, under 0.1% apart at
512 or more rows), and capped at +-1e4. The loss is TabICL's own, pinball over 999 levels
(tabicl/train/_run.py). AdamW at 1e-5, weight decay 0.01, 10% linear warmup then cosine,
gradient clip 1.0: the defaults of tabicl's FinetunedTabICLBase. The column embedder and
the row interactor are frozen (1.27 M of 28.5 M parameters), so a batch of 4 tables fits
in 1.5 GB; the ICL predictor and the quantile head train. Table order starts at an offset
set by the seed.

The checkpoint is written as {"config", "state_dict"}, the form TabICLRegressor loads
with `model_path`, so the arms are measured by the same scripts as the base model under
the names TabICLv2-FT-A0-s1, TabICLv2-FT-A1-s1 and so on (`common/models.py`).

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
On `clip_context_real.py`, freMTPL2sev, N_FIT=2000, the raw arm against without_max, the
same 20 subsamples as the base model, paired per subsample; medians over both pre-training
seeds.

F1, the control. FT-A0 keeps the base model's behaviour: its median d_q99 in the bin at 4
sd and above is within 5 points of the base model's -15.4%, and its median pinball at 0.99
in the two bins below 2 sd is within 5% of the base model's.

F2. FT-A1 at least halves the loss: its median d_q99 at 4 sd and above is -7.7% or closer
to zero, and closer to zero than the base model's in at least 4 of those 5 subsamples.

F3. FT-A1 costs little where there is no leverage: its median pinball at 0.99 in the two
bins below 2 sd is at most 5% above the base model's.

F4. Where the leverage is, FT-A1 predicts better than the control: its pinball at 0.99 in
the bins at 2 sd and above is lower than FT-A0's in at least 8 of those 10 subsamples.

OUTCOME, 16.9.2026
------------------
Four arms fine-tuned, FT-A0 and FT-A1 on seeds 1 and 2, 20000 tables each, 17:12 to 19:17;
evaluated by run/tabicl_prior_finetune_eval.sh, rows in
`results/h3_repair/clip_context_real_tabicl_ft.csv`. The base model was measured again in
the same environment (tabicl 2.2.0) and reproduces its committed rows (tabicl 2.1.1) to
7.8e-06 on all 20 cells. A subsample's value is the mean over the two seeds.

F1  FAILS, narrowly and usefully. FT-A0's median d_q99 at 4 sd and above is -11.1%
    against the base model's -15.4%, 4.3 points and inside the bound, but its median
    pinball at 0.99 below 2 sd is 5.2% lower than the base model's, outside it by 0.2.
    Continued training on the model's own prior is not a no-op: it improves the clean
    contexts and takes more than a quarter of the leverage loss away with no heavy tail
    and no contamination. The comparison for the prior is FT-A1 against FT-A0.

F2  FAILS. FT-A1's median d_q99 at 4 sd and above is -9.0%, not -7.7% or closer, though
    closer to zero than the base model's in 5 of 5 subsamples (-15.4, -14.4, -19.2, -13.0
    and -27.7% to -1.2, -9.0, -14.3, -7.0 and -20.1%). The seeds disagree: -13.0% for
    seed 1, -5.0% for seed 2.

F3  HOLDS. FT-A1 costs nothing on clean contexts: its median pinball at 0.99 below 2 sd
    is 6.5% below the base model's, and above it in 2 of 10 subsamples.

F4  HOLDS, at the threshold. At 2 sd and above FT-A1's pinball at 0.99 is lower than
    FT-A0's in 8 of 10 subsamples (one-sided sign test p = 0.055), median ratio 0.966;
    against the base model in 9 of 10, median ratio 0.952.

What this says. Heavy tails and contaminated contexts in TabICLv2's prior move its tail
under leverage in the predicted direction, but the part owed to the prior beyond
fine-tuning as such is small (median d_q99 at 4 sd and above -9.0% against FT-A0's
-11.1%), not consistent across the two seeds, and at the edge of significance on ten
subsamples of a single dataset. A cap on the context still removes the rest in every
arm (clip_200 at 4 sd and above: +2.4% and +2.5% for FT-A1, -3.6% and -3.7% for FT-A0).
The prior is not the whole of the failure, and the evidence that it is part of it is
weak; more seeds and a second dataset would be needed to say more.

    medians, subsample values averaged over the two seeds
    model      d_q99 at 4 sd+   pinball 0.99 below 2 sd   pinball 0.99 at 2 sd+ vs base
    base       -15.4%           504                        1
    FT-A0      -11.1%           478 (-5.2%)
    FT-A1       -9.0%           471 (-6.5%)                0.952 (lower in 9 of 10)

    venv-tfmp/Scripts/python.exe -u experiments/h3_repair/tabicl_prior_finetune.py \\
        --dump .cache/tfmp/dumps/base_s2026_14000.h5 --arm A1 --seed 1 --tables 20000
"""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TFMP = ROOT / ".cache" / "tfmp"
BASE_REPO, BASE_FILE = "jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt"


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--dump", required=True)
    p.add_argument("--arm", required=True, choices=["A0", "A1"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tables", type=int, required=True, help="tables to train on")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"])
    # The column and row stages hold 4% of the parameters; freezing them keeps a batch of 4
    # at 1490 MB on the 6 GB card, where the full model at batch 4 spilled out of memory.
    p.add_argument("--freeze", default="colrow", choices=["none", "colrow"])
    p.add_argument("--target_clip", type=float, default=1e4)
    p.add_argument("--preload", type=float, default=4.0)
    p.add_argument("--p_tail", type=float, default=0.5)
    p.add_argument("--p_contam", type=float, default=0.3)
    p.add_argument("--xi_lo", type=float, default=0.0)
    p.add_argument("--xi_hi", type=float, default=1.0)
    p.add_argument("--w_sd", type=float, default=2.0)
    p.add_argument("--shift_lo", type=float, default=4.0)
    p.add_argument("--shift_hi", type=float, default=50.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--max_hours", type=float, default=0.0)
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    args = parse()
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from tabicl._model.tabicl import TabICL

    from common.prior_dumps import Tables

    name = f"TabICLv2-FT-{args.arm}-s{args.seed}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in ("tabicl_ft", "work", "logs"):
        (TFMP / d).mkdir(parents=True, exist_ok=True)
    final = TFMP / "tabicl_ft" / f"{name}.ckpt"
    work = TFMP / "work" / f"{name}.pth"
    log_path = TFMP / "logs" / f"{name}.csv"
    if final.exists():
        raise SystemExit(f"{final} exists; move it aside to train again")

    base_path = hf_hub_download(BASE_REPO, BASE_FILE)
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    cfg = base["config"]

    torch.manual_seed(args.seed)
    tables = Tables(args.dump, argparse.Namespace(
        arm=f"{args.arm}B0", seed=args.seed, preload=args.preload,
        target_clip=args.target_clip, p_tail=args.p_tail, p_contam=args.p_contam,
        xi_lo=args.xi_lo, xi_hi=args.xi_hi, w_sd=args.w_sd,
        shift_lo=args.shift_lo, shift_hi=args.shift_hi))

    model = TabICL(**cfg)
    model.load_state_dict(base["state_dict"])
    model.to(dev).train()
    if args.freeze == "colrow":
        for sub in (model.col_embedder, model.row_interactor):
            sub.eval()
            for prm in sub.parameters():
                prm.requires_grad = False
    params = [prm for prm in model.parameters() if prm.requires_grad]

    steps = max(1, args.tables // args.batch_size)
    warmup = max(1, int(args.warmup_frac * steps))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if args.amp == "bf16" and dev == "cuda" and \
            not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit(f"--amp bf16 on {torch.cuda.get_device_name(0)}, which has no bf16")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    use_autocast = args.amp != "none" and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and dev == "cuda")
    levels = torch.linspace(0.0, 1.0, cfg["num_quantiles"] + 2, device=dev)[1:-1].view(1, 1, -1)

    step, seconds = 0, 0.0
    pointer = (args.seed * 10007) % tables.n
    if work.exists():
        state = torch.load(work, map_location="cpu", weights_only=False)
        same = {k: v for k, v in state["args"].items() if k != "max_hours"}
        if same != {k: v for k, v in vars(args).items() if k != "max_hours"}:
            raise SystemExit(f"{work} was written with other arguments; move it aside")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, pointer, seconds = state["step"], state["pointer"], state["seconds"]
        print(f"resuming {name} at step {step} of {steps}", flush=True)

    def save_work():
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        step=step, pointer=pointer, seconds=seconds, args=vars(args)),
                   work.with_suffix(".tmp"))
        work.with_suffix(".tmp").replace(work)

    new_log = not log_path.exists()
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["step", "tables", "loss", "lr", "seconds", "tables_per_s",
                         "max_mem_mb", "share_tail", "share_contam", "skipped"])

    t0 = time.time() - seconds
    window, n_window, n_tail, n_contam, skipped = 0.0, 0, 0, 0, 0
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while step < steps:
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
        x = torch.tensor(np.stack(xs), dtype=torch.float32, device=dev)
        y = torch.tensor(np.stack(ys), dtype=torch.float32, device=dev)
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            skipped += 1
            continue
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
            pred = model(x, y[:, :split], None)
        err = y[:, split:].unsqueeze(-1) - pred.float()
        loss = torch.maximum(levels * err, (levels - 1) * err).mean()
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        window += loss.detach().item()
        n_window += 1
        seconds = time.time() - t0

        if step % args.log_every == 0 or step == steps:
            n_tab = step * args.batch_size
            mem = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0
            writer.writerow([step, n_tab, round(window / max(n_window, 1), 6),
                             f"{scheduler.get_last_lr()[0]:.3g}", round(seconds, 1),
                             round(n_tab / max(seconds, 1e-9), 3), round(mem),
                             round(n_tail / max(n_tab, 1), 4),
                             round(n_contam / max(n_tab, 1), 4), skipped])
            log.flush()
            print(f"{name} step {step}/{steps} pinball {window / max(n_window, 1):.5f} "
                  f"{n_tab / max(seconds, 1e-9):.2f} tables/s mem {mem:.0f}MB", flush=True)
            window, n_window = 0.0, 0
        if step % args.ckpt_every == 0:
            save_work()
        if args.max_hours and seconds > args.max_hours * 3600:
            save_work()
            print(f"stopped at {args.max_hours} h, step {step}; rerun to resume", flush=True)
            return

    model.eval()
    torch.save(dict(config=cfg, state_dict={k: v.detach().cpu()
                                            for k, v in model.state_dict().items()}),
               final)
    save_work()
    log.close()
    print(json.dumps(dict(name=name, steps=step, tables=step * args.batch_size,
                          seconds=round(seconds), base=BASE_FILE,
                          base_sha256=sha256(base_path), checkpoint=str(final),
                          sha256=sha256(final))), flush=True)


if __name__ == "__main__":
    main()
