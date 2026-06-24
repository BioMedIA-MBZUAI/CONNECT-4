"""
CONNECT-4 training (multi-GPU, DistributedDataParallel).

    # single GPU
    python -m training.train --config configs/connect4.yaml
    # 4 GPUs
    torchrun --standalone --nproc_per_node=4 -m training.train --config configs/connect4.yaml

Features
--------
* patient-level train/val split (no subject leakage; `data/split.py`),
* the six-term Figure-1E objective, printed every `log_every` steps,
* periodic validation with full evaluation metrics (`eval/metrics.py`) printed
  on rank 0,
* periodic real-vs-synthetic magma visualisation (`eval/visualize.py`),
* AMP + gradient accumulation + per-epoch checkpoints (rank 0).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data.dataset_precomputed import Connect4PrecomputedDataset
from data.collate import connect4_collate_fn
from data.split import patient_level_split
from data.safe import SafeDataset
from models.connect4 import Connect4Model
from eval.metrics import compute_all
from eval.visualize import plot_real_vs_synthetic


# --------------------------------------------------------------------------- #
def setup_distributed():
    """Return (is_dist, rank, world_size, local_rank). Inits NCCL if launched by torchrun."""
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_loaders(cfg, world, rank):
    d = cfg["data"]
    full = Connect4PrecomputedDataset(
        root_dir=d["root_dir"], precomputed_dir=d["precomputed_dir"],
        target_shape=tuple(d["target_shape"]), num_frames=int(d["num_frames"]),
        normalize_intensity=d.get("normalize_intensity", True),
    )
    tr_idx, va_idx = patient_level_split(
        full.scan_ids, val_frac=cfg["training"].get("val_frac", 0.15),
        seed=cfg["training"].get("seed", 42),
    )
    if is_main(rank):
        n_pat = len({s.rsplit("_", 1)[0] for s in full.scan_ids})
        print(f"[data] {len(full.scan_ids)} scans / {n_pat} patients "
              f"-> train {len(tr_idx)} / val {len(va_idx)} (patient-level)", flush=True)
    safe = SafeDataset(full)                       # tolerate corrupt/empty NIfTIs
    train_ds, val_ds = Subset(safe, tr_idx), Subset(safe, va_idx)

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=cfg["training"].get("num_workers", 2),
        collate_fn=connect4_collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=connect4_collate_fn, pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


@torch.no_grad()
def validate(model, val_loader, device, cfg, rank, step, out_dir):
    model.eval()
    n = cfg["training"].get("val_batches", 4)
    agg, count = {}, 0
    first = None
    for i, batch in enumerate(val_loader):
        if i >= n:
            break
        batch = move(batch, device)
        out = model(batch)
        if "target" not in out:
            continue
        m = compute_all(out["prediction"], out["target"],
                        roi_masks=out.get("roi_masks"), mask=out.get("brain_mask"))
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        count += 1
        if first is None:
            first = (out["target"].detach(), out["prediction"].detach())
    model.train()
    if count == 0:
        return
    agg = {k: v / count for k, v in agg.items()}
    if is_main(rank):
        print(f"[val step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()), flush=True)
        if first is not None and cfg["training"].get("visualize", True):
            path = Path(out_dir) / f"val_step{step}_real_vs_synthetic.png"
            plot_real_vs_synthetic(first[0], first[1], str(path))
            print(f"[val step {step}] viz -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/connect4.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    is_dist, rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_sampler = build_loaders(cfg, world, rank)

    model = Connect4Model(cfg).to(device)
    if is_dist:
        model = DDP(model, device_ids=[local], output_device=local, find_unused_parameters=True)
    core = model.module if is_dist else model

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    # Prefer bf16 on Ampere+ (A100): same exponent range as fp32, so the FFT /
    # attention / conv ops don't overflow -> no GradScaler and no skipped steps
    # (fp16 was overflowing, causing the recurring "non-finite grad" skips).
    amp_on = cfg["training"].get("amp", True)
    use_bf16 = amp_on and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_on and not use_bf16))
    if is_main(rank):
        print(f"[amp] dtype={'bf16' if use_bf16 else ('fp16' if amp_on else 'fp32')} "
              f"grad_scaler={'on' if (amp_on and not use_bf16) else 'off'}", flush=True)
    accum = cfg["training"].get("grad_accum_steps", 1)
    log_every = cfg["training"].get("log_every", 20)
    eval_every = cfg["training"].get("eval_every", 200)

    ckpt_dir = Path(cfg["training"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["training"].get("out_dir", "outputs")); out_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(path, epoch, partial):
        if is_main(rank):
            torch.save({"model": core.state_dict(), "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(), "config": cfg,
                        "epoch": epoch, "step": step, "partial": partial}, path)

    # ---- resume from the most recent checkpoint so the 24h wall-time limit does
    #      not throw away progress (just resubmit; training continues). ----------
    start_epoch, step = 0, 0
    latest = ckpt_dir / "latest.pt"
    epoch_ckpts = sorted(ckpt_dir.glob("connect4_epoch*.pt"))
    resume_path = latest if latest.exists() else (epoch_ckpts[-1] if epoch_ckpts else None)
    if resume_path is not None:
        ck = torch.load(resume_path, map_location=device)
        try:
            core.load_state_dict(ck["model"])
        except Exception as e:                       # arch changed -> start fresh
            if is_main(rank):
                print(f"[resume] checkpoint incompatible ({e}); starting fresh", flush=True)
            ck = None
    if resume_path is not None and ck is not None:
        if ck.get("optimizer") is not None:
            try: opt.load_state_dict(ck["optimizer"])
            except Exception: pass
        if ck.get("scaler") is not None:
            try: scaler.load_state_dict(ck["scaler"])
            except Exception: pass
        # if the saved epoch finished, resume at the next one; else redo it
        start_epoch = ck.get("epoch", 0) + (0 if ck.get("partial") else 1)
        step = ck.get("step", start_epoch * len(train_loader))
        if is_main(rank):
            print(f"[resume] {resume_path.name} -> start epoch {start_epoch}, step {step}", flush=True)

    model.train()
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for it, batch in enumerate(train_loader):
            batch = move(batch, device)
            with torch.cuda.amp.autocast(enabled=amp_on, dtype=amp_dtype):
                out = model(batch)
                loss = out["losses"]["total"] / accum
            scaler.scale(loss).backward()
            if (it + 1) % accum == 0:
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"].get("grad_clip", 1.0))
                if torch.isfinite(gnorm):
                    scaler.step(opt)
                elif is_main(rank):
                    print(f"[warn] non-finite grad norm at step {step}; step skipped", flush=True)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if is_main(rank) and step % log_every == 0:
                terms = {k: round(float(v), 4) for k, v in out["losses"].items()}
                print(f"epoch {epoch} step {step} | {terms}", flush=True)
            if step > 0 and step % eval_every == 0:
                validate(model, val_loader, device, cfg, rank, step, out_dir)
            if step > 0 and step % cfg["training"].get("ckpt_every", 100) == 0:
                save_ckpt(ckpt_dir / "latest.pt", epoch, partial=True)   # frequent safety save
                if is_main(rank):
                    print(f"[ckpt] latest saved at step {step}", flush=True)
            step += 1

        save_ckpt(ckpt_dir / f"connect4_epoch{epoch:03d}.pt", epoch, partial=False)
        save_ckpt(ckpt_dir / "latest.pt", epoch, partial=False)
        if is_main(rank):
            print(f"[ckpt] saved epoch {epoch}", flush=True)

    if is_dist:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
