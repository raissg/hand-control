"""Train ``EMG5Bit`` on self-labeled recordings from ``src.record_bits``.

Glob pattern defaults to ``recordings/bits_*/emg.csv`` (all sessions).

Usage:
    python -m src.train_bits
    python -m src.train_bits --sessions "recordings/bits_20260420_*/emg.csv"
    python -m src.train_bits --window 100 --step 10 --epochs 60
    (No --save-name: writes models/model_<timestamp>.pt)
    python -m src.train_bits --epochs 60 --no-early-stop
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.binary_model import EMG5Bit, EMG5BitMulti  # === IMU ADDITION ===
from src.dataset_bits import (
    BitsWindowDataset,
    compute_norm_stats,
    compute_imu_norm_stats,        # === IMU ADDITION ===
    load_sessions,
    split_segments,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="recordings/bits_*/emg.csv")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--step", type=int, default=10,
                    help="Window stride in samples. Smaller = more training windows "
                         "(10 -> ~50 windows/sec held).")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--gap_s", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=12,
                    help="Stop after this many val epochs without improvement. "
                         "Unused if --no-early-stop is set.")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="Always train for the full --epochs (ignore --patience).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--save-name",
        default=None,
        help="Checkpoint filename under models/. Default: model_<timestamp>.pt",
    )
    ap.add_argument("--resume", default=None,
                    help="Path to a checkpoint to continue training from. "
                         "Reuses its norm stats + window so the learned "
                         "input distribution stays consistent.")
    ap.add_argument("--resume-lr", type=float, default=None,
                    help="Override LR when resuming (default: lower to 3e-4).")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Train only the head (linear probe / calibration). "
                         "Freezes EMG (and IMU, if --imu) feature extractors "
                         "and keeps their BatchNorm in eval mode so running "
                         "stats don't drift. With --resume, also recomputes "
                         "input norm stats on the new session data.")
    ap.add_argument("--consistency-lambda", type=float, default=0.0,
                    help="Weight on temporal-consistency MSE. 0 = disabled.")
    ap.add_argument("--consistency-deltas", default="5,50",
                    help="Range 'min,max' of sample offsets for the pair "
                         "window. At 500Hz, '5,50' = 10-100ms.")
    # === IMU ADDITION =====================================================
    # Switch to the two-stream EMG5BitMulti model. Requires recordings made
    # with `record_bits --imu` (or older ones, in which case IMU is zero-
    # filled and the IMU branch contributes nothing).
    # ======================================================================
    ap.add_argument("--imu", action="store_true",
                    help="Train two-stream EMG5BitMulti (EMG + IMU branches). "
                         "Default: EMG-only EMG5Bit.")
    ap.add_argument("--remap", action="store_true",
                    help="Apply LABEL_REMAP in dataset_bits to collapse "
                         "near-miss patterns onto canonical neighbors.")
    args = ap.parse_args()

    if args.save_name is None:
        args.save_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    print(f"Loading sessions matching {args.sessions!r}...")
    # === IMU ADDITION === when training the two-stream model, only load
    # CSVs that actually contain IMU columns. Mixing in legacy IMU-less
    # recordings would dilute the IMU branch with zeros.
    segs = load_sessions(args.sessions, gap_s=args.gap_s,
                          require_imu=args.imu,
                          remap_labels=args.remap)
    print(f"  loaded {len(segs)} segments (total samples: "
          f"{sum(s.emg.shape[0] for s in segs)})")
    if not segs:
        raise SystemExit("No usable segments. Check --sessions pattern / recordings.")

    # Per-bit positive fraction (diagnostic — any finger never bent / always bent?)
    labels = np.stack([s.label for s in segs])
    # Sample-weighted by segment length for honest stats
    weights = np.array([s.emg.shape[0] for s in segs])
    pos_frac = (labels * weights[:, None]).sum(axis=0) / weights.sum()
    print(f"Per-bit positive fraction (sample-weighted): "
          f"{np.round(pos_frac, 3).tolist()}")

    train_segs, val_segs = split_segments(segs, val_frac=args.val_frac,
                                            seed=args.seed)
    print(f"Split: {len(train_segs)} train segs | {len(val_segs)} val segs")

    # Resume mode: reuse the checkpoint's norm stats + window so the previously
    # learned weights don't suddenly see a different input distribution.
    resume_ckpt = None
    # === IMU ADDITION === IMU norm stats live alongside EMG stats but are
    # computed only when --imu is on (otherwise stay None).
    imu_mean = imu_std = None
    if args.resume is not None:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.resume)
        print(f"Resuming from {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device)
        mean = np.asarray(resume_ckpt["input_mean"], dtype=np.float32)
        std = np.asarray(resume_ckpt["input_std"], dtype=np.float32)
        ckpt_win = int(resume_ckpt["window"])
        if ckpt_win != args.window:
            print(f"  window {args.window} -> {ckpt_win} (from checkpoint)")
            args.window = ckpt_win
        # === IMU ADDITION === resume IMU stats too if the checkpoint has them.
        if "imu_mean" in resume_ckpt and "imu_std" in resume_ckpt:
            imu_mean = np.asarray(resume_ckpt["imu_mean"], dtype=np.float32)
            imu_std = np.asarray(resume_ckpt["imu_std"], dtype=np.float32)
            if not args.imu:
                print("  checkpoint has IMU stats — auto-enabling --imu.")
                args.imu = True
        print(f"  using checkpoint norm stats (skipping recomputation)")
    else:
        mean, std = compute_norm_stats(train_segs)
        if args.imu:                              # === IMU ADDITION ===
            imu_mean, imu_std = compute_imu_norm_stats(train_segs)

    # Calibration: when freezing the backbone on top of a resumed checkpoint,
    # the whole point is to adapt to a new band session, so recompute input
    # norm stats from this session's data (overrides the stats inherited above).
    if args.freeze_backbone and resume_ckpt is not None:
        print("  --freeze-backbone: recomputing input norm on session data")
        mean, std = compute_norm_stats(train_segs)
        if args.imu:
            imu_mean, imu_std = compute_imu_norm_stats(train_segs)
    print(f"EMG norm:  mean={np.round(mean, 2).tolist()}")
    print(f"           std ={np.round(std,  2).tolist()}")
    if imu_mean is not None:                              # === IMU ADDITION ===
        print(f"IMU norm:  mean={np.round(imu_mean, 3).tolist()}")
        print(f"           std ={np.round(imu_std,  3).tolist()}")

    pair_range = None
    if args.consistency_lambda > 0:
        dmin, dmax = [int(x) for x in args.consistency_deltas.split(",")]
        pair_range = (dmin, dmax)
        print(f"Consistency loss: λ={args.consistency_lambda}, "
              f"Δ∈[{dmin},{dmax}] samples ({dmin*2}-{dmax*2} ms)")

    # === IMU ADDITION === pass IMU flag + stats through to the dataset so
    # __getitem__ yields aligned IMU windows when training the multistream
    # model. When --imu is off, these are no-ops.
    train_ds = BitsWindowDataset(train_segs, args.window, args.step, mean, std,
                                  pair_delta_range=pair_range, rng_seed=args.seed,
                                  return_imu=args.imu,
                                  imu_mean=imu_mean, imu_std=imu_std)
    val_ds = BitsWindowDataset(val_segs, args.window, args.window, mean, std,
                                return_imu=args.imu,
                                imu_mean=imu_mean, imu_std=imu_std)
    print(f"Windows: train {len(train_ds)} | val {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Not enough windows — record more data or reduce --window.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0)

    # pos_weight for BCE: handles finger imbalance (thumb usually less often bent)
    pos_weight = torch.tensor(
        [(1 - p) / max(p, 1e-3) for p in pos_frac], dtype=torch.float32
    ).to(device)

    # === IMU ADDITION === pick the architecture based on --imu.
    if args.imu:
        model = EMG5BitMulti(emg_channels=8, imu_channels=6,
                             n_out=5, dropout=args.dropout).to(device)
        print("Model: EMG5BitMulti (two-stream EMG + IMU)")
    else:
        model = EMG5Bit(in_channels=8, n_out=5, dropout=args.dropout).to(device)
        print("Model: EMG5Bit (EMG only)")
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["state_dict"])
        print(f"  loaded {sum(p.numel() for p in model.parameters())} params "
              f"from checkpoint")
    lr = args.resume_lr if (resume_ckpt is not None and args.resume_lr is not None) \
         else (3e-4 if resume_ckpt is not None else args.lr)
    if resume_ckpt is not None:
        print(f"  LR: {lr} (fine-tune)")

    # Head-only fine-tune: freeze feature extractors. Keep them in .eval()
    # every epoch so BN running stats don't drift on the small calibration set.
    frozen_modules: list[nn.Module] = []
    if args.freeze_backbone:
        if args.imu:
            frozen_modules = [model.emg_feat, model.imu_feat]
        else:
            frozen_modules = [model.feat]
        for m in frozen_modules:
            for p in m.parameters():
                p.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  --freeze-backbone: training {n_trainable}/{n_total} params "
              f"(head only)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("No trainable parameters — check --freeze-backbone usage.")
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, args.save_name)

    # === IMU ADDITION ====================================================
    # forward_model(emg, imu) hides the EMG-only vs. multistream call:
    #   - EMG5Bit:      model(emg)
    #   - EMG5BitMulti: model(emg, imu)
    # IMU is unused (and always None) when --imu is off.
    # ======================================================================
    def forward_model(emg, imu):
        if args.imu:
            return model(emg, imu)
        return model(emg)

    best_val = float("inf"); stall = 0
    lam = args.consistency_lambda
    for ep in range(1, args.epochs + 1):
        model.train()
        # If frozen, force feature extractors back to eval so their BN running
        # stats stay pinned to the source-session values learned at pretrain.
        for m in frozen_modules:
            m.eval()
        tl = 0.0; tl_bce = 0.0; tl_cons = 0.0
        for batch in tqdm(train_loader, desc=f"Ep {ep}/{args.epochs} train",
                            leave=False):
            if pair_range is not None:
                # === IMU ADDITION === unpack with or without IMU pair tensors.
                if args.imu:
                    x, x_imu, x2, x2_imu, y = batch
                    x_imu = x_imu.to(device); x2_imu = x2_imu.to(device)
                else:
                    x, x2, y = batch
                    x_imu = x2_imu = None
                x = x.to(device); x2 = x2.to(device); y = y.to(device)
                # Fuse both views in one forward for efficiency.
                emg_cat = torch.cat([x, x2], dim=0)
                imu_cat = (torch.cat([x_imu, x2_imu], dim=0)
                           if args.imu else None)
                logits = forward_model(emg_cat, imu_cat)
                l1, l2 = logits[:x.size(0)], logits[x.size(0):]
                bce = 0.5 * (loss_fn(l1, y) + loss_fn(l2, y))
                cons = ((torch.sigmoid(l1) - torch.sigmoid(l2)) ** 2).mean()
                loss = bce + lam * cons
                tl_bce += bce.item(); tl_cons += cons.item()
            else:
                # === IMU ADDITION === single-view branch with optional IMU.
                if args.imu:
                    x, x_imu, y = batch
                    x_imu = x_imu.to(device)
                else:
                    x, y = batch
                    x_imu = None
                x, y = x.to(device), y.to(device)
                loss = loss_fn(forward_model(x, x_imu), y)
                tl_bce += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        n = max(1, len(train_loader))
        tl /= n; tl_bce /= n; tl_cons /= n

        model.eval()
        vl = 0.0
        correct = np.zeros(5)
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                # === IMU ADDITION ===
                if args.imu:
                    x, x_imu, y = batch
                    x_imu = x_imu.to(device)
                else:
                    x, y = batch
                    x_imu = None
                x, y = x.to(device), y.to(device)
                logits = forward_model(x, x_imu)
                vl += loss_fn(logits, y).item()
                pred = (torch.sigmoid(logits) > 0.5).float()
                correct += (pred == y).sum(dim=0).cpu().numpy()
                total += y.size(0)
        vl /= max(1, len(val_loader))
        acc = correct / max(1, total)
        extra = (f" [bce {tl_bce:.4f} cons {tl_cons:.4f}]"
                 if pair_range is not None else "")
        print(f"Ep {ep:3d}: train {tl:.4f}{extra} | val {vl:.4f} | "
              f"per-bit acc {np.round(acc, 3).tolist()} | mean {acc.mean():.3f}")

        if vl < best_val:
            best_val = vl; stall = 0
            ckpt_payload = {
                "state_dict": model.state_dict(),
                "input_mean": mean,
                "input_std": std,
                "window": args.window,
                # === IMU ADDITION === flag + IMU stats so the live predictor
                # can reconstruct the same architecture and z-score IMU
                # consistently.
                "multistream": bool(args.imu),
            }
            if args.imu:
                ckpt_payload["imu_mean"] = imu_mean
                ckpt_payload["imu_std"] = imu_std
            torch.save(ckpt_payload, save_path)
            print(f"  -> saved {save_path}")
        else:
            stall += 1
            if not args.no_early_stop and stall >= args.patience:
                print(f"Early stop at epoch {ep} (best val {best_val:.4f})")
                break

    print(f"\nBest val loss: {best_val:.4f}  ->  {save_path}")


if __name__ == "__main__":
    main()
