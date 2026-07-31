"""Train the 5-bit finger-bend classifier on the existing angles HDF5.

Labels are derived on the fly by thresholding the 5 palm-to-finger FE
angles at configurable per-joint thresholds (degrees).

Default thresholds match observed GT ranges:
    THUMB_CMC_FE  > 20°  -> bent
    INDEX_MCP_FE  > 45°
    MIDDLE_MCP_FE > 45°
    RING_MCP_FE   > 45°
    PINKY_MCP_FE  > 45°

Usage:
    python -m src.train_binary --dataset data/dataset_angles_no_imu.hdf5
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.binary_model import EMG5Bit


KEEP_INDICES = [0, 5, 9, 13, 17]
DEFAULT_THRESH_DEG = [20.0, 45.0, 45.0, 45.0, 45.0]


class BinaryEMGDataset(Dataset):
    """Windowed EMG + 5-bit labels (threshold on last-timestep angle)."""

    def __init__(self, hdf5_path, split, window_length=100, step_size=25,
                 thresh_deg=DEFAULT_THRESH_DEG, num_emg=8):
        self.path = hdf5_path
        self.split = split
        self.W = window_length
        self.step = step_size
        self.num_emg = num_emg
        self.thresh_rad = np.radians(np.asarray(thresh_deg, dtype=np.float32))
        self.keep = np.asarray(KEEP_INDICES, dtype=np.int64)

        with h5py.File(hdf5_path, "r") as f:
            if "input_mean" in f.attrs:
                self.input_mean = np.asarray(f.attrs["input_mean"], dtype=np.float32)[:num_emg]
                self.input_std = np.asarray(f.attrs["input_std"], dtype=np.float32)[:num_emg]
            else:
                self.input_mean = None; self.input_std = None
            assert str(f.attrs.get("target_type", "")) == "angles", \
                "train_binary needs an angles HDF5"
            self.samples = []
            for sname in f[split].keys():
                T = f[split][sname]["emg"].shape[0]
                for start in range(0, T - self.W + 1, self.step):
                    self.samples.append((sname, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sname, s = self.samples[idx]
        e = s + self.W
        with h5py.File(self.path, "r") as f:
            emg = f[self.split][sname]["emg"][s:e, :self.num_emg].astype(np.float32)
            ang = f[self.split][sname]["angles"][s:e].astype(np.float32)  # (W, 20) rad
        if self.input_mean is not None:
            emg = (emg - self.input_mean) / self.input_std
        # Label: median of last 25% of window, thresholded per joint
        tail = ang[-max(1, self.W // 4):, self.keep]                      # (w, 5)
        ang5 = np.median(tail, axis=0)                                    # (5,) rad
        label = (ang5 > self.thresh_rad).astype(np.float32)
        emg_t = torch.from_numpy(emg).transpose(0, 1)                     # (8, W)
        return emg_t, torch.from_numpy(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset_angles_no_imu.hdf5")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--thresh_deg", type=str,
                    default="20,45,45,45,45",
                    help="Comma-separated per-joint thresholds (degrees).")
    ap.add_argument("--save-name", default="emg5bit_best.pt")
    args = ap.parse_args()

    thresh = [float(x) for x in args.thresh_deg.split(",")]
    assert len(thresh) == 5, "need 5 thresholds"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")
    print(f"Thresholds (deg): {thresh}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_path = args.dataset if os.path.isabs(args.dataset) \
              else os.path.join(root, args.dataset)

    train_ds = BinaryEMGDataset(ds_path, "train", args.window, args.step, thresh)
    val_ds = BinaryEMGDataset(ds_path, "val", args.window, args.window, thresh)
    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")

    # Per-bit class balance report + pos_weight for BCE
    all_labels = []
    for i in range(min(len(train_ds), 5000)):
        all_labels.append(train_ds[i][1].numpy())
    all_labels = np.stack(all_labels)                                     # (N, 5)
    pos_frac = all_labels.mean(axis=0)
    print(f"Per-bit positive fraction (train sample): {pos_frac}")
    # pos_weight = neg / pos  for BCEWithLogitsLoss
    pos_weight = torch.tensor(
        [(1 - p) / max(p, 1e-3) for p in pos_frac], dtype=torch.float32
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = EMG5Bit(in_channels=8, n_out=5, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, args.save_name)

    best_val = float("inf"); stall = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for x, y in tqdm(train_loader, desc=f"Ep {ep}/{args.epochs} train"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= len(train_loader)

        model.eval()
        vl = 0.0; correct = np.zeros(5); total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vl += loss_fn(logits, y).item()
                pred = (torch.sigmoid(logits) > 0.5).float()
                correct += (pred == y).sum(dim=0).cpu().numpy()
                total += y.size(0)
        vl /= len(val_loader)
        acc = correct / total
        print(f"Ep {ep}: train {tl:.4f} | val {vl:.4f} | "
              f"per-bit acc: {np.round(acc, 3).tolist()} | mean {acc.mean():.3f}")

        if vl < best_val:
            best_val = vl; stall = 0
            torch.save({"state_dict": model.state_dict(),
                        "thresh_deg": thresh,
                        "window": args.window},
                       save_path)
            print(f"  -> saved {save_path}")
        else:
            stall += 1
            if stall >= args.patience:
                print(f"Early stop at epoch {ep} (best val {best_val:.4f})")
                break


if __name__ == "__main__":
    main()
