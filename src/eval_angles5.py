"""Offline evaluation of the 5-angle (palm-to-finger flexion) NeuroPose model.

Targets: [0 THUMB_CMC_FE, 5 INDEX_MCP_FE, 9 MIDDLE_MCP_FE,
          13 RING_MCP_FE, 17 PINKY_MCP_FE] from the 20-angle HDF5.

Usage:
    python -m src.eval_angles5 \
        --dataset data/dataset_angles_no_imu.hdf5 \
        --checkpoint models/neuropose_angles5_best.pt
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import MindroveEMGDataset
from src.landmarks_to_angles import ANGLE_NAMES
from src.train import get_model


KEEP_INDICES = [0, 5, 9, 13, 17]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset_angles_no_imu.hdf5")
    parser.add_argument("--checkpoint", default="models/neuropose_angles5_best.pt")
    parser.add_argument("--config",
                        default="emg2pose/config/network/neuropose_angles5.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--plot_out", default="eval_angles5_plot.png")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ab = lambda p: p if os.path.isabs(p) else os.path.join(root, p)
    cfg_p, ckpt_p, ds_p = ab(args.config), ab(args.checkpoint), ab(args.dataset)

    with h5py.File(ds_p, "r") as f:
        assert str(f.attrs.get("target_type", "landmarks")) == "angles"

    names5 = [ANGLE_NAMES[i] for i in KEEP_INDICES]

    ds = MindroveEMGDataset(ds_p, split=args.split,
                             window_length=args.window, step_size=args.step,
                             emg_only=True, keep_indices=KEEP_INDICES)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"{args.split} windows: {len(ds)}")
    if len(ds) == 0:
        print("No windows to evaluate.")
        return

    model = get_model(cfg_p).to(device)
    state = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt_p}")

    N = len(KEEP_INDICES)
    sq_err_sum = np.zeros(N, dtype=np.float64)
    abs_err_sum = np.zeros(N, dtype=np.float64)
    count = 0
    first_pred = first_tgt = None

    with torch.no_grad():
        for emg_x, target_y in loader:
            emg_x = emg_x.to(device); target_y = target_y.to(device)
            pred = model(emg_x)
            err = (pred - target_y).cpu().numpy()
            sq_err_sum += (err ** 2).sum(axis=(0, 2))
            abs_err_sum += np.abs(err).sum(axis=(0, 2))
            count += err.shape[0] * err.shape[2]
            if first_pred is None:
                first_pred = pred[0].cpu().numpy()
                first_tgt = target_y[0].cpu().numpy()

    rmse_deg = np.degrees(np.sqrt(sq_err_sum / count))
    mae_deg = np.degrees(abs_err_sum / count)

    print("\n" + "=" * 60)
    print(f"{'Joint':<20}  {'RMSE(deg)':>10}  {'MAE(deg)':>10}")
    print("-" * 60)
    for i, name in enumerate(names5):
        print(f"{name:<20}  {rmse_deg[i]:10.2f}  {mae_deg[i]:10.2f}")
    print("-" * 60)
    print(f"{'MEAN':<20}  {rmse_deg.mean():10.2f}  {mae_deg.mean():10.2f}")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pred_deg = np.degrees(first_pred)
        tgt_deg = np.degrees(first_tgt)
        t = np.arange(pred_deg.shape[1]) / 500.0
        fig, axes = plt.subplots(N, 1, figsize=(10, 2.2 * N), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(t, tgt_deg[i], label="GT", color="black", linewidth=1.2)
            ax.plot(t, pred_deg[i], label="Pred", color="tab:red",
                    linewidth=1.0, alpha=0.85)
            ax.set_title(f"{names5[i]}  (RMSE {rmse_deg[i]:.1f}°)", fontsize=10)
            if i == 0:
                ax.legend(fontsize=8)
            ax.set_ylabel("deg")
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"NeuroPose-5 — {args.split}  (mean RMSE {rmse_deg.mean():.1f}°)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(args.plot_out, dpi=110)
        print(f"Plot saved: {args.plot_out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
