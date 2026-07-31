"""Offline evaluation of the trained 20-angle NeuroPose model.

Runs the model on the HDF5 test split, computes per-joint RMSE in degrees,
and saves time-series plots of predicted vs. ground-truth angles for a
random test window.

Usage:
    python -m src.eval_angles \
        --dataset data/dataset_angles_no_imu.hdf5 \
        --checkpoint models/neuropose_angles_best.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import MindroveEMGDataset
from src.landmarks_to_angles import ANGLE_NAMES
from src.train import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/dataset_angles_no_imu.hdf5")
    parser.add_argument("--checkpoint", type=str, default="models/neuropose_angles_best.pt")
    parser.add_argument("--config", type=str,
                        default="emg2pose/config/network/neuropose_angles.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=1000,
                        help="Non-overlapping windows (=window) for clean eval.")
    parser.add_argument("--plot_out", type=str, default="eval_angles_plot.png")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_project_root, args.config)
    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_project_root, args.checkpoint)
    ds_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(_project_root, args.dataset)

    # Sanity: confirm dataset target is angles, and model output dim matches
    with h5py.File(ds_path, "r") as f:
        target_type = str(f.attrs.get("target_type", "landmarks"))
        target_dim = int(f.attrs.get("target_dim", 20 if target_type == "angles" else 60))
        print(f"Dataset target_type={target_type}, target_dim={target_dim}")
        assert target_type == "angles", "This eval expects angles HDF5."

    ds = MindroveEMGDataset(ds_path, split=args.split,
                             window_length=args.window, step_size=args.step,
                             emg_only=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"{args.split} windows: {len(ds)}")
    if len(ds) == 0:
        print("No windows to evaluate.")
        return

    model = get_model(cfg_path).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # Per-joint accumulators: sum of squared errors (rad^2) and count
    sq_err_sum = np.zeros(20, dtype=np.float64)
    abs_err_sum = np.zeros(20, dtype=np.float64)
    count = 0

    # Keep one window's preds+targets for plotting
    first_batch_pred = None
    first_batch_target = None

    with torch.no_grad():
        for emg_x, target_y in loader:
            emg_x = emg_x.to(device)
            target_y = target_y.to(device)            # (B, 20, T)
            pred = model(emg_x)                        # (B, 20, T)

            err = (pred - target_y).cpu().numpy()      # (B, 20, T)
            sq_err_sum += (err ** 2).sum(axis=(0, 2))  # per joint
            abs_err_sum += np.abs(err).sum(axis=(0, 2))
            count += err.shape[0] * err.shape[2]

            if first_batch_pred is None:
                first_batch_pred = pred[0].cpu().numpy()    # (20, T)
                first_batch_target = target_y[0].cpu().numpy()

    # Per-joint RMSE (radians → degrees)
    rmse_rad = np.sqrt(sq_err_sum / count)
    mae_rad = abs_err_sum / count
    rmse_deg = np.degrees(rmse_rad)
    mae_deg = np.degrees(mae_rad)

    print("\n" + "=" * 72)
    print(f"{'Joint':<20}  {'RMSE (deg)':>12}  {'MAE (deg)':>12}")
    print("-" * 72)
    for i, name in enumerate(ANGLE_NAMES):
        print(f"{name:<20}  {rmse_deg[i]:12.2f}  {mae_deg[i]:12.2f}")
    print("-" * 72)
    print(f"{'MEAN':<20}  {rmse_deg.mean():12.2f}  {mae_deg.mean():12.2f}")
    print(f"{'MEDIAN':<20}  {np.median(rmse_deg):12.2f}  {np.median(mae_deg):12.2f}")
    print("=" * 72)

    # Plot first test window
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pred_deg = np.degrees(first_batch_pred)
        tgt_deg = np.degrees(first_batch_target)
        t = np.arange(pred_deg.shape[1]) / 500.0  # seconds @ 500 Hz

        fig, axes = plt.subplots(5, 4, figsize=(16, 12), sharex=True)
        for i, ax in enumerate(axes.flat):
            if i >= 20:
                ax.axis("off")
                continue
            ax.plot(t, tgt_deg[i], label="GT", color="black", linewidth=1.2)
            ax.plot(t, pred_deg[i], label="Pred", color="tab:red",
                    linewidth=1.0, alpha=0.85)
            ax.set_title(f"{ANGLE_NAMES[i]}  (RMSE {rmse_deg[i]:.1f}°)", fontsize=9)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("deg")
        fig.suptitle(f"NeuroPose angles — {args.split} window #0  (overall RMSE {rmse_deg.mean():.1f}°)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(args.plot_out, dpi=110)
        print(f"\nPlot saved: {args.plot_out}")
    except ImportError:
        print("\nmatplotlib not available; skipped plot.")


if __name__ == "__main__":
    main()
