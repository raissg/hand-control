"""Side-by-side real-time playback: ground truth vs model prediction.

Replays a recorded session at real-time speed, running the trained model on the
same EMG stream and showing the predicted 3D hand next to the ground-truth hand
from landmarks.csv.

Usage:
    PYTHONPATH=. python src/compare_playback.py 20260415_212910
    PYTHONPATH=. python src/compare_playback.py 20260415_212910 --rate 20 --speed 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.convert_to_hdf5 import filter_emg
from src.ground_truth import HAND_CONNECTIONS, WRIST
from src.train import get_model

RECORDINGS_DIR = _ROOT / "recordings"
WINDOW = 1000
NUM_EMG = 8
NUM_LM = 21


def load_session(session_id: str):
    d = RECORDINGS_DIR / session_id
    if not d.is_dir():
        raise FileNotFoundError(f"Session not found: {d}")
    emg = pd.read_csv(d / "emg.csv")
    emg.columns = [c.strip() for c in emg.columns]
    lm = pd.read_csv(d / "landmarks.csv")
    lm.columns = [c.strip() for c in lm.columns]
    return emg, lm


def _draw_hand(ax, xyz21, title, color_bones, color_pts):
    """xyz21: (21, 3) with wrist at index 0."""
    ax.clear()
    ax.set_title(title, fontsize=11)
    if xyz21 is None:
        ax.text(0, 0, 0, "no data", ha="center")
        return
    x, y, z = xyz21[:, 0], xyz21[:, 1], xyz21[:, 2]
    for a, b in HAND_CONNECTIONS:
        ax.plot([x[a], x[b]], [y[a], y[b]], [z[a], z[b]],
                color=color_bones, lw=2.5)
    ax.scatter(x, y, z, c=color_pts, s=35, depthshade=True)
    ax.scatter([x[WRIST]], [y[WRIST]], [z[WRIST]], c="tab:red", s=80)
    # fixed axis limits so motion is visible without autoscale jitter
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--model", default="models/neuropose_mindrove_best.pt")
    ap.add_argument("--config", default="emg2pose/config/network/neuropose_mindrove.yaml")
    ap.add_argument("--dataset", default="data/dataset.hdf5",
                    help="HDF5 file containing input_mean / input_std")
    ap.add_argument("--rate", type=float, default=10.0, help="Inference / viz rate (Hz)")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0 = real-time)")
    ap.add_argument("--ema", type=float, default=0.5, help="EMA smoothing for prediction (1.0 = off)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # --- Load session ---
    emg_df, lm_df = load_session(args.session_id)
    t_emg = emg_df["timestamp"].values.astype(np.float64)
    t_lm = lm_df["timestamp"].values.astype(np.float64)

    emg_cols = [f"emg_{i}" for i in range(NUM_EMG)]
    imu_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    missing_imu = [c for c in imu_cols if c not in emg_df.columns]
    if missing_imu:
        raise RuntimeError(f"emg.csv missing IMU columns: {missing_imu}")
    all_cols = emg_cols + imu_cols
    emg_arr = emg_df[all_cols].values.astype(np.float32)  # (T, 14)

    # Ground-truth landmark table (21 joints x 3 coords).
    lm_cols_21 = [c for i in range(NUM_LM) for c in (f"x{i}", f"y{i}", f"z{i}")]
    lm_arr = lm_df[lm_cols_21].values.astype(np.float32)  # (F, 63)

    # --- Load norm stats + model ---
    with h5py.File(args.dataset, "r") as f:
        mean = np.asarray(f.attrs["input_mean"], dtype=np.float32)
        std = np.asarray(f.attrs["input_std"], dtype=np.float32)

    device = torch.device(args.device)
    model = get_model(args.config).to(device)
    state = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # --- Time setup ---
    t_start = max(t_emg[0], t_lm[0])
    t_end = min(t_emg[-1], t_lm[-1])
    # Need at least WINDOW samples before we can predict; start WINDOW/500 s in.
    t_cursor_start = t_start + WINDOW / 500.0
    if t_cursor_start >= t_end:
        raise RuntimeError("Session too short to fill a 2s window.")

    # --- Figure ---
    plt.style.use("dark_background")
    fig = plt.figure("Ground truth vs Prediction", figsize=(12, 6))
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pr = fig.add_subplot(1, 2, 2, projection="3d")

    dt = 1.0 / args.rate
    last_pred = None
    wall_start = time.time()
    sess_start = t_cursor_start

    print(f"Playing {args.session_id} from t={t_cursor_start:.2f}s to t={t_end:.2f}s "
          f"at {args.rate} Hz, speed={args.speed}x")

    try:
        while plt.fignum_exists(fig.number):
            # Map wall clock -> session time
            sess_t = sess_start + (time.time() - wall_start) * args.speed
            if sess_t >= t_end:
                print("End of session.")
                break

            # --- EMG window ending at sess_t ---
            end_idx = int(np.searchsorted(t_emg, sess_t))
            start_idx = end_idx - WINDOW
            if start_idx < 0:
                time.sleep(dt)
                continue
            win = emg_arr[start_idx:end_idx].copy()
            win[:, :NUM_EMG] = filter_emg(win[:, :NUM_EMG], fs=500)
            win = (win - mean) / std
            x = torch.from_numpy(win.T).unsqueeze(0).float().to(device)

            with torch.no_grad():
                out = model(x).cpu().numpy()[0, :, -1]  # (60,)
            pred20 = out.reshape(20, 3)
            if last_pred is not None:
                pred20 = args.ema * pred20 + (1.0 - args.ema) * last_pred
            last_pred = pred20
            # prepend wrist at origin so we have 21 joints like ground truth
            pred21 = np.vstack([np.zeros((1, 3)), pred20])

            # --- Ground truth at sess_t (nearest landmark row) ---
            lm_idx = int(np.searchsorted(t_lm, sess_t))
            lm_idx = min(max(lm_idx, 0), len(lm_arr) - 1)
            gt21 = lm_arr[lm_idx].reshape(NUM_LM, 3)

            # --- Render ---
            # Let user orbit; preserve current view angle on each redraw.
            elev_g, azim_g = ax_gt.elev, ax_gt.azim
            elev_p, azim_p = ax_pr.elev, ax_pr.azim
            _draw_hand(ax_gt, gt21, f"Ground truth  t={sess_t - t_start:5.2f}s",
                       "tab:green", "white")
            _draw_hand(ax_pr, pred21, f"Model prediction",
                       "tab:cyan", "tab:orange")
            ax_gt.view_init(elev=elev_g, azim=azim_g)
            ax_pr.view_init(elev=elev_p, azim=azim_p)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # Pace to target rate
            plt.pause(max(0.001, dt))
    except KeyboardInterrupt:
        pass

    plt.close(fig)


if __name__ == "__main__":
    main()
