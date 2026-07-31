"""Simulate real-time inference on a recorded session and measure how the model
tracks ground truth over time.

Feeds a recorded session through the model in streaming fashion (1000-sample
window, shifts by `stride` samples per step), then prints:
  - per-step output change (delta) — "how much does the pose prediction move"
  - running MSE of last-timestep prediction vs last-timestep truth
  - correlation between predicted and true fingertip trajectories
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from src.convert_to_hdf5 import filter_emg
from src.train import get_model

NUM_EMG = 8
WINDOW = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.hdf5")
    ap.add_argument("--model", default="models/neuropose_mindrove_best.pt")
    ap.add_argument("--config", default="emg2pose/config/network/neuropose_mindrove.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stride", type=int, default=50, help="samples advanced per step (50 @ 500Hz = 10Hz)")
    ap.add_argument("--n_steps", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cpu")
    model = get_model(args.config).to(device)
    state = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    with h5py.File(args.dataset, "r") as f:
        mean = np.asarray(f.attrs["input_mean"], dtype=np.float32)
        std = np.asarray(f.attrs["input_std"], dtype=np.float32)
        sess = list(f[args.split].keys())[0]
        emg_all = f[args.split][sess]["emg"][:].astype(np.float32)
        lm_all = f[args.split][sess]["landmarks"][:].astype(np.float32)

    print(f"Session: {sess}  length {len(emg_all)} samples ({len(emg_all)/500:.1f}s)")
    print(f"Stride {args.stride} samples = {500/args.stride:.1f} Hz inference rate")

    preds = []
    truths = []
    prev_pred = None
    deltas = []
    with torch.no_grad():
        for step in range(args.n_steps):
            start = step * args.stride
            end = start + WINDOW
            if end > len(emg_all):
                break
            win = emg_all[start:end].copy()
            win[:, :NUM_EMG] = filter_emg(win[:, :NUM_EMG], fs=500)
            win = (win - mean) / std
            x = torch.from_numpy(win.T).unsqueeze(0).float()
            out = model(x).numpy()[0, :, -1]  # (60,)
            truth = lm_all[end - 1]             # (60,)

            preds.append(out)
            truths.append(truth)
            if prev_pred is not None:
                deltas.append(np.linalg.norm(out - prev_pred))
            prev_pred = out

    preds = np.stack(preds)
    truths = np.stack(truths)
    deltas = np.array(deltas)

    print(f"\nSteps: {len(preds)}")
    print(f"Prediction delta between consecutive steps: mean={deltas.mean():.4f} std={deltas.std():.4f} max={deltas.max():.4f}")
    print(f"Truth      delta between consecutive steps: "
          f"mean={np.linalg.norm(np.diff(truths, axis=0), axis=1).mean():.4f}")
    print(f"MSE (last-timestep) over stream: {((preds - truths) ** 2).mean():.4f}")

    # Fingertip tracking (indices 3=thumb, 7=index, 11=middle, 15=ring, 19=pinky after wrist removed)
    print("\nFingertip tracking (correlation of 3D position, pred vs truth):")
    tips = {"thumb": 3, "index": 7, "middle": 11, "ring": 15, "pinky": 19}
    for name, j in tips.items():
        for k, axis in enumerate(["x", "y", "z"]):
            p = preds[:, j * 3 + k]
            t = truths[:, j * 3 + k]
            if p.std() < 1e-8 or t.std() < 1e-8:
                r = 0.0
            else:
                r = np.corrcoef(p, t)[0, 1]
            print(f"  {name:6s} {axis}: pred_range=[{p.min():+.3f}, {p.max():+.3f}]  "
                  f"truth_range=[{t.min():+.3f}, {t.max():+.3f}]  r={r:+.3f}")

    # Show a sample trajectory
    print("\nIndex fingertip Y over first 40 steps (pred vs truth):")
    idx_y = 7 * 3 + 1
    for i in range(0, min(40, len(preds)), 2):
        p = preds[i, idx_y]
        t = truths[i, idx_y]
        bar_p = "·" * int(max(0, min(30, (p + 1) * 15)))
        bar_t = "·" * int(max(0, min(30, (t + 1) * 15)))
        print(f"  t={i*args.stride/500:5.2f}s  pred={p:+.3f} |{bar_p:30s}|  truth={t:+.3f} |{bar_t:30s}|")


if __name__ == "__main__":
    main()
