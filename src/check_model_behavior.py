"""Diagnostic: does the model actually respond to input, or has it collapsed?

Runs a battery of offline checks on train/val/test data:

  1. Prediction spread — std of predicted landmarks across N random windows.
     Compared to the std of the ground-truth labels on the same windows.
     If model pred std << label std, the model outputs near-constant pose.

  2. Mean-predictor baseline — MSE of "always predict the training mean pose"
     compared to the model's MSE on the same windows. If model_mse >= mean_mse,
     the model has literally learned nothing useful.

  3. Prediction-vs-truth correlation — Pearson r per landmark coord.
     If the model is tracking EMG at all, correlations should be positive
     for at least some joints (even if poorly). Near-zero everywhere ⇒ collapsed.

  4. Input perturbation — feed the model the same window scaled by 0.5× and 2×.
     A working model's output should change. A collapsed model won't care.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from src.convert_to_hdf5 import filter_emg
from src.train import get_model

NUM_EMG = 8
NUM_CH = 14
WINDOW = 1000


def load_model(model_path, config_path, device):
    m = get_model(str(config_path)).to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state)
    m.eval()
    return m


def sample_windows(h5f, split, n, rng):
    """Pull n random (window, label) pairs from split, already filtered + z-scored.
    Returns X: (n, 14, 1000), Y: (n, 60) (last-timestep label), Y_full: (n, 60, 1000).
    """
    mean = np.asarray(h5f.attrs["input_mean"], dtype=np.float32)
    std = np.asarray(h5f.attrs["input_std"], dtype=np.float32)

    sessions = list(h5f[split].keys())
    if not sessions:
        return None

    # Collect all valid start indices
    candidates = []
    for s in sessions:
        T = h5f[split][s]["emg"].shape[0]
        for start in range(0, T - WINDOW + 1, 50):
            candidates.append((s, start))
    idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)

    X, Y_last, Y_full = [], [], []
    for i in idx:
        s, start = candidates[i]
        emg = h5f[split][s]["emg"][start:start + WINDOW].astype(np.float32)  # (T, 14)
        lm = h5f[split][s]["landmarks"][start:start + WINDOW].astype(np.float32)  # (T, 60)

        # Same preprocessing as inference: filter EMG then z-score
        emg[:, :NUM_EMG] = filter_emg(emg[:, :NUM_EMG], fs=500)
        emg = (emg - mean) / std

        X.append(emg.T)          # (14, T)
        Y_last.append(lm[-1])    # (60,)
        Y_full.append(lm.T)      # (60, T)
    return np.stack(X), np.stack(Y_last), np.stack(Y_full)


def run_checks(h5_path, model_path, config_path, device="cpu", n=200, seed=0):
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    model = load_model(model_path, config_path, device)

    with h5py.File(h5_path, "r") as h5f:
        results = {}
        for split in ["train", "val", "test"]:
            if split not in h5f:
                continue
            print(f"\n===== [{split}]  sampling {n} windows =====")
            got = sample_windows(h5f, split, n, rng)
            if got is None:
                continue
            X, Y_last, Y_full = got
            X_t = torch.from_numpy(X).float().to(device)

            with torch.no_grad():
                out = model(X_t).cpu().numpy()  # (N, 60, T)
            P_last = out[:, :, -1]               # (N, 60)

            # --- 1. Prediction spread vs label spread ---
            pred_std = P_last.std(axis=0)         # (60,)
            true_std = Y_last.std(axis=0)         # (60,)
            print(f"[1] Prediction spread across windows:")
            print(f"    pred  per-coord std: min={pred_std.min():.5f} median={np.median(pred_std):.5f} max={pred_std.max():.5f}")
            print(f"    truth per-coord std: min={true_std.min():.5f} median={np.median(true_std):.5f} max={true_std.max():.5f}")
            print(f"    ratio (pred/truth) median: {np.median(pred_std / (true_std + 1e-9)):.3f}  "
                  f"(1.0 = same spread, ~0 = model ignores input)")

            # --- 2. Mean-predictor baseline ---
            model_mse = ((P_last - Y_last) ** 2).mean()
            mean_pose = Y_last.mean(axis=0)
            mean_mse = ((mean_pose[None] - Y_last) ** 2).mean()
            print(f"[2] MSE vs always-predict-mean baseline:")
            print(f"    model MSE:       {model_mse:.5f}")
            print(f"    mean-pose MSE:   {mean_mse:.5f}")
            print(f"    model/mean ratio: {model_mse / mean_mse:.3f}  "
                  f"(<1 = better than mean, =1 = collapsed, >1 = worse than mean)")

            # --- 3. Per-coord Pearson r ---
            rs = []
            for c in range(60):
                p = P_last[:, c]
                t = Y_last[:, c]
                if p.std() < 1e-8 or t.std() < 1e-8:
                    rs.append(0.0)
                else:
                    rs.append(np.corrcoef(p, t)[0, 1])
            rs = np.array(rs)
            print(f"[3] Prediction vs truth Pearson r (per coord):")
            print(f"    min={rs.min():+.3f}  median={np.median(rs):+.3f}  max={rs.max():+.3f}")
            print(f"    #coords with |r|>0.3: {int((np.abs(rs) > 0.3).sum())}/60")

            results[split] = dict(
                pred_std_median=float(np.median(pred_std)),
                true_std_median=float(np.median(true_std)),
                model_mse=float(model_mse),
                mean_mse=float(mean_mse),
                r_median=float(np.median(rs)),
            )

        # --- 4. Input perturbation on train windows ---
        print("\n===== [4] Input-scale perturbation (train, first 20 windows) =====")
        got = sample_windows(h5f, "train", 20, rng)
        if got is not None:
            X, _, _ = got
            X_t = torch.from_numpy(X).float().to(device)
            with torch.no_grad():
                P_1 = model(X_t).cpu().numpy()[:, :, -1]
                P_half = model(X_t * 0.5).cpu().numpy()[:, :, -1]
                P_2 = model(X_t * 2.0).cpu().numpy()[:, :, -1]
                P_zero = model(torch.zeros_like(X_t)).cpu().numpy()[:, :, -1]
            d_half = np.linalg.norm(P_1 - P_half, axis=1).mean()
            d_2 = np.linalg.norm(P_1 - P_2, axis=1).mean()
            d_zero = np.linalg.norm(P_1 - P_zero, axis=1).mean()
            range_per_window = np.linalg.norm(P_1, axis=1).mean()
            print(f"    mean ||pred(x) - pred(0.5x)|| = {d_half:.4f}")
            print(f"    mean ||pred(x) - pred(2.0x)|| = {d_2:.4f}")
            print(f"    mean ||pred(x) - pred(0)||    = {d_zero:.4f}")
            print(f"    mean ||pred(x)||              = {range_per_window:.4f}")
            print(f"    → how much output moves vs its own magnitude: "
                  f"0.5x={d_half/range_per_window:.3f}  2x={d_2/range_per_window:.3f}  zero={d_zero/range_per_window:.3f}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.hdf5")
    ap.add_argument("--model", default="models/neuropose_mindrove_best.pt")
    ap.add_argument("--config", default="emg2pose/config/network/neuropose_mindrove.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    run_checks(args.dataset, args.model, args.config, args.device, args.n)
