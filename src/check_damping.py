"""Three experiments on the damping problem:

1. Per-coord linear calibration (gain + bias) fit on train, measured on val/test.
2. Output amplitude by timestep position within the 1000-sample window.
3. EMG energy vs prediction amplitude correlation.
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


def gather(h5f, split, mean, std, n, rng, stride=200):
    sessions = list(h5f[split].keys())
    X_list, Y_full_list = [], []
    for s in sessions:
        emg = h5f[split][s]["emg"][:]
        lm = h5f[split][s]["landmarks"][:]
        T = len(emg)
        starts = list(range(0, T - WINDOW + 1, stride))
        rng.shuffle(starts)
        starts = starts[:n]
        for start in starts:
            e = emg[start:start + WINDOW].astype(np.float32)
            l = lm[start:start + WINDOW].astype(np.float32)
            e[:, :NUM_EMG] = filter_emg(e[:, :NUM_EMG], fs=500)
            e = (e - mean) / std
            X_list.append(e.T)
            Y_full_list.append(l.T)
    return np.stack(X_list), np.stack(Y_full_list)


def run_model(model, X, device, batch=32):
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i+batch]).float().to(device)
        with torch.no_grad():
            o = model(xb).cpu().numpy()  # (B, 60, T)
        outs.append(o)
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.hdf5")
    ap.add_argument("--model", default="models/neuropose_mindrove_best.pt")
    ap.add_argument("--config", default="emg2pose/config/network/neuropose_mindrove.yaml")
    ap.add_argument("--n", type=int, default=300, help="windows per split")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
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
        X_tr, Y_tr = gather(f, "train", mean, std, args.n, rng)
        X_va, Y_va = gather(f, "val", mean, std, args.n, rng)
        X_te, Y_te = gather(f, "test", mean, std, args.n, rng)

    print(f"Train windows: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")

    # Run model once, get full (B, 60, 1000)
    P_tr = run_model(model, X_tr, device)
    P_va = run_model(model, X_va, device)
    P_te = run_model(model, X_te, device)

    # Use center timestep by default (will be overridden by experiment 2)
    CENTER = WINDOW // 2
    LAST = WINDOW - 1

    # =============================================================
    # EXPERIMENT 1 — per-coord linear calibration
    # =============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 — Per-coord linear calibration (fit on train, apply to val/test)")
    print("=" * 60)

    # Flatten over (windows, timesteps) so the fit uses ALL samples, not just centers.
    p_tr = P_tr.transpose(0, 2, 1).reshape(-1, 60)
    y_tr = Y_tr.transpose(0, 2, 1).reshape(-1, 60)

    # Fit gain & bias per coord
    # y ≈ gain * p + bias
    gain = np.zeros(60, dtype=np.float32)
    bias = np.zeros(60, dtype=np.float32)
    for c in range(60):
        p = p_tr[:, c]
        y = y_tr[:, c]
        v = p.var()
        if v < 1e-8:
            gain[c] = 0.0
            bias[c] = y.mean()
        else:
            gain[c] = np.cov(p, y, bias=True)[0, 1] / v
            bias[c] = y.mean() - gain[c] * p.mean()
    print(f"Fitted gains:  min={gain.min():.3f}  median={np.median(gain):.3f}  max={gain.max():.3f}  mean={gain.mean():.3f}")
    print(f"Fitted biases: min={bias.min():+.3f} median={np.median(bias):+.3f} max={bias.max():+.3f}")

    def calibrated_mse(P, Y):
        # apply per-coord gain/bias along time axis
        Pc = P * gain[None, :, None] + bias[None, :, None]
        raw = ((P - Y) ** 2).mean()
        cal = ((Pc - Y) ** 2).mean()
        return raw, cal

    for name, P, Y in [("train", P_tr, Y_tr), ("val", P_va, Y_va), ("test", P_te, Y_te)]:
        raw, cal = calibrated_mse(P, Y)
        mean_pose = y_tr.mean(axis=0)  # constant-prediction baseline
        mean_mse = ((mean_pose[None, :, None] - Y) ** 2).mean()
        print(f"  {name:5s}: raw MSE={raw:.5f}   calibrated MSE={cal:.5f}   "
              f"always-mean MSE={mean_mse:.5f}   calibrated/raw={cal/raw:.3f}")

    # =============================================================
    # EXPERIMENT 2 — timestep-position sweep
    # =============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — Output amplitude & MSE by timestep position within window")
    print("=" * 60)
    positions = [0, 100, 250, 500, 750, 900, 999]
    print(f"  {'t':>5s}  {'pred_std':>10s}  {'truth_std':>10s}  {'ratio':>6s}  {'mse':>10s}")
    for t in positions:
        p = P_tr[:, :, t]     # (N, 60)
        y = Y_tr[:, :, t]
        pstd = p.std(axis=0).mean()
        ystd = y.std(axis=0).mean()
        mse = ((p - y) ** 2).mean()
        print(f"  {t:5d}  {pstd:10.5f}  {ystd:10.5f}  {pstd/(ystd+1e-9):6.3f}  {mse:10.5f}")

    # =============================================================
    # EXPERIMENT 3 — EMG energy vs prediction amplitude
    # =============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 — Does prediction amplitude follow EMG energy?")
    print("=" * 60)
    # EMG energy per window (z-scored) over the 8 EMG channels, summed RMS.
    emg_energy = np.sqrt((X_tr[:, :NUM_EMG, :] ** 2).mean(axis=(1, 2)))  # (N,)
    # Prediction amplitude: per-window stdev across timesteps (how much the hand moves during the window)
    pred_amp = P_tr.std(axis=2).mean(axis=1)   # (N,)
    truth_amp = Y_tr.std(axis=2).mean(axis=1)  # (N,)

    r_pt = np.corrcoef(emg_energy, pred_amp)[0, 1]
    r_tt = np.corrcoef(emg_energy, truth_amp)[0, 1]
    r_pp = np.corrcoef(pred_amp, truth_amp)[0, 1]
    print(f"  Pearson r(EMG_energy, pred_amp)   = {r_pt:+.3f}")
    print(f"  Pearson r(EMG_energy, truth_amp)  = {r_tt:+.3f}")
    print(f"  Pearson r(pred_amp, truth_amp)    = {r_pp:+.3f}")
    print(f"  pred_amp  range: [{pred_amp.min():.4f}, {pred_amp.max():.4f}]  mean={pred_amp.mean():.4f}")
    print(f"  truth_amp range: [{truth_amp.min():.4f}, {truth_amp.max():.4f}]  mean={truth_amp.mean():.4f}")
    print(f"  pred_amp / truth_amp mean ratio: {(pred_amp / (truth_amp + 1e-9)).mean():.3f}")

    # Split windows into high- vs low-energy halves; what's the damping in each?
    hi = emg_energy > np.median(emg_energy)
    print(f"  High-EMG windows (n={hi.sum()}): pred_amp={pred_amp[hi].mean():.4f}  truth_amp={truth_amp[hi].mean():.4f}  ratio={pred_amp[hi].mean()/truth_amp[hi].mean():.3f}")
    print(f"  Low-EMG  windows (n={(~hi).sum()}): pred_amp={pred_amp[~hi].mean():.4f}  truth_amp={truth_amp[~hi].mean():.4f}  ratio={pred_amp[~hi].mean()/truth_amp[~hi].mean():.3f}")


if __name__ == "__main__":
    main()
