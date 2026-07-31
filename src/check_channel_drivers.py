"""Which input channels actually drive prediction motion?

For each recorded window, compute:
  - per-channel activity: std of that channel's values within the 1000-sample window
  - prediction motion: std of predicted landmarks across timesteps within the window
  - truth motion: std of ground-truth landmarks across timesteps within the window

Then correlate each channel's activity with prediction motion and truth motion
across windows. A channel the model uses should correlate with prediction
motion. A channel that carries real information should correlate with truth
motion. Comparing the two tells us whether the model is listening to the
right channels.

Also: mutual information via a simple two-bin joint histogram, as a
non-linear cross-check against Pearson r.
"""

import argparse

import h5py
import numpy as np
import torch

from src.convert_to_hdf5 import filter_emg
from src.train import get_model

NUM_EMG = 8
WINDOW = 1000
CHANNEL_NAMES = (
    [f"emg_{i}" for i in range(NUM_EMG)]
    + ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
)


def gather(h5f, split, mean, std, n, rng, stride=200):
    sessions = list(h5f[split].keys())
    X, X_raw, Y_full = [], [], []
    for s in sessions:
        emg = h5f[split][s]["emg"][:]
        lm = h5f[split][s]["landmarks"][:]
        T = len(emg)
        starts = list(range(0, T - WINDOW + 1, stride))
        rng.shuffle(starts)
        starts = starts[:n]
        for start in starts:
            e_raw = emg[start:start + WINDOW].astype(np.float32)
            # Filter EMG to match training/inference preprocessing
            e = e_raw.copy()
            e[:, :NUM_EMG] = filter_emg(e[:, :NUM_EMG], fs=500)
            # Raw-but-filtered (used for channel activity — closer to what the
            # model actually ingests than z-scored, since z-score hides absolute
            # scale we care about here).
            X_raw.append(e.T)
            # Z-scored (for the model forward pass).
            e_norm = (e - mean) / std
            X.append(e_norm.T)
            Y_full.append(lm[start:start + WINDOW].astype(np.float32).T)
    return np.stack(X), np.stack(X_raw), np.stack(Y_full)


def run_model(model, X, device, batch=32):
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(device)
        with torch.no_grad():
            o = model(xb).cpu().numpy()
        outs.append(o)
    return np.concatenate(outs, axis=0)


def mutual_info_bin(x, y, bins=4):
    """Crude MI estimate via binned joint histogram. Good enough for ordering."""
    # Digitize with quantile bins so each bin has ~equal count.
    xq = np.quantile(x, np.linspace(0, 1, bins + 1))
    yq = np.quantile(y, np.linspace(0, 1, bins + 1))
    xq[0] -= 1e-9; xq[-1] += 1e-9
    yq[0] -= 1e-9; yq[-1] += 1e-9
    xi = np.clip(np.digitize(x, xq) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, yq) - 1, 0, bins - 1)
    pxy = np.zeros((bins, bins), dtype=np.float64)
    for a, b in zip(xi, yi):
        pxy[a, b] += 1
    pxy /= pxy.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        mi = np.nansum(pxy * (np.log(pxy + 1e-12) - np.log(px + 1e-12) - np.log(py + 1e-12)))
    return float(mi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.hdf5")
    ap.add_argument("--model", default="models/neuropose_mindrove_best.pt")
    ap.add_argument("--config", default="emg2pose/config/network/neuropose_mindrove.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--emg-only", action="store_true",
                    help="Slice model input to first 8 channels (must match checkpoint's training).")
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
        X_norm, X_raw, Y_full = gather(f, args.split, mean, std, args.n, rng)

    print(f"[{args.split}] windows: {len(X_norm)}")

    # Forward pass — slice to 8 channels if model was trained EMG-only.
    X_for_model = X_norm[:, :NUM_EMG, :] if args.emg_only else X_norm
    if args.emg_only:
        print(f"--emg-only: model input sliced to {X_for_model.shape[1]} channels")
    P = run_model(model, X_for_model, device)   # (N, 60, T)

    # Per-window scalars
    pred_motion = P.std(axis=2).mean(axis=1)        # (N,) — avg over 60 coords of per-coord time-std
    truth_motion = Y_full.std(axis=2).mean(axis=1)  # (N,)

    # Per-channel activity per window (std of the channel within the window, in raw filtered units)
    ch_activity = X_raw.std(axis=2)                 # (N, 14)

    # Sanity on aggregate scales
    print(f"\nPer-window motion: pred mean={pred_motion.mean():.4f}  truth mean={truth_motion.mean():.4f}")
    print(f"Per-window pred vs truth motion correlation: r={np.corrcoef(pred_motion, truth_motion)[0,1]:+.3f}")

    print("\nChannel-by-channel correlations:")
    print(f"  {'channel':10s}  {'activity_mean':>12s}  {'r(ch, pred_mot)':>15s}  {'r(ch, truth_mot)':>17s}  {'MI(ch, pred)':>12s}  {'MI(ch, truth)':>13s}")
    rows = []
    for c in range(14):
        act = ch_activity[:, c]
        r_p = np.corrcoef(act, pred_motion)[0, 1] if act.std() > 1e-9 else 0.0
        r_t = np.corrcoef(act, truth_motion)[0, 1] if act.std() > 1e-9 else 0.0
        mi_p = mutual_info_bin(act, pred_motion)
        mi_t = mutual_info_bin(act, truth_motion)
        rows.append((CHANNEL_NAMES[c], act.mean(), r_p, r_t, mi_p, mi_t))
        print(f"  {CHANNEL_NAMES[c]:10s}  {act.mean():12.3f}  {r_p:+15.3f}  {r_t:+17.3f}  {mi_p:12.3f}  {mi_t:13.3f}")

    # Rank channels by their correlation with prediction motion, then with truth motion.
    ranked_pred = sorted(rows, key=lambda r: -abs(r[2]))
    ranked_truth = sorted(rows, key=lambda r: -abs(r[3]))
    print("\nTop-5 channels by |r(ch, PRED_motion)| — what the MODEL listens to:")
    for name, _, rp, rt, *_ in ranked_pred[:5]:
        print(f"  {name:10s}  r(pred)={rp:+.3f}   r(truth)={rt:+.3f}")
    print("\nTop-5 channels by |r(ch, TRUTH_motion)| — what the DATA actually contains:")
    for name, _, rp, rt, *_ in ranked_truth[:5]:
        print(f"  {name:10s}  r(pred)={rp:+.3f}   r(truth)={rt:+.3f}")

    # Full-channel covariance between per-window time-std of each EMG/IMU channel
    # and per-coord motion std of predictions & truth.
    # This tells us if *specific coords* of the output are driven by specific channels,
    # which is a finer-grained picture than the aggregate motion scalar.
    per_coord_motion_pred = P.std(axis=2)            # (N, 60)
    per_coord_motion_truth = Y_full.std(axis=2)      # (N, 60)

    # For each channel, find the BEST-correlated output coord.
    print("\nPer-channel → best-correlated output coord (|r| over the 60 coords):")
    for c in range(14):
        act = ch_activity[:, c]
        if act.std() < 1e-9:
            print(f"  {CHANNEL_NAMES[c]:10s}  channel flat, skipping")
            continue
        r_pred = np.array([np.corrcoef(act, per_coord_motion_pred[:, k])[0, 1] for k in range(60)])
        r_truth = np.array([np.corrcoef(act, per_coord_motion_truth[:, k])[0, 1] for k in range(60)])
        bp = np.argmax(np.abs(r_pred)); bt = np.argmax(np.abs(r_truth))
        joint_p, axis_p = bp // 3, ["x", "y", "z"][bp % 3]
        joint_t, axis_t = bt // 3, ["x", "y", "z"][bt % 3]
        print(
            f"  {CHANNEL_NAMES[c]:10s}  "
            f"best-pred: joint{joint_p:02d}_{axis_p} r={r_pred[bp]:+.3f}   "
            f"best-truth: joint{joint_t:02d}_{axis_t} r={r_truth[bt]:+.3f}"
        )


if __name__ == "__main__":
    main()
