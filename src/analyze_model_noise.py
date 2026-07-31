"""Characterize the current 5-bit model's output noise on the training sessions.

Goal: quantify the "jumpy probability" failure mode using real data, so we can
pick a training-time fix grounded in what's actually happening.

Measures (per bit and per session):
  - Within-segment prob std (how much does prob wobble in a stable pose)
  - Single-frame jump distribution (|p_t - p_{t-1}|): histogram + tail stats
  - Spike rate: fraction of frames where |p_t - p_{t-1}| > 0.5
  - Autocorrelation at lags 1, 2, 4, 8
  - Confusion: accuracy at 0.5 threshold, compared to the segment's true label
  - Duration of "bit-wrong" episodes (how many consecutive frames does a
    misprediction last?)

Runs the trained checkpoint on *every* hop-step window of both sessions.
Writes a text report + a PNG per session.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.binary_model import EMG5Bit
from src.dataset_bits import load_sessions, Segment

NUM_EMG = 8
SAMPLE_RATE = 500
LABELS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]


def sliding_predict(model, seg: Segment, mean, std, window: int,
                    hop: int, device) -> np.ndarray:
    """Run model over every hop-step window of a segment.

    Returns probs array of shape (N_windows, 5).
    """
    T = seg.emg.shape[0]
    if T < window:
        return np.zeros((0, 5), dtype=np.float32)

    starts = np.arange(0, T - window + 1, hop)
    X = np.empty((len(starts), NUM_EMG, window), dtype=np.float32)
    for i, s in enumerate(starts):
        w = (seg.emg[s:s + window] - mean) / std
        X[i] = w.T

    probs_all = []
    model.eval()
    with torch.no_grad():
        # Batch to avoid OOM
        B = 512
        for i in range(0, len(X), B):
            xb = torch.from_numpy(X[i:i + B]).to(device)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            probs_all.append(p)
    return np.concatenate(probs_all, axis=0).astype(np.float32)


def run_length_episodes(series: np.ndarray, target: int) -> list[int]:
    """Lengths of runs where series == target."""
    out = []
    cur = 0
    for v in series:
        if v == target:
            cur += 1
        else:
            if cur:
                out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def analyze_session(probs_by_seg: list[tuple[Segment, np.ndarray]],
                     name: str, out_png: str, hop_ms: float) -> dict:
    """Returns a dict of aggregated per-bit stats for the session."""
    # Concatenate to global streams BUT remember segment boundaries so
    # jump stats don't cross them (within-seg stats only).
    per_bit_std = np.zeros(5)
    per_bit_acc = np.zeros(5)
    per_bit_frames = 0

    jumps_per_bit: list[list[float]] = [[] for _ in range(5)]
    spike_counts = np.zeros(5, dtype=np.int64)
    wrong_run_lens: list[list[int]] = [[] for _ in range(5)]
    autocorr_lags = [1, 2, 4, 8, 16]
    per_bit_autocorr: list[dict[int, list[float]]] = [
        {lag: [] for lag in autocorr_lags} for _ in range(5)
    ]

    total_spike_thresh = 0.5

    for seg, probs in probs_by_seg:
        if probs.shape[0] < 4:
            continue
        per_bit_frames += probs.shape[0]
        for b in range(5):
            ps = probs[:, b]
            # (a) within-segment std
            per_bit_std[b] += ps.std() * probs.shape[0]
            # (b) jump magnitudes
            dp = np.abs(np.diff(ps))
            jumps_per_bit[b].extend(dp.tolist())
            spike_counts[b] += int((dp > total_spike_thresh).sum())
            # (c) accuracy at 0.5
            pred = (ps > 0.5).astype(np.int32)
            true = int(seg.label[b])
            per_bit_acc[b] += (pred == true).sum()
            # (d) wrong-episode lengths
            wrong = (pred != true).astype(np.int32)
            wrong_run_lens[b].extend(run_length_episodes(wrong, 1))
            # (e) autocorrelation
            for lag in autocorr_lags:
                if ps.shape[0] > lag + 1:
                    a = ps[:-lag]
                    b_ = ps[lag:]
                    a_c = a - a.mean()
                    b_c = b_ - b_.mean()
                    den = (a_c.std() * b_c.std() * len(a))
                    if den > 1e-8:
                        per_bit_autocorr[b][lag].append(float((a_c * b_c).sum() / den))

    per_bit_std /= max(1, per_bit_frames)
    per_bit_acc /= max(1, per_bit_frames)

    # Spike rate = spikes / transitions (= frames - n_segs)
    n_diffs = sum(p.shape[0] - 1 for _, p in probs_by_seg if p.shape[0] >= 2)
    spike_rate = spike_counts / max(1, n_diffs)

    # ---------- Plot ----------
    fig, axes = plt.subplots(5, 2, figsize=(14, 14),
                              gridspec_kw={"width_ratios": [3, 1]})
    for b in range(5):
        ax_ts, ax_hist = axes[b]
        # Time-series from the concatenation (marked by segment edges)
        offset = 0
        for seg, probs in probs_by_seg:
            if probs.shape[0] == 0:
                continue
            t = np.arange(offset, offset + probs.shape[0]) * hop_ms / 1000.0
            ax_ts.plot(t, probs[:, b], lw=0.6, alpha=0.8)
            ax_ts.axhline(seg.label[b], color="k", lw=0.3, alpha=0.1)
            offset += probs.shape[0]
        ax_ts.set_ylim(-0.05, 1.05)
        ax_ts.set_ylabel(f"p({LABELS[b]})")
        ax_ts.axhline(0.5, color="r", lw=0.5, alpha=0.5)
        ax_ts.set_title(
            f"{LABELS[b]}: acc={per_bit_acc[b]:.3f}  "
            f"spike>{total_spike_thresh:.1f}/frame={spike_rate[b]:.3%}  "
            f"std={per_bit_std[b]:.3f}"
        )
        # Histogram of adjacent-frame jumps
        jm = np.asarray(jumps_per_bit[b])
        ax_hist.hist(jm, bins=50, log=True)
        ax_hist.set_xlabel("|Δp|")
        ax_hist.set_ylabel("log count")
        ax_hist.set_xlim(0, 1)

    axes[-1, 0].set_xlabel("time (s) — concatenated segments")
    fig.suptitle(f"Session: {name}  (hop {hop_ms:.0f}ms)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    plt.close(fig)

    # ---------- Summary ----------
    report = {
        "name": name,
        "n_segs": len(probs_by_seg),
        "n_frames": per_bit_frames,
        "per_bit_std": per_bit_std.round(4).tolist(),
        "per_bit_acc": per_bit_acc.round(4).tolist(),
        "spike_rate_>0.5": spike_rate.round(5).tolist(),
        "autocorr": {
            lag: [
                round(float(np.mean(per_bit_autocorr[b][lag])), 4)
                if per_bit_autocorr[b][lag] else None
                for b in range(5)
            ] for lag in autocorr_lags
        },
        "wrong_run_median_frames": [
            int(np.median(r)) if r else 0 for r in wrong_run_lens
        ],
        "wrong_run_p95_frames": [
            int(np.percentile(r, 95)) if r else 0 for r in wrong_run_lens
        ],
        "jump_p99": [
            round(float(np.percentile(np.asarray(j), 99)), 4) if j else 0.0
            for j in jumps_per_bit
        ],
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/emg5bit_selflabel_best.pt")
    ap.add_argument("--sessions", default=(
        "recordings/bits_20260420_112939/emg.csv,"
        "recordings/bits_20260420_113810/emg.csv"))
    ap.add_argument("--hop", type=int, default=25,
                     help="inference hop in samples (default 25 = 50ms)")
    ap.add_argument("--out-dir", default="data/noise_analysis")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    device = torch.device(args.device)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, args.out_dir), exist_ok=True)

    # Load checkpoint
    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(root, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    mean = np.asarray(ckpt["input_mean"], dtype=np.float32)[:NUM_EMG]
    std = np.asarray(ckpt["input_std"], dtype=np.float32)[:NUM_EMG]
    window = int(ckpt["window"])
    print(f"Loaded checkpoint: window={window}, mean={mean.round(2)}, std={std.round(2)}")

    model = EMG5Bit(in_channels=NUM_EMG, n_out=5).to(device)
    model.load_state_dict(state)
    model.eval()

    # Load segments (filtering done inside load_sessions)
    segs = load_sessions(args.sessions, gap_s=0.1)
    print(f"Loaded {len(segs)} segments from {args.sessions}")

    # Group by session
    by_session: dict[str, list[Segment]] = {}
    for s in segs:
        by_session.setdefault(s.source, []).append(s)

    hop_ms = args.hop * 1000.0 / SAMPLE_RATE

    reports = []
    for sess_name, sess_segs in sorted(by_session.items()):
        probs_by_seg = []
        for seg in sess_segs:
            probs = sliding_predict(model, seg, mean, std, window, args.hop, device)
            probs_by_seg.append((seg, probs))
        out_png = os.path.join(root, args.out_dir, f"{sess_name}.png")
        rep = analyze_session(probs_by_seg, sess_name, out_png, hop_ms)
        reports.append(rep)

    print("\n" + "=" * 70)
    for r in reports:
        print(f"\nSession: {r['name']}")
        print(f"  segments: {r['n_segs']}  frames: {r['n_frames']}")
        print(f"  per-bit  acc        : {r['per_bit_acc']}")
        print(f"  per-bit  prob-std   : {r['per_bit_std']}")
        print(f"  per-bit  spike>0.5  : {r['spike_rate_>0.5']}   (spikes / transitions)")
        print(f"  per-bit  jump p99   : {r['jump_p99']}")
        print(f"  per-bit  wrong-run median: {r['wrong_run_median_frames']} frames  "
              f"(p95={r['wrong_run_p95_frames']})")
        print(f"  autocorr (lag->per-bit):")
        for lag in [1, 2, 4, 8, 16]:
            vals = r['autocorr'][lag]
            print(f"    lag={lag:2d}  {vals}")

    print(f"\nPNGs -> {os.path.join(root, args.out_dir)}")


if __name__ == "__main__":
    main()
