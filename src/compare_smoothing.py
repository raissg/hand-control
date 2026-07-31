"""Apples-to-apples comparison of the two trained checkpoints across
all post-hoc smoothing configurations.

For each (model × smoothing) combo we report on held-out VALIDATION segments:
  - frame-level accuracy (per-bit threshold 0.5 after smoothing)
  - spike rate (|Δp| > 0.5 per adjacent-frame pair — how often the raw prob jumps)
  - bit-flip rate (bit transitions per second within a constant-label segment)
  - wrong-run p50 / p95 (lengths of consecutive mis-predicted frames)

The val split exactly mirrors the training-time split (same seed / val_frac).

Usage:
    python -m src.compare_smoothing
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.binary_model import EMG5Bit
from src.dataset_bits import load_sessions, split_segments, Segment

NUM_EMG = 8
SAMPLE_RATE = 500
LABELS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]


def sliding_predict(model, seg: Segment, mean, std, window, hop, device) -> np.ndarray:
    T = seg.emg.shape[0]
    if T < window:
        return np.zeros((0, 5), dtype=np.float32)
    starts = np.arange(0, T - window + 1, hop)
    X = np.empty((len(starts), NUM_EMG, window), dtype=np.float32)
    for i, s in enumerate(starts):
        w = (seg.emg[s:s + window] - mean) / std
        X[i] = w.T
    out = []
    with torch.no_grad():
        B = 512
        for i in range(0, len(X), B):
            xb = torch.from_numpy(X[i:i + B]).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def smooth(probs: np.ndarray, median_k: int, ema_alpha: float) -> np.ndarray:
    """Apply median filter (per bit, rolling) then EMA. Matches live_binary."""
    N, B = probs.shape
    out = probs.copy()

    # Median filter: causal rolling window of size K
    if median_k > 1:
        med = np.empty_like(out)
        for n in range(N):
            lo = max(0, n - median_k + 1)
            med[n] = np.median(out[lo:n + 1], axis=0)
        out = med

    # EMA (α=1 disables)
    if ema_alpha < 1.0:
        ema = out.copy()
        for n in range(1, N):
            ema[n] = ema_alpha * out[n] + (1.0 - ema_alpha) * ema[n - 1]
        out = ema

    return out


def apply_hysteresis(probs: np.ndarray, on_thr: float, off_thr: float) -> np.ndarray:
    N, B = probs.shape
    bits = np.zeros_like(probs, dtype=np.int32)
    state = np.zeros(B, dtype=np.int32)
    for n in range(N):
        for b in range(B):
            p = probs[n, b]
            if p > on_thr:
                state[b] = 1
            elif p < off_thr:
                state[b] = 0
            bits[n, b] = state[b]
    return bits


def run_lengths(series: np.ndarray, target: int) -> list[int]:
    out = []
    cur = 0
    for v in series:
        if v == target:
            cur += 1
        else:
            if cur:
                out.append(cur); cur = 0
    if cur:
        out.append(cur)
    return out


def eval_model_config(model, segs, mean, std, window, hop_samples,
                       median_k, ema_alpha, use_hysteresis, device) -> dict:
    # hop_samples: inference hop (e.g. 25 for 50ms)
    hop_s = hop_samples / SAMPLE_RATE
    per_bit_acc = np.zeros(5)
    per_bit_frames = 0
    spikes = np.zeros(5, dtype=np.int64)
    n_diffs = 0
    flips = np.zeros(5, dtype=np.int64)
    total_secs = 0.0
    wrong_runs: list[list[int]] = [[] for _ in range(5)]

    for seg in segs:
        probs_raw = sliding_predict(model, seg, mean, std, window, hop_samples, device)
        if probs_raw.shape[0] < 2:
            continue

        probs = smooth(probs_raw, median_k, ema_alpha)

        if use_hysteresis:
            bits = apply_hysteresis(probs, 0.6, 0.4)
        else:
            bits = (probs > 0.5).astype(np.int32)

        per_bit_frames += probs.shape[0]
        total_secs += probs.shape[0] * hop_s

        for b in range(5):
            true = int(seg.label[b])
            per_bit_acc[b] += (bits[:, b] == true).sum()
            # spikes on the SMOOTHED prob stream (that's what hysteresis sees)
            dp = np.abs(np.diff(probs[:, b]))
            spikes[b] += int((dp > 0.5).sum())
            # bit-flips: number of 0<->1 transitions in the predicted bit stream
            flips[b] += int(np.abs(np.diff(bits[:, b])).sum())
            wrong = (bits[:, b] != true).astype(np.int32)
            wrong_runs[b].extend(run_lengths(wrong, 1))

        n_diffs += probs.shape[0] - 1

    acc = per_bit_acc / max(1, per_bit_frames)
    spike_rate = spikes / max(1, n_diffs)
    flip_per_sec = flips / max(1e-6, total_secs)
    wr_p50 = [int(np.median(r)) if r else 0 for r in wrong_runs]
    wr_p95 = [int(np.percentile(r, 95)) if r else 0 for r in wrong_runs]
    return {
        "acc": acc,
        "acc_mean": float(acc.mean()),
        "spike_rate": spike_rate,
        "spike_mean": float(spike_rate.mean()),
        "flip_per_sec": flip_per_sec,
        "flip_sum_per_sec": float(flip_per_sec.sum()),
        "wr_p50": wr_p50,
        "wr_p95": wr_p95,
    }


def load_ckpt(path: str, device) -> tuple:
    ckpt = torch.load(path, map_location=device)
    model = EMG5Bit(in_channels=NUM_EMG, n_out=5).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = np.asarray(ckpt["input_mean"], dtype=np.float32)[:NUM_EMG]
    std = np.asarray(ckpt["input_std"], dtype=np.float32)[:NUM_EMG]
    window = int(ckpt["window"])
    return model, mean, std, window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="models/emg5bit_selflabel_best.pt")
    ap.add_argument("--smooth", default="models/emg5bit_smooth_best.pt")
    ap.add_argument("--sessions", default=(
        "recordings/bits_20260420_112939/emg.csv,"
        "recordings/bits_20260420_113810/emg.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--hop-samples", type=int, default=25, help="50ms at 500Hz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    device = torch.device(args.device)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ab = lambda p: p if os.path.isabs(p) else os.path.join(root, p)

    print("Loading sessions...")
    segs = load_sessions(args.sessions, gap_s=0.1)
    _, val_segs = split_segments(segs, val_frac=args.val_frac, seed=args.seed)
    print(f"Val segments: {len(val_segs)} / {len(segs)}")

    baseline = load_ckpt(ab(args.baseline), device)
    smoothm = load_ckpt(ab(args.smooth), device)
    print(f"Baseline checkpoint window={baseline[3]} | Smooth checkpoint window={smoothm[3]}")

    # Smoothing configurations to try (applied at inference time)
    smooth_configs = [
        # (median_k, ema_alpha, use_hysteresis, name)
        (1,  1.0,  False, "raw (thr=0.5)"),
        (1,  1.0,  True,  "hyst 0.6/0.4"),
        (1,  0.35, False, "ema α=0.35"),
        (5,  1.0,  False, "median K=5"),
        (5,  0.35, False, "median K=5 + ema α=0.35"),
        (5,  0.35, True,  "median K=5 + ema α=0.35 + hyst"),
    ]

    models = [
        ("baseline (200ms, no consistency)", baseline),
        ("smooth    (400ms, λ=1.0)",         smoothm),
    ]

    print("\n" + "=" * 120)
    header = f"{'model':<38}{'smoothing':<32}{'acc':>6}  {'spike%':>7}  {'flips/s':>8}  {'wr_p50':>8}  {'wr_p95':>8}"
    print(header)
    print("-" * 120)

    rows = []
    for mname, (mdl, mean, std, win) in models:
        for k, a, hyst, cfg_name in smooth_configs:
            r = eval_model_config(mdl, val_segs, mean, std, win,
                                    args.hop_samples, k, a, hyst, device)
            rows.append((mname, cfg_name, r))
            wr50 = ",".join(str(x) for x in r["wr_p50"])
            wr95 = ",".join(str(x) for x in r["wr_p95"])
            print(f"{mname:<38}{cfg_name:<32}"
                  f"{r['acc_mean']:>6.3f}  "
                  f"{r['spike_mean']*100:>6.2f}%  "
                  f"{r['flip_sum_per_sec']:>7.2f}  "
                  f"{wr50:>8}  "
                  f"{wr95:>8}")
    print("=" * 120)

    # Highlight best-of by flip rate (perceived smoothness) while acc>=0.88
    candidates = [(r[2]["flip_sum_per_sec"], r) for r in rows if r[2]["acc_mean"] >= 0.88]
    candidates.sort(key=lambda t: t[0])
    print("\nBest by perceived smoothness (acc ≥ 0.88):")
    for fl, (mname, cfg, r) in candidates[:4]:
        print(f"  flips/s={fl:.2f}  acc={r['acc_mean']:.3f}  "
              f"spike%={r['spike_mean']*100:.2f}  [{mname}]  [{cfg}]")


if __name__ == "__main__":
    main()
