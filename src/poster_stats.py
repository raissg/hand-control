"""One-shot evaluation for poster numbers.

Evaluates each trained model on an IDENTICAL held-out validation set
(3 sessions, same seed, same val_frac) so all numbers are apples-to-apples.

Per (model × smoothing) combo, reports on the same val segments:
  - per-bit val accuracy (per-finger + mean)
  - exact-match gesture accuracy (all 5 bits simultaneously correct)
  - flicker rate (bit-flips / second during constant-label holds)
  - spike rate on the smoothed prob stream (|Δp| > 0.5)
  - wrong-run p50 / p95 (lengths of consecutive mis-predicted frames)

Output format is a single clean table for pasting into the poster.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from src.binary_model import EMG5Bit
from src.dataset_bits import load_sessions, split_segments, Segment

NUM_EMG = 8
SAMPLE_RATE = 500
LABELS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
HOP_SAMPLES = 25  # 50 ms at 500 Hz


def sliding_predict(model, seg: Segment, mean, std, window, hop, device):
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
    N, B = probs.shape
    out = probs.copy()
    if median_k > 1:
        med = np.empty_like(out)
        for n in range(N):
            lo = max(0, n - median_k + 1)
            med[n] = np.median(out[lo:n + 1], axis=0)
        out = med
    if ema_alpha < 1.0:
        ema = out.copy()
        for n in range(1, N):
            ema[n] = ema_alpha * out[n] + (1.0 - ema_alpha) * ema[n - 1]
        out = ema
    return out


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


def eval_config(model, segs, mean, std, window, hop, median_k, ema_alpha, device):
    per_bit_correct = np.zeros(5)
    exact_correct = 0
    total_frames = 0
    total_secs = 0.0
    spikes = np.zeros(5, dtype=np.int64)
    n_diffs = 0
    flips = np.zeros(5, dtype=np.int64)
    wrong_runs: list[list[int]] = [[] for _ in range(5)]
    hop_s = hop / SAMPLE_RATE

    for seg in segs:
        probs_raw = sliding_predict(model, seg, mean, std, window, hop, device)
        if probs_raw.shape[0] < 2:
            continue
        probs = smooth(probs_raw, median_k, ema_alpha)
        bits = (probs > 0.5).astype(np.int32)
        N = probs.shape[0]
        total_frames += N
        total_secs += N * hop_s

        true_bits = seg.label.astype(np.int32)
        per_bit_correct += (bits == true_bits[None, :]).sum(axis=0)
        exact_correct += (bits == true_bits[None, :]).all(axis=1).sum()

        for b in range(5):
            dp = np.abs(np.diff(probs[:, b]))
            spikes[b] += int((dp > 0.5).sum())
            flips[b] += int(np.abs(np.diff(bits[:, b])).sum())
            wrong = (bits[:, b] != true_bits[b]).astype(np.int32)
            wrong_runs[b].extend(run_lengths(wrong, 1))
        n_diffs += N - 1

    acc_per_bit = per_bit_correct / max(1, total_frames)
    exact_match = exact_correct / max(1, total_frames)
    flip_per_sec = flips / max(1e-6, total_secs)
    spike_rate = spikes / max(1, n_diffs)
    wr_p50 = [int(np.median(r)) if r else 0 for r in wrong_runs]
    wr_p95 = [int(np.percentile(r, 95)) if r else 0 for r in wrong_runs]
    return {
        "acc_per_bit": acc_per_bit,
        "acc_mean": float(acc_per_bit.mean()),
        "exact_match": float(exact_match),
        "flip_sum_per_sec": float(flip_per_sec.sum()),
        "flip_per_bit_per_sec": flip_per_sec,
        "spike_mean": float(spike_rate.mean()),
        "wr_p50": wr_p50,
        "wr_p95": wr_p95,
        "total_frames": total_frames,
        "total_secs": total_secs,
    }


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EMG5Bit(in_channels=NUM_EMG, n_out=5).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = np.asarray(ckpt["input_mean"], dtype=np.float32)[:NUM_EMG]
    std = np.asarray(ckpt["input_std"], dtype=np.float32)[:NUM_EMG]
    window = int(ckpt["window"])
    return model, mean, std, window


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sessions = (
        "recordings/bits_20260420_112939/emg.csv,"
        "recordings/bits_20260420_113810/emg.csv,"
        "recordings/bits_20260422_214501/emg.csv"
    )

    print("Loading sessions (all 3)...")
    segs = load_sessions(sessions, gap_s=0.1)
    _, val_segs = split_segments(segs, val_frac=0.2, seed=0)
    val_frames_per_seg = [s.emg.shape[0] for s in val_segs]
    val_seconds = sum(val_frames_per_seg) / SAMPLE_RATE
    print(f"  {len(segs)} total segs | val: {len(val_segs)} segs ({val_seconds:.1f} s)\n")

    models_to_eval = [
        ("baseline_noconsistency", "models/emg5bit_selflabel_best.pt"),
        ("old_smooth",             "models/old_smooth.pt"),
        ("lite_smooth",            "models/lite_smooth.pt"),
        ("combo_smooth",           "models/combo_smooth.pt"),
    ]

    # Smoothing configs for each model
    smooth_configs = [
        (1, 1.0,  "raw"),
        (1, 0.35, "ema α=0.35"),
        (5, 1.0,  "median K=5"),
        (5, 0.35, "median K=5 + ema α=0.35"),
    ]

    # === Evaluate ===
    all_results = {}
    for mname, mpath in models_to_eval:
        abs_path = mpath if os.path.isabs(mpath) else os.path.join(root, mpath)
        model, mean, std, win = load_ckpt(abs_path, device)
        print(f"{mname:<14} window={win} samples ({win*2} ms)")
        all_results[mname] = {"window": win}
        for k, a, cfgn in smooth_configs:
            r = eval_config(model, val_segs, mean, std, win, HOP_SAMPLES, k, a, device)
            all_results[mname][cfgn] = r

    # === Print big table ===
    print("\n" + "=" * 110)
    print("ALL MODELS × ALL SMOOTHING CONFIGS  (same val set: 3 sessions, seed=0, val_frac=0.2)")
    print("=" * 110)
    header = f"{'model':<14}{'smoothing':<28}{'per-bit acc':>12}{'exact':>8}{'flips/s':>10}{'spike%':>9}{'wr_p95':>22}"
    print(header)
    print("-" * 110)
    for mname, _ in models_to_eval:
        for k, a, cfgn in smooth_configs:
            r = all_results[mname][cfgn]
            wr95 = ",".join(str(x) for x in r["wr_p95"])
            print(f"{mname:<14}{cfgn:<28}"
                  f"{r['acc_mean']:>12.3f}"
                  f"{r['exact_match']:>8.3f}"
                  f"{r['flip_sum_per_sec']:>10.2f}"
                  f"{r['spike_mean']*100:>8.2f}%"
                  f"  {wr95:>18}")
        print()
    print("=" * 110)

    # === Poster-ready summary ===
    print("\n" + "#" * 70)
    print("# POSTER-READY NUMBERS")
    print("#" * 70)

    for mname, _ in models_to_eval:
        w = all_results[mname]["window"]
        raw = all_results[mname]["raw"]
        med = all_results[mname]["median K=5"]
        medema = all_results[mname]["median K=5 + ema α=0.35"]
        print(f"\n{mname}  (window={w*2}ms)")
        print(f"  raw:                  acc={raw['acc_mean']:.3f}  "
              f"exact={raw['exact_match']:.3f}  flicker={raw['flip_sum_per_sec']:.2f}/s")
        print(f"  + median K=5:         acc={med['acc_mean']:.3f}  "
              f"exact={med['exact_match']:.3f}  flicker={med['flip_sum_per_sec']:.2f}/s")
        print(f"  + median+EMA:         acc={medema['acc_mean']:.3f}  "
              f"exact={medema['exact_match']:.3f}  flicker={medema['flip_sum_per_sec']:.2f}/s")

    print("\nPer-finger accuracy (raw, no smoothing):")
    print(f"  {'model':<14}" + "".join(f"{L:>9}" for L in LABELS) + f"{'MEAN':>9}")
    for mname, _ in models_to_eval:
        raw = all_results[mname]["raw"]
        print(f"  {mname:<14}" +
              "".join(f"{v:>9.3f}" for v in raw["acc_per_bit"]) +
              f"{raw['acc_mean']:>9.3f}")


if __name__ == "__main__":
    main()
