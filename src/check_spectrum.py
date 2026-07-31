"""Spectrum + filter audit.

For each session (or one, via --session), compute the per-channel power
spectrum of the RAW EMG and compare it to the FILTERED EMG produced by
filter_emg (the function in src/convert_to_hdf5.py that runs before
training). Answers three questions:

  1. Is there a 50 Hz or 60 Hz power-line peak? (Determines whether the
     hardcoded 60 Hz notch is even in the right place.)
  2. How much power lives BELOW 20 Hz? That's the envelope-band content
     the bandpass throws away.
  3. Does the filter actually null what it's supposed to?

Also reports DC offset and slow-drift amplitude per channel — these are
the things that would dominate z-score if you trained on raw.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.convert_to_hdf5 import filter_emg

NUM_EMG = 8
FS = 500.0


def session_spectrum(csv_path, nperseg=2048):
    """Welch-like averaged periodogram across non-overlapping segments.

    Returns (freqs, raw_psd[nch, F], filt_psd[nch, F]) where psd is power per
    bin (same normalization between raw and filtered).
    """
    df = pd.read_csv(csv_path)
    cols = [f"emg_{i}" for i in range(NUM_EMG)]
    raw = df[cols].values.astype(np.float32)   # (T, 8)
    T = raw.shape[0]
    # Truncate to integer multiple of nperseg
    nseg = T // nperseg
    if nseg < 2:
        return None
    raw = raw[: nseg * nperseg]
    filt = filter_emg(raw.copy(), fs=FS)

    def psd(sig):
        # sig: (T, C). Segment along time.
        segs = sig.reshape(nseg, nperseg, NUM_EMG)          # (S, N, C)
        win = np.hanning(nperseg)[None, :, None]
        segs = segs * win
        F = np.fft.rfft(segs, axis=1)                        # (S, N/2+1, C)
        P = (F.real ** 2 + F.imag ** 2).mean(axis=0)         # (N/2+1, C)
        return P.T                                           # (C, F)

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / FS)
    return freqs, psd(raw), psd(filt)


def describe_peak(freqs, P, f_lo, f_hi):
    """Peak power and freq within [f_lo, f_hi]."""
    m = (freqs >= f_lo) & (freqs <= f_hi)
    idx = np.argmax(P[m])
    ff = freqs[m][idx]
    pp = P[m][idx]
    return ff, pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--session", default=None, help="Single session dir name; else all.")
    ap.add_argument("--top", type=int, default=1, help="Print top-N sessions by 50Hz/60Hz ratio.")
    args = ap.parse_args()

    rec = Path(args.recordings)
    if args.session:
        sessions = [rec / args.session]
    else:
        sessions = sorted(d for d in rec.iterdir() if d.is_dir() and (d / "emg.csv").exists())

    rows = []
    for sdir in sessions:
        res = session_spectrum(sdir / "emg.csv")
        if res is None:
            print(f"{sdir.name}: too short, skipping")
            continue
        freqs, raw_psd, filt_psd = res

        # Aggregate across channels
        raw_mean = raw_psd.mean(axis=0)   # (F,)
        filt_mean = filt_psd.mean(axis=0)

        # Power-line check: 50 vs 60
        p50, v50 = describe_peak(freqs, raw_mean, 48, 52)
        p60, v60 = describe_peak(freqs, raw_mean, 58, 62)
        # Local baseline away from both lines
        base_mask = ((freqs > 70) & (freqs < 120))
        base = raw_mean[base_mask].mean() + 1e-12
        r50 = v50 / base
        r60 = v60 / base

        # Band energy fractions
        total = raw_mean.sum() + 1e-12
        e_sub20 = raw_mean[freqs < 20].sum() / total
        e_20_240 = raw_mean[(freqs >= 20) & (freqs <= 240)].sum() / total
        e_over240 = raw_mean[freqs > 240].sum() / total

        # What the filter actually removes (ratio filt/raw in each band)
        def band_frac(P, lo, hi):
            return P[(freqs >= lo) & (freqs <= hi)].sum() + 1e-12
        kept_sub20 = band_frac(filt_mean, 0, 20) / band_frac(raw_mean, 0, 20)
        kept_50 = band_frac(filt_mean, 48, 52) / band_frac(raw_mean, 48, 52)
        kept_60 = band_frac(filt_mean, 58, 62) / band_frac(raw_mean, 58, 62)
        kept_band = band_frac(filt_mean, 20, 240) / band_frac(raw_mean, 20, 240)

        # Raw-signal DC and drift (per channel, before filtering)
        df = pd.read_csv(sdir / "emg.csv")
        raw_sig = df[[f"emg_{i}" for i in range(NUM_EMG)]].values.astype(np.float32)
        dc = np.abs(raw_sig.mean(axis=0)).mean()
        # 1-second (500-sample) moving-average std as a proxy for drift amplitude
        if len(raw_sig) > 500:
            kern = 500
            cs = np.cumsum(raw_sig, axis=0)
            ma = (cs[kern:] - cs[:-kern]) / kern
            drift_std = ma.std(axis=0).mean()
        else:
            drift_std = float("nan")
        hf_std = filter_emg(raw_sig.copy(), fs=FS).std(axis=0).mean()

        print(f"\n=== {sdir.name} ({len(df)} samples, {len(df)/FS:.1f}s) ===")
        print(f"  Power-line peaks (vs 70-120Hz baseline):")
        print(f"    50Hz band: peak at {p50:.1f}Hz, {r50:.1f}x baseline")
        print(f"    60Hz band: peak at {p60:.1f}Hz, {r60:.1f}x baseline")
        print(f"    -> mains is probably {'50Hz' if r50 > r60 else '60Hz'} "
              f"({'MATCHES notch' if r60 > r50 else 'MISMATCH — notch is at 60 but peak is at 50'})")
        print(f"  Band energy fractions (RAW):  <20Hz={e_sub20:.2%}  20-240Hz={e_20_240:.2%}  >240Hz={e_over240:.2%}")
        print(f"  Post-filter keep ratios:     <20Hz={kept_sub20:.3f}  50Hz={kept_50:.3f}  60Hz={kept_60:.3f}  20-240={kept_band:.3f}")
        print(f"  Raw DC offset (|mean|, avg across ch): {dc:.3f}")
        print(f"  Raw 1s drift std (avg across ch):      {drift_std:.3f}")
        print(f"  Post-filter signal std (avg across ch): {hf_std:.3f}")
        print(f"  -> DC/drift is {dc/max(hf_std,1e-9):.1f}x / {drift_std/max(hf_std,1e-9):.1f}x the filtered signal")

        rows.append(dict(name=sdir.name, r50=r50, r60=r60,
                         kept_50=kept_50, kept_60=kept_60,
                         dc=dc, drift=drift_std, sig=hf_std))

    # Summary
    if len(rows) > 1:
        print("\n=== Summary across sessions ===")
        r50s = np.array([r["r50"] for r in rows])
        r60s = np.array([r["r60"] for r in rows])
        print(f"  50Hz/baseline ratio: mean={r50s.mean():.1f}  median={np.median(r50s):.1f}  max={r50s.max():.1f}")
        print(f"  60Hz/baseline ratio: mean={r60s.mean():.1f}  median={np.median(r60s):.1f}  max={r60s.max():.1f}")
        if r50s.mean() > r60s.mean():
            print("  CONCLUSION: 50Hz peak dominates on average. The hardcoded 60Hz notch is wrong for your mains.")
        else:
            print("  CONCLUSION: 60Hz peak dominates. Current notch is correct.")


if __name__ == "__main__":
    main()
