"""Sanity-check ground truth landmarks across all sessions.

Because MediaPipe is configured with max_hands=1 and only writes a row when
a hand is detected, a landmarks.csv can silently hide three kinds of
problems:

  1. Coverage: the hand was OUT OF FRAME for N frames — no row written,
     just a gap in frame_index. EMG keeps streaming during that time, so
     every window that overlaps the gap is paired with extrapolated/older
     landmarks. Bad training signal.

  2. Wrong hand detected: the user's other hand briefly entered the
     camera view. MediaPipe returned it as the "first hand" and wrote
     normalized coords for THAT hand. Signatures:
       - handedness flips between "Left" and "Right" in the same session
       - landmark position jumps by a huge L2 between adjacent frames
       - scale (wrist-to-middle-MCP distance in raw frame coords) jumps
         because the hands are at different distances from camera.

  3. Bad detection (occlusion, finger out of frame, hand half-cropped):
     MediaPipe still returns 21 landmarks but some are pure hallucination.
     Signatures:
       - a landmark coordinate goes far outside reasonable range
         (wrist-to-tip distance usually <3 in normalized units)
       - a single-frame spike followed by return to normal pose

We flag all three here. Output is per-session plus an aggregate.

Usage:
    PYTHONPATH=. python src/check_landmarks.py
    PYTHONPATH=. python src/check_landmarks.py --session 20260416_123151
    PYTHONPATH=. python src/check_landmarks.py --verbose   # prints every gap/spike
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LANDMARK_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
NUM_LANDMARKS = 21


def load_session(sdir: Path):
    """Return (frame_times_df, landmarks_df_or_None)."""
    ft_path = sdir / "frame_times.csv"
    lm_path = sdir / "landmarks.csv"
    if not ft_path.exists():
        return None, None
    ft = pd.read_csv(ft_path)
    lm = pd.read_csv(lm_path) if lm_path.exists() else None
    return ft, lm


def check_coverage(ft, lm):
    """Fraction of video frames that have a landmark row."""
    total_frames = len(ft)
    detected = len(lm) if lm is not None else 0
    coverage = detected / max(total_frames, 1)

    # Find gaps — consecutive video frames with no detection
    detected_set = set(lm["frame_index"].astype(int).tolist()) if lm is not None else set()
    all_frames = ft["frame_index"].astype(int).tolist()
    missing = [f for f in all_frames if f not in detected_set]

    # Group missing into contiguous runs
    gaps = []
    if missing:
        run_start = missing[0]
        prev = missing[0]
        for f in missing[1:]:
            if f == prev + 1:
                prev = f
            else:
                gaps.append((run_start, prev, prev - run_start + 1))
                run_start = f
                prev = f
        gaps.append((run_start, prev, prev - run_start + 1))
    return total_frames, detected, coverage, gaps


def check_handedness(lm):
    """Counts of Left vs Right and flip count."""
    if lm is None or len(lm) == 0:
        return {}, 0
    counts = lm["handedness"].value_counts().to_dict()
    handed = lm["handedness"].values
    flips = int((handed[1:] != handed[:-1]).sum())
    return counts, flips


def check_nans(lm):
    """Any NaN/Inf in landmarks?"""
    if lm is None or len(lm) == 0:
        return 0
    arr = lm[LANDMARK_COLS].values.astype(np.float64)
    return int(np.sum(~np.isfinite(arr)))


def check_scale(lm):
    """Stats on the raw-pixel scale (wrist-to-middle-MCP distance).

    Small + jumpy ⇒ different hand or wrong hand entered frame.
    """
    if lm is None or len(lm) == 0:
        return None
    s = lm["scale"].values.astype(np.float64)
    s_valid = s[np.isfinite(s) & (s > 0)]
    if len(s_valid) == 0:
        return None
    return dict(
        mean=float(s_valid.mean()),
        std=float(s_valid.std()),
        cv=float(s_valid.std() / (s_valid.mean() + 1e-9)),
        min=float(s_valid.min()),
        max=float(s_valid.max()),
        # Rows where scale is < half the median — likely tracking a smaller/farther hand
        tiny_frac=float((s_valid < 0.5 * np.median(s_valid)).mean()),
    )


def check_jumps(lm, z_thresh=5.0):
    """Find frames where the per-frame landmark delta is an outlier.

    We compute L2 distance between consecutive frames' flattened 63-vectors
    (normalized landmarks). Most deltas are small (continuous motion);
    huge deltas mean MediaPipe swapped to a different hand or the detection
    is wildly wrong.

    Returns (num_outliers, list_of_(frame_idx, delta, z)).
    """
    if lm is None or len(lm) < 2:
        return 0, []
    X = lm[LANDMARK_COLS].values.astype(np.float64)       # (N, 63)
    dX = np.linalg.norm(X[1:] - X[:-1], axis=1)           # (N-1,)
    # Robust z: median and MAD
    med = np.median(dX)
    mad = np.median(np.abs(dX - med)) + 1e-9
    z = (dX - med) / (1.4826 * mad)
    frame_idx = lm["frame_index"].values[1:]
    outliers = []
    for i in np.where(z > z_thresh)[0]:
        outliers.append((int(frame_idx[i]), float(dX[i]), float(z[i])))
    return len(outliers), outliers


def check_flips_vs_jumps(lm):
    """Are handedness flips correlated with landmark coordinate jumps?

    If yes: flips mark genuinely bad detections (model would see junk).
    If no: flips are cosmetic labeling noise in MediaPipe, landmarks still continuous.

    We compare per-frame L2 delta at flip vs non-flip frames.
    """
    if lm is None or len(lm) < 3:
        return None
    handed = lm["handedness"].values
    X = lm[LANDMARK_COLS].values.astype(np.float64)
    dX = np.linalg.norm(X[1:] - X[:-1], axis=1)
    flip_mask = (handed[1:] != handed[:-1])
    if flip_mask.sum() == 0:
        return None
    return dict(
        median_delta_on_flip=float(np.median(dX[flip_mask])),
        median_delta_no_flip=float(np.median(dX[~flip_mask])),
        ratio=float(np.median(dX[flip_mask]) / (np.median(dX[~flip_mask]) + 1e-9)),
        n_flips=int(flip_mask.sum()),
    )


def check_bbox(lm):
    """Per-axis min/max of landmarks. After wrist-centering and scaling,
    every finger tip should lie within a few units of origin.

    Flags rows where any coord is beyond ABS_MAX.
    """
    ABS_MAX = 5.0  # normalized units — a real hand fits within ~3
    if lm is None or len(lm) == 0:
        return None
    arr = lm[LANDMARK_COLS].values.astype(np.float64).reshape(-1, 21, 3)
    abs_max_per_row = np.abs(arr).reshape(len(arr), -1).max(axis=1)
    bad = int((abs_max_per_row > ABS_MAX).sum())
    return dict(
        overall_min=float(arr.min()),
        overall_max=float(arr.max()),
        bad_rows=bad,
        bad_frac=float(bad / len(arr)),
    )


def report_session(sdir: Path, verbose=False):
    ft, lm = load_session(sdir)
    if ft is None:
        print(f"\n=== {sdir.name} — no frame_times.csv, skipping ===")
        return None
    if lm is None:
        print(f"\n=== {sdir.name} — no landmarks.csv yet ({len(ft)} video frames) ===")
        return None

    total, detected, cov, gaps = check_coverage(ft, lm)
    hcounts, flips = check_handedness(lm)
    nans = check_nans(lm)
    sc = check_scale(lm)
    n_jumps, jumps = check_jumps(lm)
    bbox = check_bbox(lm)
    fvj = check_flips_vs_jumps(lm)

    print(f"\n=== {sdir.name} ===")
    print(f"  Coverage: {detected}/{total} video frames ({cov*100:.1f}%)")
    if gaps:
        longest = max(gaps, key=lambda g: g[2])
        total_missing = sum(g[2] for g in gaps)
        print(f"  Missing: {total_missing} frames across {len(gaps)} gaps  "
              f"(longest: frames {longest[0]}–{longest[1]}, {longest[2]} frames = {longest[2]/30:.1f}s)")
        if verbose:
            for s, e, n in sorted(gaps, key=lambda g: -g[2])[:10]:
                print(f"      gap frames {s}–{e}  ({n} frames, {n/30:.1f}s)")

    print(f"  Handedness: {hcounts}   flips={flips}")
    if flips > 0:
        # Find the flip frame indices
        handed = lm["handedness"].values
        flip_at = np.where(handed[1:] != handed[:-1])[0]
        print(f"    FLIP WARNING — {flips} handedness switch(es). "
              f"MediaPipe saw the other hand at frames: "
              f"{lm['frame_index'].iloc[flip_at[:5]+1].tolist()}"
              f"{'...' if flips > 5 else ''}")
        if fvj is not None:
            print(f"    Median landmark delta at flip frames: {fvj['median_delta_on_flip']:.4f}  "
                  f"vs non-flip: {fvj['median_delta_no_flip']:.4f}  "
                  f"(ratio {fvj['ratio']:.1f}x)")
            if fvj["ratio"] < 3.0:
                print(f"      → flips are COSMETIC (coords continuous). Safe to ignore.")
            else:
                print(f"      → flips CORRUPT landmarks (coord jump {fvj['ratio']:.1f}× normal). Bad frames.")

    print(f"  NaN/Inf coords: {nans}")

    if sc:
        print(f"  Raw scale (wrist→midMCP px): mean={sc['mean']:.3f}  std={sc['std']:.3f}  "
              f"CV={sc['cv']:.2f}  min={sc['min']:.3f}  max={sc['max']:.3f}")
        if sc["tiny_frac"] > 0.05:
            print(f"    WARNING: {sc['tiny_frac']*100:.1f}% of frames have scale < 50% of median "
                  f"(likely other/farther hand)")
        if sc["cv"] > 0.3:
            print(f"    WARNING: scale CV={sc['cv']:.2f} is high (>0.3). Distance/hand identity is unstable.")

    if bbox:
        print(f"  Landmark bbox: min={bbox['overall_min']:+.2f}  max={bbox['overall_max']:+.2f}  "
              f"out-of-bounds rows={bbox['bad_rows']} ({bbox['bad_frac']*100:.2f}%)")
        if bbox["bad_frac"] > 0.01:
            print(f"    WARNING: >1% of frames have a landmark beyond ±5.0 — clear hallucinations.")

    print(f"  Frame-to-frame jumps beyond 5σ (MAD): {n_jumps}")
    if n_jumps > 0 and verbose:
        for f, d, z in sorted(jumps, key=lambda x: -x[2])[:10]:
            print(f"      frame {f}: delta={d:.3f}  z={z:.1f}")

    # Composite health flag
    bad = False
    reasons = []
    if cov < 0.85:
        bad = True; reasons.append(f"coverage {cov*100:.0f}%")
    if flips > 2:
        bad = True; reasons.append(f"{flips} hand flips")
    if sc and sc["cv"] > 0.3:
        bad = True; reasons.append(f"unstable scale (CV={sc['cv']:.2f})")
    if bbox and bbox["bad_frac"] > 0.01:
        bad = True; reasons.append(f"{bbox['bad_frac']*100:.1f}% out-of-bbox")
    if nans > 0:
        bad = True; reasons.append(f"{nans} NaN coords")
    if bad:
        print(f"  → SESSION FLAGGED: {', '.join(reasons)}")
    else:
        print(f"  → OK")

    return dict(
        name=sdir.name, total=total, detected=detected, coverage=cov,
        gaps=len(gaps), missing=sum(g[2] for g in gaps) if gaps else 0,
        flips=flips, nans=nans, bad=bad, reasons=reasons,
        scale_cv=(sc["cv"] if sc else None),
        bbox_bad=(bbox["bad_frac"] if bbox else None),
        jumps=n_jumps,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--session", default=None, help="One session name, else all.")
    ap.add_argument("--verbose", action="store_true", help="Print individual gap / jump frames.")
    args = ap.parse_args()

    rec = Path(args.recordings)
    if args.session:
        sessions = [rec / args.session]
    else:
        sessions = sorted(d for d in rec.iterdir() if d.is_dir())

    results = []
    for sdir in sessions:
        r = report_session(sdir, verbose=args.verbose)
        if r is not None:
            results.append(r)

    # Aggregate
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("AGGREGATE")
        print("=" * 70)
        flagged = [r for r in results if r["bad"]]
        print(f"Sessions checked: {len(results)}")
        print(f"Sessions flagged: {len(flagged)}")
        if flagged:
            print("\nFlagged sessions (likely harming training):")
            for r in flagged:
                print(f"  {r['name']}: {', '.join(r['reasons'])}")
        cov_mean = np.mean([r["coverage"] for r in results])
        total_missing = sum(r["missing"] for r in results)
        total_frames = sum(r["total"] for r in results)
        total_flips = sum(r["flips"] for r in results)
        print(f"\nOverall coverage: {cov_mean*100:.1f}%  "
              f"({total_missing} missing / {total_frames} total video frames)")
        print(f"Total handedness flips across all sessions: {total_flips}")


if __name__ == "__main__":
    main()
