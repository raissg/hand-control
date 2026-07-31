"""Dry-run the cleaning pipeline for specific sessions and print the
time intervals that would be dropped.

Runs exactly the logic that convert_to_hdf5.py will run at build time
(landmark cleaning + gap-based valid mask + sub-session splitting)
without writing any files. Purpose: let you inspect EXACTLY what time
ranges of each session are being removed before rebuilding the dataset.

Usage:
    PYTHONPATH=. python src/report_dropped.py \
        recordings/20260415_212910 recordings/20260416_123151 ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean_landmarks import clean_landmarks
from src.convert_to_hdf5 import IMU_COLS, MAX_INTERP_GAP_S, MIN_SUBSESSION_SAMPLES


def analyze_session(session_dir: Path, num_emg: int = 8):
    emg_path = session_dir / "emg.csv"
    lm_path = session_dir / "landmarks.csv"
    if not emg_path.exists() or not lm_path.exists():
        print(f"\n{session_dir.name}: missing emg.csv or landmarks.csv — skipping")
        return None

    df_emg = pd.read_csv(emg_path)
    df_lm_raw = pd.read_csv(lm_path)

    emg_cols = [f"emg_{i}" for i in range(num_emg)]
    # Same header-whitespace defense as convert_to_hdf5 would benefit from
    df_emg.columns = [c.strip() for c in df_emg.columns]
    df_lm_raw.columns = [c.strip() for c in df_lm_raw.columns]

    input_cols = [c for c in emg_cols + IMU_COLS if c in df_emg.columns]
    df_emg = df_emg.dropna(subset=["timestamp"] + input_cols)
    df_lm_raw = df_lm_raw.dropna(subset=["timestamp"])

    # ------------------------------------------------------------------
    # Step 1: clean landmarks (flips + bbox)
    # ------------------------------------------------------------------
    df_lm_clean, stats = clean_landmarks(df_lm_raw, session_name=session_dir.name)
    removed_mask = ~df_lm_raw["timestamp"].isin(df_lm_clean["timestamp"]).values
    removed_times = df_lm_raw.loc[removed_mask, "timestamp"].values
    removed_intervals = _group_as_intervals(removed_times, merge_tol=0.1)

    # ------------------------------------------------------------------
    # Step 2: EMG valid mask (gap > MAX_INTERP_GAP_S = invalid)
    # ------------------------------------------------------------------
    t_emg = df_emg["timestamp"].values
    t_lm = df_lm_clean["timestamp"].values
    t_start = max(t_emg.min(), t_lm.min())
    t_end = min(t_emg.max(), t_lm.max())
    mask_in_window = (t_emg >= t_start) & (t_emg <= t_end)
    t_emg_sync = t_emg[mask_in_window]

    idx = np.searchsorted(t_lm, t_emg_sync)
    idx_c = np.clip(idx, 1, len(t_lm) - 1)
    gap = t_lm[idx_c] - t_lm[idx_c - 1]
    valid = (
        (gap <= MAX_INTERP_GAP_S)
        & (t_emg_sync >= t_lm[0])
        & (t_emg_sync <= t_lm[-1])
    )

    # Time intervals where valid == False
    invalid_intervals = _runs_to_intervals(t_emg_sync, ~valid)

    # ------------------------------------------------------------------
    # Step 3: sub-session splitting (contiguous True runs in `valid`)
    # ------------------------------------------------------------------
    v = np.concatenate(([0], valid.astype(np.int8), [0]))
    starts = np.where(np.diff(v) == 1)[0]
    ends = np.where(np.diff(v) == -1)[0]
    kept_subs = []
    dropped_short_subs = []
    for s, e in zip(starts, ends):
        n = e - s
        duration = (t_emg_sync[e - 1] - t_emg_sync[s]) if n > 0 else 0
        t_lo, t_hi = t_emg_sync[s], t_emg_sync[e - 1]
        if n < MIN_SUBSESSION_SAMPLES:
            dropped_short_subs.append((t_lo, t_hi, n, duration))
        else:
            kept_subs.append((t_lo, t_hi, n, duration))

    total_duration = t_emg.max() - t_emg.min() if len(t_emg) else 0.0
    emg_dropped_as_invalid = int((~valid).sum()) + int((~mask_in_window).sum())
    kept_samples = sum(n for _, _, n, _ in kept_subs)
    kept_duration = sum(d for _, _, _, d in kept_subs)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"{session_dir.name}  ({len(t_emg)} EMG samples, {total_duration:.1f}s total)")
    print("=" * 70)
    print(f"Landmark cleaning: {stats.report()}")

    if len(removed_intervals) > 0:
        print(f"\n  Removed landmark rows group into {len(removed_intervals)} time interval(s):")
        for t_lo, t_hi, n in removed_intervals[:50]:
            dur = t_hi - t_lo
            dur_str = f"{dur*1000:.0f}ms" if dur < 1 else f"{dur:.2f}s"
            print(f"    t=[{t_lo:7.2f}s .. {t_hi:7.2f}s]  ({n} rows, {dur_str})")
        if len(removed_intervals) > 50:
            print(f"    ... and {len(removed_intervals) - 50} more")

    if len(invalid_intervals) > 0:
        print(f"\n  EMG invalid intervals (gap > {MAX_INTERP_GAP_S}s or outside cleaned-landmark range):")
        for t_lo, t_hi, n in invalid_intervals:
            dur = t_hi - t_lo
            dur_str = f"{dur*1000:.0f}ms" if dur < 1 else f"{dur:.2f}s"
            print(f"    t=[{t_lo:7.2f}s .. {t_hi:7.2f}s]  ({n} EMG samples, {dur_str})")
    else:
        print(f"\n  EMG invalid intervals: none")

    print(f"\n  Sub-sessions KEPT ({len(kept_subs)}, >= {MIN_SUBSESSION_SAMPLES} samples each):")
    for t_lo, t_hi, n, dur in kept_subs:
        print(f"    t=[{t_lo:7.2f}s .. {t_hi:7.2f}s]  ({n} samples, {dur:.2f}s)")

    if dropped_short_subs:
        print(f"\n  Sub-sessions DROPPED (shorter than {MIN_SUBSESSION_SAMPLES} samples):")
        for t_lo, t_hi, n, dur in dropped_short_subs:
            dur_str = f"{dur*1000:.0f}ms" if dur < 1 else f"{dur:.2f}s"
            print(f"    t=[{t_lo:7.2f}s .. {t_hi:7.2f}s]  ({n} samples, {dur_str})")

    print(f"\n  Summary:  {kept_samples}/{len(t_emg)} EMG samples kept "
          f"({100*kept_samples/max(len(t_emg),1):.1f}%)  |  "
          f"{kept_duration:.1f}s / {total_duration:.1f}s training time retained")

    return dict(
        name=session_dir.name,
        total_samples=len(t_emg),
        kept_samples=kept_samples,
        total_duration=total_duration,
        kept_duration=kept_duration,
        n_kept_subs=len(kept_subs),
        n_dropped_subs=len(dropped_short_subs),
    )


def _group_as_intervals(times: np.ndarray, merge_tol: float = 0.1):
    """Group consecutive timestamps into [t_lo, t_hi, count] intervals.
    Adjacent points within merge_tol seconds fuse into one interval.
    """
    if len(times) == 0:
        return []
    times = np.sort(times)
    intervals = []
    lo = hi = times[0]
    n = 1
    for t in times[1:]:
        if t - hi <= merge_tol:
            hi = t
            n += 1
        else:
            intervals.append((float(lo), float(hi), n))
            lo = hi = t
            n = 1
    intervals.append((float(lo), float(hi), n))
    return intervals


def _runs_to_intervals(times: np.ndarray, mask: np.ndarray):
    """Find time-intervals of contiguous True runs in `mask`.
    Returns list of (t_lo, t_hi, count)."""
    if len(mask) == 0 or not mask.any():
        return []
    m = np.concatenate(([False], mask, [False]))
    starts = np.where(np.diff(m.astype(np.int8)) == 1)[0]
    ends = np.where(np.diff(m.astype(np.int8)) == -1)[0]
    out = []
    for s, e in zip(starts, ends):
        out.append((float(times[s]), float(times[e - 1]), int(e - s)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="Paths to session directories")
    args = ap.parse_args()

    summaries = []
    for p in args.sessions:
        sdir = Path(p).resolve()
        if not sdir.is_dir():
            print(f"Not a directory: {sdir}")
            continue
        res = analyze_session(sdir)
        if res:
            summaries.append(res)

    if len(summaries) > 1:
        print("\n" + "=" * 70)
        print("OVERALL")
        print("=" * 70)
        tot = sum(s["total_samples"] for s in summaries)
        kept = sum(s["kept_samples"] for s in summaries)
        tot_d = sum(s["total_duration"] for s in summaries)
        kept_d = sum(s["kept_duration"] for s in summaries)
        print(f"  Sessions: {len(summaries)}")
        print(f"  EMG samples: {kept}/{tot} kept ({100*kept/max(tot,1):.1f}%)")
        print(f"  Training duration: {kept_d:.1f}s / {tot_d:.1f}s  ({kept_d/60:.1f} / {tot_d/60:.1f} min)")


if __name__ == "__main__":
    main()
