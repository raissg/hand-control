"""Landmark cleaning — remove MediaPipe detection errors from landmarks.csv.

Used by convert_to_hdf5.py before alignment + interpolation. Separated into
its own module so the cleaning rules are inspectable and testable without
running the whole HDF5 build.

What gets removed (and why):

  A. Minority-handedness rows (+ their ±1 frame neighbors).
     Each session has a dominant handedness (e.g., 99% "Left"). The short
     "Right" runs are detection flips where MediaPipe briefly locked onto
     the wrong hand or got confused by a transitional pose. Verified
     empirically: median coord-delta at flip frames is 4×–16× larger than
     at non-flip frames. Neighbors are included because video mode applies
     temporal smoothing, which pollutes the 1 frame before/after a flip.

  B. Hallucinated-coord rows.
     After wrist-centering + scale-normalization, every landmark of a real
     hand lies within ~3 normalized units of origin. We drop rows where
     ANY coord exceeds ±5 — anatomically impossible, always a detection
     failure (finger out of frame, MediaPipe completing phantom fingers).

What is deliberately NOT removed:

  - Jumps without flips. A large frame-to-frame delta can be legitimate
    fast motion. Only removed if also a flip or out-of-bbox (both rules
    already catch the real corruptions).
  - Scale variation. Hand distance to camera naturally varies; don't drop
    on that alone.
  - NaN/Inf. We never see these in practice; if they appear, dropna in
    convert_to_hdf5 handles them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


LANDMARK_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
COORD_ABS_MAX = 5.0         # beyond this = hallucinated
FLIP_NEIGHBOR_RADIUS = 1    # expand flip removal by ±N frames


@dataclass
class CleaningStats:
    session: str
    n_in: int
    n_out: int
    removed_flip: int = 0
    removed_bbox: int = 0
    removed_both: int = 0           # overlap — counted once in total
    dominant_handedness: str = ""
    minority_handedness_count: int = 0

    def report(self) -> str:
        kept_pct = 100.0 * self.n_out / max(self.n_in, 1)
        return (
            f"{self.session}: {self.n_in} → {self.n_out} landmark rows ({kept_pct:.1f}% kept)  "
            f"| removed: flip±{FLIP_NEIGHBOR_RADIUS}={self.removed_flip}  "
            f"bbox={self.removed_bbox}  both={self.removed_both}  "
            f"| dominant handedness: {self.dominant_handedness}"
        )


def _flip_mask(handedness: np.ndarray) -> np.ndarray:
    """True at rows whose handedness != session majority, expanded by ±N neighbors.

    Uses majority (not per-frame flip detection) so that a long "Right" run
    isn't misinterpreted as "the one frame where Right→Left is the flip" —
    the whole run is minority and gets removed.
    """
    if len(handedness) == 0:
        return np.zeros(0, dtype=bool)
    # Majority label
    vals, counts = np.unique(handedness, return_counts=True)
    majority = vals[np.argmax(counts)]
    is_minority = handedness != majority

    # Expand by ±FLIP_NEIGHBOR_RADIUS
    mask = is_minority.copy()
    for d in range(1, FLIP_NEIGHBOR_RADIUS + 1):
        mask[d:] |= is_minority[:-d]
        mask[:-d] |= is_minority[d:]
    return mask


def _bbox_mask(coords: np.ndarray) -> np.ndarray:
    """True at rows where any normalized coord exceeds COORD_ABS_MAX."""
    # coords: (N, 63)
    return (np.abs(coords) > COORD_ABS_MAX).any(axis=1)


def clean_landmarks(df_lm: pd.DataFrame, session_name: str = "") -> tuple[pd.DataFrame, CleaningStats]:
    """Return a cleaned copy of the landmarks DataFrame + stats.

    Expects the schema written by recorder.postprocess_landmarks:
        frame_index, timestamp, x0,y0,z0,...,x20,y20,z20, handedness, scale
    Missing handedness / missing coord columns are tolerated (nothing is
    removed on those dimensions if the column isn't present).
    """
    n_in = len(df_lm)
    stats = CleaningStats(session=session_name or "(unknown)", n_in=n_in, n_out=n_in)

    if n_in == 0:
        return df_lm.copy(), stats

    # Handedness-based mask
    if "handedness" in df_lm.columns:
        handed = df_lm["handedness"].values
        vals, counts = np.unique(handed, return_counts=True)
        stats.dominant_handedness = str(vals[np.argmax(counts)])
        stats.minority_handedness_count = int(counts.sum() - counts.max())
        flip = _flip_mask(handed)
    else:
        flip = np.zeros(n_in, dtype=bool)

    # Bbox mask
    present_cols = [c for c in LANDMARK_COLS if c in df_lm.columns]
    if len(present_cols) == 63:
        coords = df_lm[present_cols].values.astype(np.float64)
        bbox = _bbox_mask(coords)
    else:
        bbox = np.zeros(n_in, dtype=bool)

    both = flip & bbox
    bad = flip | bbox

    stats.removed_flip = int((flip & ~bbox).sum())
    stats.removed_bbox = int((bbox & ~flip).sum())
    stats.removed_both = int(both.sum())

    cleaned = df_lm.loc[~bad].copy().reset_index(drop=True)
    stats.n_out = len(cleaned)
    return cleaned, stats


# ---------------------------------------------------------------------------
# CLI: dry-run the cleaning on every session and report what WOULD be removed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Dry-run landmark cleaning (no files modified).")
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--session", default=None)
    args = ap.parse_args()

    rec = Path(args.recordings)
    if args.session:
        sessions = [rec / args.session]
    else:
        sessions = sorted(d for d in rec.iterdir() if d.is_dir())

    total_in = total_out = 0
    for sdir in sessions:
        p = sdir / "landmarks.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        _, stats = clean_landmarks(df, session_name=sdir.name)
        print(stats.report())
        total_in += stats.n_in
        total_out += stats.n_out
    if total_in:
        pct = 100.0 * total_out / total_in
        print(f"\nTOTAL: {total_in} → {total_out} rows ({pct:.2f}% kept, "
              f"{total_in - total_out} removed)")
