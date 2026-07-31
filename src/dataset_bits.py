"""Dataset for self-labeled 5-bit recordings from ``src.record_bits``.

Input layout (produced by record_bits.py):
    recordings/bits_YYYYMMDD_HHMMSS/emg.csv
    Columns: timestamp, ch0..ch7, b0..b4

For each session CSV:
    1. Split rows into **segments** — contiguous stretches of samples that
       share the same 5-bit label AND have no time gap (> --gap_s) between
       consecutive timestamps (gaps occur between HOLD phases since REST is
       not recorded).
    2. Bandpass + notch filter the EMG per segment (filtfilt needs a
       contiguous signal; filtering across a gap would smear).
    3. Slide ``window`` samples with stride ``step`` inside each segment.
       Label for a window is the segment's bit string.

Usage (from code):
    from src.dataset_bits import load_sessions, BitsWindowDataset
    segs = load_sessions("recordings/bits_*/emg.csv", gap_s=0.1)
    train, val = split_segments(segs, val_frac=0.2, seed=0)
    ds = BitsWindowDataset(train, window=100, step=25,
                            mean=mean, std=std)
"""

from __future__ import annotations

import csv
import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.convert_to_hdf5 import filter_emg


NUM_EMG = 8
NUM_IMU = 6                    # === IMU ADDITION === accel(3) + gyro(3)
SAMPLE_RATE = 500

# === LABEL REMAP =========================================================
# Collapse near-miss gestures onto their canonical neighbors so the model
# only has to predict 7 distinct patterns. Enabled per-call via the
# `remap_labels=True` argument to `load_sessions`. Recorded EMG is
# unchanged; only the saved bit string is rewritten at load time.
# =========================================================================
LABEL_REMAP = {
    "10001": "10000",
    "01110": "11111",
    "11001": "11000",
    "01111": "11111",
    "10111": "11111",
    "11011": "11111",
    "11101": "11111",
}

# The 15 bit-strings recorded by the default ``src.record_bits`` protocol
# (kept here so both training-time and inference-time code derive the
# canonical output set from the same source).
RECORDED_PATTERNS = [
    "00000", "11111", "10000", "01000", "00100",
    "11000", "01100", "10001", "01110", "11001",
    "11100", "01111", "10111", "11011", "11101",
]


def get_supported_patterns(remap: bool = True) -> list[str]:
    """Canonical bit-strings the trained model can output.

    With ``remap=True`` (matches ``train_bits.py --remap``), LABEL_REMAP
    collapses near-miss patterns onto canonical neighbors, yielding 8
    distinct patterns. With ``remap=False``, returns the full 15.
    """
    if remap:
        return sorted({LABEL_REMAP.get(p, p) for p in RECORDED_PATTERNS})
    return list(RECORDED_PATTERNS)


def _remap_label(bits_row: np.ndarray) -> np.ndarray:
    s = "".join(str(int(b)) for b in bits_row)
    s2 = LABEL_REMAP.get(s, s)
    if s2 == s:
        return bits_row
    return np.array([int(c) for c in s2], dtype=bits_row.dtype)


@dataclass
class Segment:
    """Filtered EMG + constant label for one contiguous pose-hold."""
    emg: np.ndarray            # (T, 8) float32, filtered
    # === IMU ADDITION ====================================================
    # Aligned IMU samples for the same time interval. Always present; if the
    # source CSV had no IMU columns (legacy recordings), filled with zeros so
    # downstream multistream code can still index it cleanly.
    # =====================================================================
    imu: np.ndarray            # (T, 6) float32, raw (no EMG-style filter)
    label: np.ndarray          # (5,) float32, 0/1
    source: str                # session dir name (for bookkeeping)


def _read_csv(path: Path):
    """Returns (ts, emg, imu, bits) as numpy arrays.

    === IMU ADDITION ====================================================
    Reads accel_x/y/z, gyro_x/y/z by *column name*. Older sessions that
    only have ``timestamp, ch0..ch7, b0..b4`` are detected from the header
    and filled with zeros for IMU. This keeps backward compatibility.
    =====================================================================
    """
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        # Locate columns by name so we don't depend on positional layout.
        emg_idx = [header.index(f"ch{i}") for i in range(NUM_EMG)]
        bit_idx = [header.index(f"b{i}") for i in range(5)]
        imu_names = ["accel_x", "accel_y", "accel_z",
                     "gyro_x", "gyro_y", "gyro_z"]
        # === IMU ADDITION ===
        has_imu = all(name in header for name in imu_names)
        imu_idx = [header.index(n) for n in imu_names] if has_imu else None
        ts_idx = header.index("timestamp")

        ts, emg, imu, bits = [], [], [], []
        for row in r:
            ts.append(float(row[ts_idx]))
            emg.append([float(row[i]) for i in emg_idx])
            if has_imu:
                imu.append([float(row[i]) for i in imu_idx])
            bits.append([int(row[i]) for i in bit_idx])

    ts_arr = np.asarray(ts, dtype=np.float64)
    emg_arr = np.asarray(emg, dtype=np.float32)
    bits_arr = np.asarray(bits, dtype=np.float32)
    if has_imu and imu:
        imu_arr = np.asarray(imu, dtype=np.float32)
    else:
        # === IMU ADDITION === legacy CSV: fall back to zeros.
        imu_arr = np.zeros((len(ts_arr), NUM_IMU), dtype=np.float32)
    return ts_arr, emg_arr, imu_arr, bits_arr


def _csv_has_imu(path: Path) -> bool:
    """=== IMU ADDITION === peek at the header to see if the CSV has IMU cols."""
    with open(path) as f:
        header = next(csv.reader(f), [])
    return all(c in header for c in ("accel_x", "accel_y", "accel_z",
                                      "gyro_x", "gyro_y", "gyro_z"))


def load_sessions(pattern: str, gap_s: float = 0.1,
                   min_seg_samples: int = 100,
                   require_imu: bool = False,
                   remap_labels: bool = False) -> list[Segment]:
    """Load all matching session CSVs and split into labeled segments.

    gap_s: if two consecutive timestamps differ by more than this, we close
           the current segment (a new hold is starting).
    min_seg_samples: drop segments shorter than this (too short to filter/window).

    require_imu=True skips any CSV that doesn't have accel/gyro columns.
    remap_labels=True applies LABEL_REMAP at load time, collapsing the 14
    recorded patterns onto the 7 canonical ones.
    """
    # Accept comma-separated patterns so callers can pick specific sessions.
    sub_patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    paths: list[Path] = []
    for pat in sub_patterns:
        matches = sorted(Path(".").glob(pat)) if "*" in pat else [Path(pat)]
        paths.extend(matches)
    # De-dupe while preserving order.
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError(f"No CSVs matched: {pattern}")

    # === IMU ADDITION === filter out IMU-less CSVs when the caller demands IMU.
    if require_imu:
        kept, dropped = [], []
        for p in paths:
            (kept if _csv_has_imu(p) else dropped).append(p)
        if dropped:
            print(f"--imu: skipping {len(dropped)} session(s) without IMU columns:")
            for p in dropped:
                print(f"    {p}")
        if not kept:
            raise FileNotFoundError(
                "No sessions with IMU columns matched. Re-record with "
                "`python -m src.record_bits --imu` or drop --imu."
            )
        paths = kept

    segments: list[Segment] = []
    for p in paths:
        ts, emg, imu, bits = _read_csv(p)   # === IMU ADDITION === extra return
        if len(ts) == 0:
            continue
        dt = np.diff(ts, prepend=ts[0])
        label_changed = np.any(np.diff(bits, axis=0, prepend=bits[0:1]) != 0,
                                axis=1)
        gap = dt > gap_s
        # New segment boundary at label change OR time gap
        boundaries = np.where(label_changed | gap)[0]
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(ts)]
        for s, e in zip(starts, ends):
            if e - s < min_seg_samples:
                continue
            raw = emg[s:e]
            try:
                filt = filter_emg(raw.copy(), fs=SAMPLE_RATE)
            except Exception:
                # Segment too short for the designed filter; skip it.
                continue
            # === IMU ADDITION === slice the aligned IMU window. No bandpass
            # filter on IMU — gravity (DC) and slow tilts are the signal we
            # want to keep.
            imu_seg = imu[s:e].astype(np.float32)
            seg_label = bits[s].astype(np.float32)
            if remap_labels:
                seg_label = _remap_label(seg_label.astype(np.int32)).astype(np.float32)
            segments.append(Segment(
                emg=filt.astype(np.float32),
                imu=imu_seg,
                label=seg_label,
                source=str(p.parent.name),
            ))
    if remap_labels:
        print("LABEL_REMAP active:", LABEL_REMAP)
    return segments


def split_segments(segments: list[Segment], val_frac: float = 0.2,
                    seed: int = 0) -> tuple[list[Segment], list[Segment]]:
    """Random segment-level split — no window leakage."""
    rng = random.Random(seed)
    idxs = list(range(len(segments)))
    rng.shuffle(idxs)
    n_val = max(1, int(round(len(idxs) * val_frac)))
    val_ids = set(idxs[:n_val])
    train = [segments[i] for i in idxs[n_val:]]
    val = [segments[i] for i in idxs[:n_val]]
    return train, val


def compute_norm_stats(segments: list[Segment]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over concatenated train segments."""
    data = np.concatenate([s.emg for s in segments], axis=0)   # (T_total, 8)
    mean = data.mean(axis=0).astype(np.float32)
    std = data.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


# === IMU ADDITION =========================================================
# IMU norm stats live in their own pair so the (very different) accel/gyro
# scales don't get squashed by EMG normalization. Accel ~ ±9.8 m/s² with a
# strong DC gravity component; gyro ~ rad/s with mean ≈ 0. Z-scoring per
# channel removes the gravity offset and equalizes magnitudes.
# ==========================================================================
def compute_imu_norm_stats(segments: list[Segment]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std for the IMU stream, computed across all train segs."""
    data = np.concatenate([s.imu for s in segments], axis=0)   # (T_total, 6)
    mean = data.mean(axis=0).astype(np.float32)
    std = data.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


class BitsWindowDataset(Dataset):
    """Sliding-window view over segments, yielding (emg, label) pairs.

    If ``pair_delta_range`` is given (tuple of sample offsets), each item also
    returns a second window at start+Δ (same segment, same label) so that a
    consistency regularizer can penalize changes in output across time.
    Δ is sampled uniformly from the range at __getitem__ time.
    If start+Δ runs past the segment end, Δ is clamped to the largest valid
    value (down to 0, which duplicates the window — consistency MSE then 0).

    === IMU ADDITION =====================================================
    When ``return_imu=True``, every item additionally yields the aligned
    IMU window (shape ``(6, W)``) right after the EMG tensor. The
    consistency-pair branch yields IMU for both timesteps too. Set
    ``imu_mean`` / ``imu_std`` to z-score the IMU stream — these are kept
    separate from EMG stats because the scales are wildly different.
    =======================================================================
    """

    def __init__(self, segments: list[Segment],
                 window: int = 100, step: int = 25,
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None,
                 pair_delta_range: tuple[int, int] | None = None,
                 rng_seed: int = 0,
                 # === IMU ADDITION ===
                 return_imu: bool = False,
                 imu_mean: np.ndarray | None = None,
                 imu_std: np.ndarray | None = None):
        self.segments = segments
        self.W = window
        self.step = step
        self.mean = mean
        self.std = std
        self.pair_delta_range = pair_delta_range  # (min_delta, max_delta)
        self._rng = random.Random(rng_seed)
        # === IMU ADDITION ===
        self.return_imu = return_imu
        self.imu_mean = imu_mean
        self.imu_std = imu_std

        # Pre-compute (seg_idx, start) pairs
        self.index: list[tuple[int, int]] = []
        for si, seg in enumerate(segments):
            T = seg.emg.shape[0]
            for start in range(0, T - window + 1, step):
                self.index.append((si, start))

    def _window_tensor(self, seg: Segment, s: int) -> torch.Tensor:
        w = seg.emg[s:s + self.W]
        if self.mean is not None:
            w = (w - self.mean) / self.std
        return torch.from_numpy(w).float().transpose(0, 1)  # (8, W)

    # === IMU ADDITION =====================================================
    # Slice + z-score the IMU window the same way as EMG and return as
    # (6, W) tensor so the multistream model can do Conv1d directly.
    # ======================================================================
    def _imu_window_tensor(self, seg: Segment, s: int) -> torch.Tensor:
        w = seg.imu[s:s + self.W]
        if self.imu_mean is not None:
            w = (w - self.imu_mean) / self.imu_std
        return torch.from_numpy(w).float().transpose(0, 1)  # (6, W)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, s = self.index[idx]
        seg = self.segments[si]
        emg_t = self._window_tensor(seg, s)
        lbl_t = torch.from_numpy(seg.label).float()
        # === IMU ADDITION ===
        imu_t = self._imu_window_tensor(seg, s) if self.return_imu else None

        if self.pair_delta_range is None:
            if self.return_imu:
                return emg_t, imu_t, lbl_t
            return emg_t, lbl_t

        dmin, dmax = self.pair_delta_range
        T = seg.emg.shape[0]
        max_valid = T - self.W - s                     # largest Δ that still fits
        if max_valid <= 0:
            emg_pair = emg_t                           # duplicate, MSE will be 0
            # === IMU ADDITION === pair the IMU view at the same offset.
            imu_pair = imu_t if self.return_imu else None
        else:
            hi = min(dmax, max_valid)
            lo = min(dmin, hi)
            d = self._rng.randint(lo, hi) if hi > lo else hi
            emg_pair = self._window_tensor(seg, s + d)
            imu_pair = (self._imu_window_tensor(seg, s + d)
                        if self.return_imu else None)

        if self.return_imu:
            return emg_t, imu_t, emg_pair, imu_pair, lbl_t
        return emg_t, emg_pair, lbl_t
