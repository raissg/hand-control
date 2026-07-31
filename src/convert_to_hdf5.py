import os
import glob
import yaml
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirnotch
import argparse
import matplotlib.pyplot as plt

from src.clean_landmarks import clean_landmarks
from src.landmarks_to_angles import (
    ANGLE_NAMES,
    NUM_ANGLES,
    landmarks_seq_to_angles,
)

# EMG samples whose nearest clean landmark is farther than this (in seconds)
# are excluded — don't fabricate ground truth across long detection gaps.
MAX_INTERP_GAP_S = 0.25
# Sub-sessions shorter than one training window are dropped entirely.
MIN_SUBSESSION_SAMPLES = 1000

def load_config(config_path="src/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

IMU_COLS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']


def filter_emg(emg, fs, low=20.0, high=240.0, notch=50.0, order=4):
    """Zero-phase bandpass + notch filter over a continuous EMG stream.

    Applied offline on the full session. filtfilt runs the filter forward
    and backward to cancel phase, and uses persistent state — no per-batch
    zeroing like the real-time filter in acquire.py.

    Defaults: 20–240 Hz band (Nyquist for 500 Hz sampling is 250 Hz).
    notch default is 50 Hz — measured on this rig via check_spectrum.py;
    60 Hz left the real mains spike (12x baseline) untouched in training data.

    emg: (T, C) array of raw EMG samples.
    """
    nyq = 0.5 * fs
    # Clamp high cutoff below Nyquist so butter() doesn't explode
    high = min(high, 0.95 * nyq)
    low = max(low, 1.0)
    b_bp, a_bp = butter(order, [low / nyq, high / nyq], btype='band')
    b_n, a_n = iirnotch(notch / nyq, Q=30.0)
    # filtfilt along time axis
    out = filtfilt(b_bp, a_bp, emg, axis=0)
    out = filtfilt(b_n, a_n, out, axis=0)
    return out.astype(np.float32)


def process_session(
    session_dir,
    emg_cols,
    plot_debug=False,
    filter_enable=True,
    include_imu=True,
    target_type="landmarks",
):
    """
    Process a single recording session:
    - Load emg.csv and landmarks.csv
    - Align 30Hz landmarks to 500Hz EMG timestamps

    filter_enable=False skips the offline bandpass+notch on EMG. Use this
    to build a raw-EMG dataset and test whether filtering is destructive.

    include_imu=False builds an input matrix of EMG columns only (still filtered
    when filter_enable=True). IMU is ignored even if present in emg.csv.

    target_type selects the regression target:
      - "landmarks": 20×3 = 60 wrist-relative coords (legacy pipeline).
      - "angles":    20 joint angles in radians (NeuroPose-aligned). Angles are
                    computed from clean 30 Hz landmarks, THEN interpolated to
                    500 Hz — cheaper than interpolating landmarks first and
                    more robust near finger-curl extremes.

    Returns (timestamps, emg_data, target_data, valid) or None.
    """
    session_path = Path(session_dir)
    emg_path = session_path / "emg.csv"
    landmarks_path = session_path / "landmarks.csv"
    angles_path = session_path / "angles.csv"

    # angles.csv (written by postprocess_angles) takes priority when target=angles
    use_angles_csv = target_type == "angles" and angles_path.exists()

    if not emg_path.exists():
        print(f"Skipping {session_dir} (missing emg.csv)")
        return None
    if not use_angles_csv and not landmarks_path.exists():
        print(f"Skipping {session_dir} (missing landmarks.csv and angles.csv)")
        return None
        
    print(f"Processing session: {session_dir}")
    
    # Read EMG data
    df_emg = pd.read_csv(emg_path)
    # Read ground truth
    if use_angles_csv:
        df_landmarks = pd.read_csv(angles_path)
        print("  Using angles.csv (direct MediaPipe angles)")
    else:
        df_landmarks = pd.read_csv(landmarks_path)
    
    # Drop rows with NaN if any
    input_cols = list(emg_cols)
    if include_imu:
        input_cols.extend(IMU_COLS)
    # Some older recordings may not have IMU columns; tolerate that.
    input_cols = [c for c in input_cols if c in df_emg.columns]
    df_emg = df_emg.dropna(subset=['timestamp'] + input_cols)
    df_landmarks = df_landmarks.dropna(subset=['timestamp'])

    if len(df_emg) == 0 or len(df_landmarks) < 2:
        print(f"Skipping {session_dir} (insufficient data)")
        return None

    # --- Clean landmark detection errors (handedness flips, hallucinated coords).
    # Only applicable when reading landmarks.csv — angles.csv doesn't carry
    # the raw coord columns or handedness needed by clean_landmarks.
    if not use_angles_csv:
        df_landmarks, clean_stats = clean_landmarks(df_landmarks, session_name=session_path.name)
        print(f"  {clean_stats.report()}")
        if len(df_landmarks) < 2:
            print(f"Skipping {session_dir} (no clean landmarks left after filtering)")
            return None
        
    # Extract timestamps
    t_emg = df_emg['timestamp'].values
    t_landmarks = df_landmarks['timestamp'].values
    
    # Common time window (only where we have both EMG and landmarks)
    t_start = max(t_emg.min(), t_landmarks.min())
    t_end = min(t_emg.max(), t_landmarks.max())
    
    # Filter arrays to the common time window
    mask_emg = (t_emg >= t_start) & (t_emg <= t_end)
    t_emg_sync = t_emg[mask_emg]
    emg_data = df_emg.loc[mask_emg, input_cols].values.astype(np.float32)

    # Offline filter on EMG channels only (IMU is raw accel/gyro, no filter).
    emg_ch_idx = [input_cols.index(c) for c in emg_cols if c in input_cols]
    if filter_enable and emg_ch_idx and len(emg_data) > 30:  # filtfilt needs enough samples
        fs = 1.0 / np.median(np.diff(t_emg_sync)) if len(t_emg_sync) > 1 else 500.0
        emg_only = emg_data[:, emg_ch_idx]
        emg_data[:, emg_ch_idx] = filter_emg(emg_only, fs=fs)
        print(f"  Filtered {len(emg_ch_idx)} EMG channels (fs={fs:.1f} Hz)")
    elif not filter_enable:
        print(f"  Filter DISABLED — EMG channels passed through raw")
    if not include_imu:
        print("  Dataset input: EMG channels only (IMU omitted)")

    # --- Build the target (landmarks or angles) at 30 Hz, then interpolate. ---
    if target_type == "angles":
        if use_angles_csv:
            # angles.csv already has 20 precomputed angle columns
            angle_cols = [c for c in df_landmarks.columns if c not in {'frame_index', 'timestamp'}]
            target_30hz = df_landmarks[angle_cols].values.astype(np.float64)
            assert target_30hz.shape[1] == NUM_ANGLES, \
                f"angles.csv has {target_30hz.shape[1]} cols, expected {NUM_ANGLES}"
            target_dim = NUM_ANGLES
            print(f"  Loaded {NUM_ANGLES} angles per frame from angles.csv ({target_30hz.shape[0]} rows)")
        else:
            # Compute angles from landmarks.csv
            non_coord_cols = {'frame_index', 'timestamp', 'handedness', 'scale'}
            landmark_cols = [c for c in df_landmarks.columns if c not in non_coord_cols]
            raw = df_landmarks[landmark_cols].values.astype(np.float64)
            assert raw.shape[1] == 63, \
                f"target=angles needs all 21 landmarks (63 cols), got {raw.shape[1]}"
            lm21 = raw.reshape(-1, 21, 3)
            target_30hz = landmarks_seq_to_angles(lm21).astype(np.float64)
            target_dim = NUM_ANGLES
            print(f"  Computed {NUM_ANGLES} angles per frame from {lm21.shape[0]} landmark rows")
    else:
        # Legacy: 20 landmarks × 3 = 60 coords (wrist dropped — already origin).
        non_coord_cols = {'frame_index', 'timestamp', 'handedness', 'scale',
                          'x0', 'y0', 'z0'}
        landmark_cols = [c for c in df_landmarks.columns if c not in non_coord_cols]
        target_30hz = df_landmarks[landmark_cols].values.astype(np.float64)
        target_dim = 60
        assert target_30hz.shape[1] == target_dim, \
            f"Expected {target_dim} landmark coords (20×3), got {target_30hz.shape[1]}"

    # Interpolation (same linear interp works for both angles and landmarks)
    print(f"  Interpolating {len(target_30hz)} {target_type} frames to {len(t_emg_sync)} EMG samples...")
    interpolator = interp1d(t_landmarks, target_30hz, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    landmarks_sync = interpolator(t_emg_sync)  # name kept for downstream compat

    # --- Valid-sample mask: EMG samples where the nearest two clean landmarks
    # bracket them within MAX_INTERP_GAP_S. Outside this, interpolation would
    # be fabricating ground truth across too-large a detection gap.
    #
    # t_landmarks is already the CLEANED landmark timestamps (bad rows removed).
    # A gap in this array means either genuinely missed detections or a
    # cleaning-induced removal — both are legitimate reasons to not trust
    # the interpolated value there.
    idx = np.searchsorted(t_landmarks, t_emg_sync)
    idx_clipped = np.clip(idx, 1, len(t_landmarks) - 1)
    gap = t_landmarks[idx_clipped] - t_landmarks[idx_clipped - 1]
    valid = (
        (gap <= MAX_INTERP_GAP_S)
        & (t_emg_sync >= t_landmarks[0])
        & (t_emg_sync <= t_landmarks[-1])
    )
    invalid_frac = 1.0 - valid.mean()
    if invalid_frac > 0.001:
        print(f"  Valid-mask: {valid.sum()}/{len(valid)} samples ({100 * valid.mean():.1f}% kept, "
              f"{invalid_frac * 100:.2f}% excluded by {MAX_INTERP_GAP_S}s gap cap)")
    
    if plot_debug:
        # Plot first target dim (first landmark coord or first angle)
        plt.figure(figsize=(10, 4))
        plt.plot(t_landmarks - t_start, target_30hz[:, 0], 'ro', label=f'Original 30Hz ({target_type}[0])')
        plt.plot(t_emg_sync - t_start, landmarks_sync[:, 0], 'b-', alpha=0.6, label='Interpolated 500Hz')
        plt.title(f"Interpolation Debug - {session_path.name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Coordinate Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return t_emg_sync, emg_data, landmarks_sync, valid

def build_dataset(
    config_path="src/config.yaml",
    debug=False,
    session_ids=None,
    filter_enable=True,
    output_name="dataset.hdf5",
    include_imu=True,
    target_type="landmarks",
):
    """Build HDF5 dataset from recorded sessions.

    Args:
        config_path: Path to config.yaml
        debug: Show interpolation debug plots
        session_ids: List of session directory names to process.
                     If None, processes all sessions in recordings/.
        filter_enable: Apply offline bandpass+notch to EMG (default True).
                       Pass False to build a raw-EMG dataset variant.
        output_name: Filename under dataset.output_dir (default dataset.hdf5).
        include_imu: If False, HDF5 ``emg`` datasets are (T, num_emg) with filtered
                     EMG only; attrs ``imu_included=0`` and ``input_channels=num_emg``.
    """
    config = load_config(config_path)

    rec_dir = Path(config['recording']['output_dir'])
    out_dir = Path(config['dataset']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    output_hdf5_path = out_dir / output_name

    # Find sessions
    if session_ids:
        sessions = [rec_dir / sid for sid in session_ids]
        missing = [s for s in sessions if not s.is_dir()]
        if missing:
            print(f"Session directories not found: {missing}")
            return
    else:
        sessions = sorted([d for d in rec_dir.iterdir() if d.is_dir()])
    
    # EMG channels configured
    num_channels = config['emg']['num_channels']
    emg_cols = [f'emg_{i}' for i in range(num_channels)]
    
    # Splitting ratios
    train_r = config['dataset']['train_ratio']
    val_r = config['dataset']['val_ratio']
    test_r = config['dataset']['test_ratio']
    
    processed_sessions = []
    
    for session_dir in sessions:
        res = process_session(
            session_dir,
            emg_cols,
            plot_debug=debug,
            filter_enable=filter_enable,
            include_imu=include_imu,
            target_type=target_type,
        )
        if res is not None:
            processed_sessions.append((session_dir.name, res))
            
    if not processed_sessions:
        print("No complete sessions found. Ensure you have run inference/ground truth post-processing.")
        return
        
    # --- Split each session into contiguous VALID sub-sessions.
    # A "sub-session" is a run of EMG samples whose aligned landmarks are all
    # valid (no large interpolation gaps). Invalid samples act as hard cuts:
    # we never produce a training window that spans one, because each
    # sub-session is its own HDF5 group and MindroveEMGDataset only slides
    # windows inside a group.
    subsessions = []  # list of (session_name, sub_idx, t, emg, lm)
    for session_name, (t_sync, emg_sync, lm_sync, valid) in processed_sessions:
        # Find contiguous True runs in `valid`
        # (np.diff on valid.astype(int): +1 starts a run, -1 ends one)
        v = np.concatenate(([0], valid.astype(np.int8), [0]))
        starts = np.where(np.diff(v) == 1)[0]
        ends = np.where(np.diff(v) == -1)[0]
        kept = 0
        for i, (s, e) in enumerate(zip(starts, ends)):
            if e - s < MIN_SUBSESSION_SAMPLES:
                continue
            subsessions.append((session_name, kept, t_sync[s:e], emg_sync[s:e], lm_sync[s:e]))
            kept += 1
        total_parts = len(starts)
        if total_parts > 1 or kept != total_parts:
            print(f"  {session_name}: split into {total_parts} parts, kept {kept} "
                  f"(dropped {total_parts - kept} shorter than {MIN_SUBSESSION_SAMPLES} samples)")

    if not subsessions:
        print("No usable sub-sessions after cleaning.")
        return

    # Compute per-channel z-score stats from train portion across all sub-sessions
    # (EMG and IMU have very different scales — normalize each channel independently).
    # Done on the FILTERED EMG so stats reflect real muscle amplitude, not DC/drift.
    train_chunks = []
    for _, _, _, emg_sync, _ in subsessions:
        n = len(emg_sync)
        train_chunks.append(emg_sync[:int(n * train_r)])
    train_stack = np.concatenate(train_chunks, axis=0)
    input_mean = train_stack.mean(axis=0).astype(np.float32)
    input_std = train_stack.std(axis=0).astype(np.float32)
    input_std[input_std < 1e-6] = 1.0  # avoid div-by-zero on dead channels

    print(f"\nInput channel stats (n={train_stack.shape[0]} train samples across {len(subsessions)} sub-sessions):")
    for i, (m, s) in enumerate(zip(input_mean, input_std)):
        print(f"  ch{i}: mean={m:.4f}, std={s:.4f}")

    print(f"\nWriting to {output_hdf5_path}...")
    with h5py.File(output_hdf5_path, 'w') as h5f:
        h5f.attrs['input_mean'] = input_mean
        h5f.attrs['input_std'] = input_std
        # Final channel count from the last sub-session's EMG (same for all).
        h5f.attrs['input_channels'] = subsessions[-1][3].shape[1]
        h5f.attrs['imu_included'] = np.uint8(1 if include_imu else 0)
        h5f.attrs['emg_filtered'] = np.uint8(1 if filter_enable else 0)
        h5f.attrs['target_type'] = target_type
        target_dim = NUM_ANGLES if target_type == "angles" else 60
        h5f.attrs['target_dim'] = np.int32(target_dim)
        if target_type == "angles":
            h5f.attrs['angle_names'] = np.array(ANGLE_NAMES, dtype='S32')

        for session_name, sub_idx, t_sync, emg_sync, lm_sync in subsessions:
            # Sub-session name: "<session>" if the session wasn't split, else "<session>_partNN".
            same_session_count = sum(1 for ss in subsessions if ss[0] == session_name)
            sub_name = session_name if same_session_count == 1 else f"{session_name}_part{sub_idx:02d}"

            total_len = len(t_sync)
            train_end = int(total_len * train_r)
            val_end = train_end + int(total_len * val_r)

            splits = {
                'train': (0, train_end),
                'val': (train_end, val_end),
                'test': (val_end, total_len),
            }

            for split_name, (start_idx, end_idx) in splits.items():
                if end_idx <= start_idx:
                    continue

                grp_path = f"{split_name}/{sub_name}"
                grp = h5f.create_group(grp_path)

                grp.create_dataset('timestamp', data=t_sync[start_idx:end_idx])
                grp.create_dataset('emg', data=emg_sync[start_idx:end_idx])
                # Target dataset name mirrors target_type so downstream code can
                # branch on the HDF5 attr without guessing.
                target_key = 'angles' if target_type == 'angles' else 'landmarks'
                grp.create_dataset(target_key, data=lm_sync[start_idx:end_idx])

                print(f"  Created {grp_path}: EMG {emg_sync[start_idx:end_idx].shape}, "
                      f"{target_key} {lm_sync[start_idx:end_idx].shape}")
                
    print(
        "\nDataset generation complete "
        f"(imu_included={int(include_imu)}, emg_filtered={int(filter_enable)}, "
        f"input_channels={subsessions[-1][3].shape[1]})."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert recorded CSVs to grouped HDF5 dataset.")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Show interpolation debug plots")
    parser.add_argument("--sessions", nargs="*", default=None, help="Specific session IDs to process (default: all)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip bandpass+notch on EMG (build raw-EMG dataset variant).")
    parser.add_argument(
        "--no-imu",
        action="store_true",
        help="Omit accel/gyro from HDF5 inputs (EMG only; still filtered unless --no-filter).",
    )
    parser.add_argument("--output", type=str, default="dataset.hdf5",
                        help="Output HDF5 filename under dataset.output_dir.")
    parser.add_argument("--target", type=str, default="landmarks",
                        choices=["landmarks", "angles"],
                        help="Regression target: 'landmarks' (60-D coords, legacy) "
                             "or 'angles' (20-D joint angles in radians, NeuroPose-aligned).")
    args = parser.parse_args()

    out = args.output
    if out == "dataset.hdf5":
        parts = ["dataset"]
        if args.target == "angles":
            parts.append("angles")
        if args.no_imu:
            parts.append("no_imu")
        if args.no_filter:
            parts.append("raw_emg")
        out = "_".join(parts) + ".hdf5"

    build_dataset(
        config_path=args.config,
        debug=args.debug,
        session_ids=args.sessions,
        filter_enable=not args.no_filter,
        output_name=out,
        include_imu=not args.no_imu,
        target_type=args.target,
    )
