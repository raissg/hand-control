"""Diagnostic: is there actually signal variety in dataset.hdf5?

Prints for each split and channel:
  EMG/IMU: mean, std, min, max, and std-over-time of 1-second RMS windows
           (so a channel that's mostly DC gets flagged even if its raw std is large).

  Landmarks: per-coordinate std across all samples, and the spread of the
             whole-hand "pose signature" (norm of each frame minus the mean frame).
             If pose variety is tiny, the model literally can't learn to move.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def summarize_channel(name: str, x: np.ndarray, fs: int = 500) -> None:
    """x: (T,) time series. Prints stats + how much the signal *energy* varies over time."""
    mean = x.mean()
    std = x.std()
    mn, mx = x.min(), x.max()
    # 1-second non-overlapping RMS windows -> std of that tells us how much the
    # channel's activity envelope changes over the session (not just DC spread).
    win = fs
    n_full = len(x) // win
    rms_env = np.sqrt(
        (x[: n_full * win].reshape(n_full, win) ** 2).mean(axis=1)
    ) if n_full > 0 else np.array([0.0])
    env_mean = rms_env.mean()
    env_std = rms_env.std()
    env_cv = env_std / (env_mean + 1e-9)  # coefficient of variation of envelope
    print(
        f"  {name:10s} mean={mean:+10.3f} std={std:9.3f} "
        f"range=[{mn:+10.2f}, {mx:+10.2f}] | "
        f"1s-RMS mean={env_mean:9.3f} std={env_std:9.3f} CV={env_cv:.3f}"
    )


def check_split(h5f: h5py.File, split: str) -> None:
    if split not in h5f:
        print(f"\n[{split}] missing")
        return
    grp = h5f[split]
    sessions = list(grp.keys())
    if not sessions:
        print(f"\n[{split}] no sessions")
        return

    # Concat all sessions in this split
    emg_list, lm_list = [], []
    for s in sessions:
        emg_list.append(grp[s]["emg"][:])
        lm_list.append(grp[s]["landmarks"][:])
    emg = np.concatenate(emg_list, axis=0)  # (T, 14)
    lm = np.concatenate(lm_list, axis=0)    # (T, 60)

    print(f"\n[{split}] {len(sessions)} session(s), {len(emg)} samples ({len(emg)/500:.1f}s)")

    print(" EMG + IMU channels:")
    emg_names = [f"emg_{i}" for i in range(8)] + [
        "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"
    ]
    for i, n in enumerate(emg_names):
        summarize_channel(n, emg[:, i])

    print(" Landmarks (60 coords = 20 joints x 3):")
    lm_std = lm.std(axis=0)
    lm_range = lm.max(axis=0) - lm.min(axis=0)
    print(f"  per-coord std:   min={lm_std.min():.4f}  median={np.median(lm_std):.4f}  max={lm_std.max():.4f}")
    print(f"  per-coord range: min={lm_range.min():.4f}  median={np.median(lm_range):.4f}  max={lm_range.max():.4f}")

    # "Pose signature" variety: how far each frame deviates from the mean pose.
    mean_pose = lm.mean(axis=0)
    deviations = np.linalg.norm(lm - mean_pose, axis=1)  # (T,)
    print(
        f"  pose deviation from mean (L2 over 60 coords): "
        f"mean={deviations.mean():.4f}  std={deviations.std():.4f}  "
        f"min={deviations.min():.4f}  max={deviations.max():.4f}"
    )

    # Per-joint movement: how much each of the 20 joints moves (std of 3D position)
    lm_3d = lm.reshape(-1, 20, 3)
    joint_std = lm_3d.std(axis=0).mean(axis=1)  # (20,) avg over xyz
    print(
        f"  per-joint movement (mean-xyz std): "
        f"min={joint_std.min():.4f} (joint {joint_std.argmin()}) "
        f"max={joint_std.max():.4f} (joint {joint_std.argmax()})"
    )

    # Fingertip indices in MediaPipe after wrist removed: original wrist=0 dropped,
    # so tips {4,8,12,16,20} -> {3,7,11,15,19} in our 20-joint array.
    tips = [3, 7, 11, 15, 19]
    tip_names = ["thumb", "index", "middle", "ring", "pinky"]
    print("  fingertip movement (std of 3D position):")
    for idx, n in zip(tips, tip_names):
        print(f"    {n:6s} tip: {joint_std[idx]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.hdf5")
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        if "input_mean" in f.attrs:
            im = np.asarray(f.attrs["input_mean"])
            is_ = np.asarray(f.attrs["input_std"])
            print(f"Stored input_mean: {np.round(im, 3).tolist()}")
            print(f"Stored input_std : {np.round(is_, 3).tolist()}")
        for split in ["train", "val", "test"]:
            check_split(f, split)


if __name__ == "__main__":
    main()
