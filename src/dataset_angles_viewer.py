# src/dataset_angles_viewer.py
"""Playback: recorded video + skeleton overlay + HDF5 joint angles (dataset_angles).

``dataset_angles.hdf5`` stores **angles** (20, rad) and **timestamps** aligned to EMG —
it does **not** store landmarks. Landmarks are read from ``recordings/<session>/landmarks.csv``
(same MediaPipe-normalized pose used to build the angles). Per video frame we:

1. Look up **landmarks** by ``frame_index`` (with hold-forward on gaps, like ``playback.py``).
2. Map **camera time** → nearest HDF5 row by ``timestamp`` in the chosen HDF5 group
   (handles ``_partNN`` subsessions).

Usage:
    python -m src.dataset_angles_viewer train/20260416_124422
    python -m src.dataset_angles_viewer 20260416_124422 --dataset data/dataset_angles.hdf5

Controls: SPACE pause, q/ESC quit, n/p next/prev frame (when paused).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.landmarks_to_angles import ANGLE_NAMES  # noqa: E402
from src.playback import (  # noqa: E402
    RECORDINGS_DIR,
    draw_csv_landmarks_bgr,
    lookup_landmark_row,
    row_to_landmarks_xyz,
)

NUM_LM = 21
NUM_ANGLES = 20


def recording_id_from_h5_group(group_name: str) -> str:
    """Strip ``_part00`` subsession suffix to get folder under recordings/."""
    return re.sub(r"_part\d+$", "", group_name)


def resolve_h5_group(h5f: h5py.File, session_arg: str) -> str:
    """Return ``split/name`` path. Accepts ``train/foo`` or ``foo``."""
    if "/" in session_arg:
        if session_arg not in h5f:
            raise KeyError(f"No HDF5 group: {session_arg}")
        return session_arg
    for split in ("train", "val", "test"):
        if split not in h5f:
            continue
        g = h5f[split]
        if session_arg in g:
            return f"{split}/{session_arg}"
        for name in g.keys():
            if name == session_arg or name.startswith(session_arg):
                return f"{split}/{name}"
    raise KeyError(f"Session not found in HDF5: {session_arg!r}")


def angle_names_from_h5(h5f: h5py.File) -> list[str]:
    if "angle_names" not in h5f.attrs:
        return list(ANGLE_NAMES)
    raw = h5f.attrs["angle_names"]
    out = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode("ascii", errors="replace"))
        else:
            out.append(str(x))
    return out


def nearest_index(ts_arr: np.ndarray, t_query: float) -> int:
    """Index of EMG/HDF5 row whose timestamp is closest to ``t_query``."""
    if len(ts_arr) == 0:
        return 0
    i = int(np.searchsorted(ts_arr, t_query))
    if i <= 0:
        return 0
    if i >= len(ts_arr):
        return len(ts_arr) - 1
    if abs(ts_arr[i] - t_query) < abs(ts_arr[i - 1] - t_query):
        return i
    return i - 1


def render_sidebar(
    height: int,
    angle_deg: np.ndarray,
    names: list[str],
    meta: str,
) -> np.ndarray:
    """BGR panel with one line per joint angle."""
    w = 340
    pad = np.zeros((height, w, 3), dtype=np.uint8)
    pad[:] = (28, 28, 28)
    y = 22
    cv2.putText(pad, "Joint angles (deg)", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 255), 1)
    y += 24
    for line in meta.split("\n"):
        cv2.putText(pad, line[:42], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 180), 1)
        y += 18
    y += 8
    for i in range(min(len(names), len(angle_deg))):
        label = names[i][:18]
        cv2.putText(
            pad,
            f"{label:18s} {angle_deg[i]:7.2f}",
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 255, 200) if i % 2 == 0 else (220, 240, 220),
            1,
        )
        y += 17
        if y > height - 12:
            break
    return pad


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "session",
        help="HDF5 group name or basename, e.g. train/20260416_124422 or 20260416_124422",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "dataset_angles.hdf5"),
        help="Path to dataset_angles.hdf5",
    )
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.is_file():
        print(f"Dataset not found: {ds_path}")
        sys.exit(1)

    win = "Dataset angles + video (q/ESC quit, SPACE pause)"

    with h5py.File(ds_path, "r") as h5f:
        grp_path = resolve_h5_group(h5f, args.session.strip().strip("/"))
        grp = h5f[grp_path]
        if "angles" not in grp or "timestamp" not in grp:
            print(f"Group {grp_path} missing angles or timestamp")
            sys.exit(1)
        angles = np.asarray(grp["angles"])
        ts_h5 = np.asarray(grp["timestamp"], dtype=np.float64)
        names = angle_names_from_h5(h5f)
        if len(names) != NUM_ANGLES:
            names = list(ANGLE_NAMES)

        group_leaf = grp_path.split("/")[-1]
        rec_id = recording_id_from_h5_group(group_leaf)
        session_dir = RECORDINGS_DIR / rec_id
        if not session_dir.is_dir():
            print(f"Recording folder not found: {session_dir}")
            sys.exit(1)

        video_path = session_dir / "video.mp4"
        times_path = session_dir / "frame_times.csv"
        lm_path = session_dir / "landmarks.csv"
        if not video_path.is_file() or not times_path.is_file():
            print(f"Need {video_path} and {times_path}")
            sys.exit(1)

        frame_times = pd.read_csv(times_path)
        landmarks = None
        if lm_path.is_file():
            landmarks = pd.read_csv(lm_path)
            landmarks.columns = [c.strip() for c in landmarks.columns]
        else:
            print(f"Warning: no {lm_path} — skeleton overlay disabled.")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video {video_path}")
            sys.exit(1)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay_ms = max(1, int(1000 / (fps_src * args.speed)))

        # Restrict lookup to HDF5 time span (sub-sessions).
        t_min, t_max = float(ts_h5[0]), float(ts_h5[-1])

        paused = False
        f = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                f = 0
                continue

            frame = cv2.flip(frame, 1)
            h = frame.shape[0]

            ft_row = min(f, len(frame_times) - 1) if len(frame_times) else 0
            if len(frame_times):
                t_vid = float(frame_times.iloc[ft_row]["timestamp"])
            else:
                t_vid = t_min

            in_range = t_min <= t_vid <= t_max
            idx = nearest_index(ts_h5, t_vid) if in_range else 0
            ang_rad = angles[idx] if in_range else np.full(NUM_ANGLES, np.nan)
            ang_deg = np.degrees(ang_rad)

            if landmarks is not None:
                row = lookup_landmark_row(landmarks, int(f))
                if row is not None:
                    xyz = row_to_landmarks_xyz(row)
                    draw_csv_landmarks_bgr(frame, xyz)

            overlay = (
                f"{grp_path}  frame {f}/{n_frames}  h5_idx {idx}/{len(ts_h5)}"
                + ("" if in_range else "  [t out of h5 range]")
            )
            cv2.putText(
                frame,
                overlay[: min(80, len(overlay))],
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            meta = (
                f"recording: {rec_id}\n"
                f"t_vid={t_vid:.3f}s  nearest h5 row\n"
                f"paused={'Y' if paused else 'N'}"
            )
            side = render_sidebar(h, ang_deg, names, meta)
            combo = np.hstack([frame, side])

            cv2.imshow(win, combo)
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32:  # space
                paused = not paused
            if paused:
                if key in (ord("n"), ord("N")):
                    f = min(f + 1, max(0, n_frames - 1))
                elif key in (ord("p"), ord("P")):
                    f = max(f - 1, 0)
            else:
                f += 1
                if f >= n_frames:
                    f = 0

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
