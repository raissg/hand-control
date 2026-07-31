"""
Session viewer: video + skeleton on the left, interactive 3D landmarks on the right.

Left: recorded video with optional MediaPipe overlay and optional ``landmarks.csv``
skeleton on the frame (same style as ``playback.py``).

Right: large rotatable 3D hand (no axis grid); uses ``landmarks.csv`` when present,
otherwise the last MediaPipe detection. **Click and drag** on the 3D panel to orbit
(matplotlib default 3D mouse controls).

Usage:
    python src/session_3d_viewer.py 20260415_182040
    python src/session_3d_viewer.py 20260415_182040 --speed 0.5
    python src/session_3d_viewer.py 20260415_182040 --no-overlay

Controls:
    q / ESC       quit
    space / p     pause / resume
    Return        pause / resume
    Pause button  lower right on video
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — register 3d projection

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.ground_truth import HandPoseExtractor, HAND_CONNECTIONS, WRIST  # noqa: E402
from src.playback import (  # noqa: E402
    draw_csv_landmarks_bgr,
    draw_landmarks,
    load_session,
    lookup_landmark_row,
    row_to_landmarks_xyz,
)


def _read_3d_view(ax) -> tuple[float, float]:
    try:
        return float(ax.elev), float(ax.azim)
    except (AttributeError, TypeError, ValueError):
        return -22.0, -60.0


def _landmarks_from_mediapipe_list(hand_landmarks) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float64,
    )


def _style_3d_hand_axes(ax) -> None:
    """No grid lines / pane fill — hand only."""
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((1, 1, 1, 0))
        except (AttributeError, TypeError):
            pass
        try:
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0)  # noqa: SLF001
        except (AttributeError, KeyError, TypeError):
            pass
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


def redraw_skeleton_3d(
    ax,
    xyz: Optional[np.ndarray],
    elev: float,
    azim: float,
    _subtitle: str = "",
) -> None:
    ax.clear()
    if xyz is None or xyz.size == 0:
        ax.text(0, 0, 0, "No pose", ha="center", va="center", fontsize=12)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        _style_3d_hand_axes(ax)
        return

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for a, b in HAND_CONNECTIONS:
        ax.plot(
            [x[a], x[b]],
            [y[a], y[b]],
            [z[a], z[b]],
            color="tab:cyan",
            lw=3.0,
        )
    ax.scatter(x, y, z, c="tab:orange", s=55, depthshade=True)
    ax.scatter([x[WRIST]], [y[WRIST]], [z[WRIST]], c="tab:red", s=120, depthshade=True)

    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    span = float(np.max(hi - lo)) + 1e-6
    pad = 0.04 * span
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    if hasattr(ax, "set_box_aspect"):
        dx = hi[0] - lo[0] + 2 * pad
        dy = hi[1] - lo[1] + 2 * pad
        dz = hi[2] - lo[2] + 2 * pad
        ax.set_box_aspect((max(dx, 1e-6), max(dy, 1e-6), max(dz, 1e-6)))

    ax.view_init(elev=elev, azim=azim)
    _style_3d_hand_axes(ax)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session_id", help="Directory name under recordings/")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0 = real-time)")
    ap.add_argument("--no-overlay", action="store_true", help="Skip MediaPipe on video")
    ap.add_argument(
        "--no-csv-on-video",
        action="store_true",
        help="Do not draw CSV skeleton on the video frame (3D panel still uses CSV if present)",
    )
    args = ap.parse_args()

    session_dir, emg, frame_times, video, landmarks = load_session(args.session_id)
    print(f"Loaded {session_dir.name}: {len(frame_times)} frames", end="")
    if landmarks is not None:
        print(f", landmarks.csv ({len(landmarks)} rows)")
    else:
        print(" (no landmarks.csv — 3D uses MediaPipe only when overlay is on)")

    t_emg = emg["timestamp"].values.astype(np.float64)
    t_frame = frame_times["timestamp"].values.astype(np.float64)
    frame_idx_col = frame_times["frame_index"].values.astype(np.int64)
    t_start = min(float(t_emg[0]), float(t_frame[0]))
    t_end = max(float(t_emg[-1]), float(t_frame[-1]))
    duration = t_end - t_start

    extractor = None if args.no_overlay else HandPoseExtractor(running_mode="video")

    fig = plt.figure(figsize=(14, 8))
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "set_window_title"):
        mgr.set_window_title(f"3D viewer — {session_dir.name}")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.94, bottom=0.02, wspace=0.06)

    # Video is a strip on the left; ~3/4 of the window is the rotatable 3D hand.
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.26, 0.74], wspace=0.03)
    ax_vid = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_vid.axis("off")
    vid_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    vid_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    vid_im = ax_vid.imshow(np.zeros((vid_h, vid_w, 3), dtype=np.uint8), aspect="equal")
    ax_vid.set_aspect("equal", adjustable="box")
    vid_title = ax_vid.set_title(f"{session_dir.name}  t=0.00s")
    redraw_skeleton_3d(ax_3d, None, -22.0, -60.0)

    paused = False
    wall_start = time.time()
    sim_offset = 0.0
    pause_btn_holder: dict[str, Optional[Button]] = {"btn": None}

    def _sync_pause_label() -> None:
        b = pause_btn_holder["btn"]
        if b is not None:
            b.label.set_text("Resume" if paused else "Pause")

    def toggle_pause(_event=None) -> None:
        nonlocal paused, wall_start, sim_offset
        if paused:
            wall_start = time.time() - sim_offset / args.speed
            paused = False
        else:
            sim_offset = (time.time() - wall_start) * args.speed
            paused = True
        _sync_pause_label()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key is None:
            return
        if event.key in ("q", "escape"):
            plt.close(fig)
        elif event.key in (" ", "p", "P", "enter", "return"):
            toggle_pause()

    fig.canvas.mpl_connect("key_press_event", on_key)
    btn_ax = inset_axes(ax_vid, width="16%", height="7%", loc="lower right", borderpad=0.5)
    pause_btn = Button(btn_ax, "Pause", color="0.85", hovercolor="0.75")
    pause_btn.on_clicked(toggle_pause)
    pause_btn_holder["btn"] = pause_btn

    current_frame_pos = 0
    current_frame_bgr = None

    def advance_to_frame(target_idx: int):
        nonlocal current_frame_pos, current_frame_bgr
        while current_frame_pos <= target_idx:
            ret, f = video.read()
            if not ret:
                return current_frame_bgr
            current_frame_bgr = f
            current_frame_pos += 1
        return current_frame_bgr

    last_mp_frame_idx = -1
    last_mp_ts_ms = -1
    last_mp_landmarks = None

    plt.show(block=False)
    try:
        while plt.fignum_exists(fig.number):
            if paused:
                t_sim = sim_offset
            else:
                t_sim = (time.time() - wall_start) * args.speed
            t_now = t_start + t_sim
            if t_now > t_end:
                break

            frame_mask = t_frame <= t_now
            if not frame_mask.any():
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.08 if paused else 0.005)
                continue

            i = int(np.searchsorted(t_frame, t_now, side="right") - 1)
            target_idx = int(frame_idx_col[i])
            frame_bgr = advance_to_frame(target_idx)
            if frame_bgr is None:
                plt.pause(0.02)
                continue

            disp = frame_bgr.copy()
            xyz_3d: Optional[np.ndarray] = None

            if landmarks is not None:
                csv_row = lookup_landmark_row(landmarks, target_idx)
                if csv_row is not None:
                    xyz_3d = row_to_landmarks_xyz(csv_row)
                    if not args.no_csv_on_video:
                        draw_csv_landmarks_bgr(disp, xyz_3d)

            if extractor is not None:
                import mediapipe as mp
                if target_idx != last_mp_frame_idx:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts_ms = int(t_frame[i] * 1000)
                    if ts_ms <= last_mp_ts_ms:
                        ts_ms = last_mp_ts_ms + 1
                    last_mp_ts_ms = ts_ms
                    last_mp_frame_idx = target_idx
                    res = extractor._landmarker.detect_for_video(mp_img, ts_ms)
                    if res.hand_landmarks:
                        last_mp_landmarks = res.hand_landmarks[0]
                    else:
                        last_mp_landmarks = None
                if last_mp_landmarks is not None:
                    draw_landmarks(disp, last_mp_landmarks)
                    if xyz_3d is None:
                        xyz_3d = _landmarks_from_mediapipe_list(last_mp_landmarks)

            elev, azim = _read_3d_view(ax_3d)
            redraw_skeleton_3d(ax_3d, xyz_3d, elev, azim)

            vid_im.set_data(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            vid_title.set_text(f"{session_dir.name}  t={t_sim:.2f}s / {duration:.1f}s")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.08 if paused else 0.005)
    finally:
        video.release()
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    main()
