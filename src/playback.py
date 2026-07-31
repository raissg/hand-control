"""
Synchronized playback of a recorded session.

Replays the video with MediaPipe hand overlay, driven off the shared wall-clock
timestamps written during recording.

If ``recordings/<session>/landmarks.csv`` exists (e.g. from offline processing),
the same frame index is used to plot **3D landmarks** as four **orthographic**
views below the video (front / top / side / bottom) and to draw a second
skeleton overlay on the video (cyan) in addition to MediaPipe when enabled.

Usage:
    python src/playback.py 20260415_134100
    python src/playback.py 20260415_134100 --speed 0.5    # half speed
    python src/playback.py 20260415_134100 -o playback.mp4   # save layout as video, then exit

Controls (interactive only):
    q / ESC       quit
    space / p     pause / resume (keyboard)
    Return        pause / resume (keyboard)
    Pause button  pause / resume (lower right on video; works without key focus)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


def _argv_requests_video_export(argv: list[str]) -> bool:
    if "--output" in argv:
        return True
    if any(a.startswith("--output=") for a in argv):
        return True
    if "-o" in argv:
        return True
    return False


if _argv_requests_video_export(sys.argv):
    import matplotlib

    matplotlib.use("Agg", force=True)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.ground_truth import HandPoseExtractor, HAND_CONNECTIONS, WRIST  # noqa: E402

RECORDINGS_DIR = _PROJECT_ROOT / "recordings"
NUM_LM = 21
LM_COORD_COLS = [c for i in range(NUM_LM) for c in (f"x{i}", f"y{i}", f"z{i}")]


def load_session(session_id: str):
    """Load emg.csv, frame_times.csv, video handle for a session."""
    session_dir = RECORDINGS_DIR / session_id
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    emg = pd.read_csv(session_dir / "emg.csv")
    # strip accidental whitespace in headers
    emg.columns = [c.strip() for c in emg.columns]

    frame_times = pd.read_csv(session_dir / "frame_times.csv")

    video = cv2.VideoCapture(str(session_dir / "video.mp4"))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {session_dir / 'video.mp4'}")

    lm_path = session_dir / "landmarks.csv"
    landmarks = None
    if lm_path.is_file():
        landmarks = pd.read_csv(lm_path)
        landmarks.columns = [c.strip() for c in landmarks.columns]
        miss = [c for c in LM_COORD_COLS + ["frame_index"] if c not in landmarks.columns]
        if miss:
            print(f"Warning: landmarks.csv missing columns {miss[:5]}… — ignoring file")
            landmarks = None
        else:
            landmarks = landmarks.sort_values("frame_index").reset_index(drop=True)

    return session_dir, emg, frame_times, video, landmarks


def draw_landmarks(frame_bgr, img_landmarks):
    """Draw MediaPipe landmarks + connections on a BGR frame (in place)."""
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in img_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 200), 2)
    for i, (cx, cy) in enumerate(pts):
        color = (0, 0, 255) if i == WRIST else (0, 255, 0)
        r = 6 if i == WRIST else 3
        cv2.circle(frame_bgr, (cx, cy), r, color, -1)


def row_to_landmarks_xyz(row: pd.Series) -> np.ndarray:
    """(21, 3) float from one landmarks.csv row."""
    return np.array(
        [[float(row[f"x{i}"]), float(row[f"y{i}"]), float(row[f"z{i}"])] for i in range(NUM_LM)],
        dtype=np.float64,
    )


def lookup_landmark_row(landmarks: pd.DataFrame, frame_idx: int) -> Optional[pd.Series]:
    """Last CSV row with frame_index <= frame_idx (hold when gaps)."""
    sub = landmarks[landmarks["frame_index"] <= frame_idx]
    if sub.empty:
        return None
    return sub.iloc[-1]


def draw_csv_landmarks_bgr(frame_bgr: np.ndarray, xyz: np.ndarray) -> None:
    """Project normalized 3D hand onto the video frame (orthographic x, y); BGR."""
    h, w = frame_bgr.shape[:2]
    ext = float(np.max(np.linalg.norm(xyz[:, :2], axis=1)) + 1e-6)
    k = 0.42 * min(w, h) / max(ext, 0.12)
    cx, cy = w * 0.5, h * 0.5
    pts = []
    for i in range(NUM_LM):
        px = int(np.clip(cx + k * xyz[i, 0], 0, w - 1))
        py = int(np.clip(cy - k * xyz[i, 1], 0, h - 1))
        pts.append((px, py))
    color_line = (255, 255, 0)   # BGR: cyan
    color_pt = (0, 140, 255)     # BGR: orange
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], color_line, 2, lineType=cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        r = 5 if i == WRIST else 3
        cv2.circle(frame_bgr, (px, py), r, color_pt, -1, lineType=cv2.LINE_AA)


def _draw_skeleton_2d(
    ax,
    u: np.ndarray,
    v: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Plot one 2D orthographic projection of the 21 joints (u, v = coords per joint)."""
    ax.clear()
    for a, b in HAND_CONNECTIONS:
        ax.plot(
            [u[a], u[b]],
            [v[a], v[b]],
            color="tab:cyan",
            lw=1.8,
            zorder=1,
        )
    ax.scatter(u, v, c="tab:orange", s=18, zorder=2)
    ax.scatter([u[WRIST]], [v[WRIST]], c="tab:red", s=36, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def draw_csv_landmark_views(
    axes: tuple,
    xyz: np.ndarray,
    frame_index: int,
) -> None:
    """Four orthographic views of normalized 3D landmarks (wrist at origin).

    Axes order: front (x,−y), top (x,−z from +y), side (z,−y from +x),
    bottom (−x,z from −y).
    """
    ax_front, ax_top, ax_side, ax_bottom = axes
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    _draw_skeleton_2d(ax_front, x, -y, f"Front  fi={frame_index}", "x", "−y")
    _draw_skeleton_2d(ax_top, x, -z, "Top (+y →)", "x", "−z")
    _draw_skeleton_2d(ax_side, z, -y, "Side (+x →)", "z", "−y")
    _draw_skeleton_2d(ax_bottom, -x, z, "Bottom (−y →)", "−x", "z")


def clear_landmark_views(axes: tuple, message: str) -> None:
    for ax in axes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title(message, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id", help="Directory name under recordings/")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0 = real-time)")
    ap.add_argument("--no-overlay", action="store_true", help="Skip MediaPipe overlay (faster)")
    ap.add_argument(
        "--no-csv-on-video",
        action="store_true",
        help="With landmarks.csv, only draw the subplot below the video, not on the video frame",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH.mp4",
        help="Render the full layout to this MP4 file and exit (no interactive window)",
    )
    args = ap.parse_args()

    session_dir, emg, frame_times, video, landmarks = load_session(args.session_id)
    print(f"Loaded {session_dir.name}: "
          f"{len(emg)} EMG samples, {len(frame_times)} frames"
          + (f", landmarks.csv ({len(landmarks)} rows)" if landmarks is not None else ""))

    t_emg = emg["timestamp"].values.astype(np.float64)
    t_frame = frame_times["timestamp"].values.astype(np.float64)
    frame_idx_col = frame_times["frame_index"].values.astype(np.int64)

    t_start = min(t_emg[0], t_frame[0])
    t_end = max(t_emg[-1], t_frame[-1])
    duration = t_end - t_start

    # MediaPipe extractor (optional)
    extractor = None if args.no_overlay else HandPoseExtractor(running_mode="video")

    # --- Figure layout (video + optional landmark views only) ---
    fig = plt.figure(figsize=(9, 10 if landmarks is not None else 7))
    if args.output is None:
        _mgr = getattr(fig.canvas, "manager", None)
        if _mgr is not None and hasattr(_mgr, "set_window_title"):
            _mgr.set_window_title(f"Playback — {session_dir.name}")
    lm_axes: Optional[tuple] = None
    if landmarks is not None:
        gs = GridSpec(4, 1, figure=fig, hspace=0.4)
        ax_vid = fig.add_subplot(gs[0:3, 0])
        gs_lm = gs[3, 0].subgridspec(2, 2, wspace=0.32, hspace=0.5)
        ax_lm_front = fig.add_subplot(gs_lm[0, 0])
        ax_lm_top = fig.add_subplot(gs_lm[0, 1])
        ax_lm_side = fig.add_subplot(gs_lm[1, 0])
        ax_lm_bottom = fig.add_subplot(gs_lm[1, 1])
        lm_axes = (ax_lm_front, ax_lm_top, ax_lm_side, ax_lm_bottom)
    else:
        gs = GridSpec(1, 1, figure=fig)
        ax_vid = fig.add_subplot(gs[0, 0])

    ax_vid.axis("off")
    # Seed imshow with a correctly-shaped placeholder so aspect locks to the
    # actual video resolution, not the 2x2 default.
    vid_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    vid_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    vid_im = ax_vid.imshow(
        np.zeros((vid_h, vid_w, 3), dtype=np.uint8),
        aspect="equal",
    )
    ax_vid.set_aspect("equal", adjustable="box")
    vid_title = ax_vid.set_title(f"{session_dir.name}  t=0.00s")

    # --- Playback state ---
    paused = False
    wall_start = time.time()
    sim_offset = 0.0  # session-time offset when paused/unpaused
    pause_btn_holder: dict[str, Optional[Button]] = {"btn": None}

    def _sync_pause_button_label() -> None:
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
        _sync_pause_button_label()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key is None:
            return
        if event.key in ("q", "escape"):
            plt.close(fig)
        elif event.key in (" ", "p", "P", "enter", "return"):
            toggle_pause()

    if args.output is None:
        fig.canvas.mpl_connect("key_press_event", on_key)
        btn_ax = inset_axes(ax_vid, width="16%", height="7%", loc="lower right", borderpad=0.5)
        pause_btn = Button(btn_ax, "Pause", color="0.85", hovercolor="0.75")
        pause_btn.on_clicked(toggle_pause)
        pause_btn_holder["btn"] = pause_btn

    # Read video frames sequentially. Map frame_index -> (timestamp, bgr_frame).
    # Strategy: seek is expensive on mp4, so we iterate and skip as needed.
    current_frame_pos = 0
    current_frame_bgr = None

    def advance_to_frame(target_idx):
        nonlocal current_frame_pos, current_frame_bgr
        while current_frame_pos <= target_idx:
            ret, f = video.read()
            if not ret:
                return current_frame_bgr
            current_frame_bgr = f
            current_frame_pos += 1
        return current_frame_bgr

    # MediaPipe detect_for_video() requires strictly increasing timestamps; the
    # main loop runs many times per video frame (pause / GUI), so we only call
    # it when target_idx advances and we bump ts_ms if recording times glitch.
    last_mp_frame_idx = -1
    last_mp_ts_ms = -1
    last_mp_landmarks = None

    def one_step(t_now: float, t_sim_display: float) -> None:
        """Update video and landmark panels for one session time ``t_now``."""
        nonlocal last_mp_frame_idx, last_mp_ts_ms, last_mp_landmarks
        frame_mask = t_frame <= t_now
        if frame_mask.any():
            i = int(np.searchsorted(t_frame, t_now, side="right") - 1)
            target_idx = int(frame_idx_col[i])
            frame_bgr = advance_to_frame(target_idx)
            if frame_bgr is not None:
                disp = frame_bgr.copy()
                if landmarks is not None:
                    csv_row = lookup_landmark_row(landmarks, target_idx)
                    if csv_row is not None:
                        xyz_csv = row_to_landmarks_xyz(csv_row)
                        fi = int(csv_row["frame_index"])
                        if lm_axes is not None:
                            draw_csv_landmark_views(lm_axes, xyz_csv, fi)
                        if not args.no_csv_on_video:
                            draw_csv_landmarks_bgr(disp, xyz_csv)
                    elif lm_axes is not None:
                        clear_landmark_views(
                            lm_axes,
                            f"landmarks.csv — no row yet (video fi={target_idx})",
                        )
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
                vid_im.set_data(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
                vid_title.set_text(f"{session_dir.name}  t={t_sim_display:.2f}s / {duration:.1f}s")

    # --- Main loop (interactive) or render to file ---
    try:
        if args.output is not None:
            out_path = Path(args.output).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            dts = np.diff(t_frame.astype(np.float64))
            pos_dt = dts[np.isfinite(dts) & (dts > 0)]
            fps_raw = 1.0 / float(np.median(pos_dt)) if len(pos_dt) > 0 else 30.0
            fps = float(np.clip(fps_raw, 8.0, 60.0))
            writer: Optional[cv2.VideoWriter] = None
            nfr = len(t_frame)
            print(f"Rendering {nfr} frames → {out_path} @ {fps:.2f} fps …")
            for fi in range(nfr):
                t_now = float(t_frame[fi])
                one_step(t_now, t_now - t_start)
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                h, w, _ = buf.shape
                bgr = cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(out_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter failed to open: {out_path}")
                writer.write(bgr)
                if nfr <= 10 or (fi + 1) % max(1, nfr // 10) == 0 or fi == nfr - 1:
                    print(f"  frame {fi + 1}/{nfr}")
            if writer is not None:
                writer.release()
            print(f"Wrote {out_path}")
            plt.close(fig)
        else:
            plt.show(block=False)
            while plt.fignum_exists(fig.number):
                if paused:
                    t_sim = sim_offset
                else:
                    t_sim = (time.time() - wall_start) * args.speed

                t_now = t_start + t_sim
                if t_now > t_end:
                    break

                one_step(t_now, t_sim)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.08 if paused else 0.005)
    finally:
        video.release()
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    main()
