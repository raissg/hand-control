# src/mcp_angles_live.py
"""Show 5 MCP flexion angles live from webcam via MediaPipe Hand Landmarker (tasks API).
The two bone segments that form each angle are drawn in a bright per-finger color; the
MCP (angle vertex) is a filled dot. The rest of the skeleton is dim for context.

Usage:
    python -m src.mcp_angles_live
    python -m src.mcp_angles_live --camera 1
    python -m src.mcp_angles_live --compare   # side-by-side + L_* and R_* slider sets (6 trackbars)

Requires ``hand_landmarker.task`` at the project root (same as ``src/ground_truth.py``).
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Repo root on path for optional shared constants
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ground_truth import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    HAND_CONNECTIONS,
    MP_HAND_MIN_DETECTION_CONFIDENCE,
    MP_HAND_MIN_TRACKING_CONFIDENCE,
    MP_HAND_NUM_HANDS,
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Video timestep increment — match HandPoseExtractor.extract() (~30 FPS).
FRAME_DELTA_MS = 33

# OpenCV trackbars use 0..100 → confidence 0.00..1.00
TRACK_MAX = 100


# Landmark triplets: (name, parent, joint, child) — angle at `joint` for MCP flexion.
# Thumb uses CMC–MCP–IP (1–2–3). Others: wrist–MCP–PIP.
FINGERS = [
    ("thumb", 1, 2, 3),
    ("index", 0, 5, 6),
    ("middle", 0, 9, 10),
    ("ring", 0, 13, 14),
    ("pinky", 0, 17, 18),
]

# BGR — one color per row; edges (parent→joint) & (joint→child) drawn in this color.
HIGHLIGHT_BGR = [
    (0, 165, 255),   # thumb — orange
    (255, 144, 30),  # index — light blue
    (60, 255, 255),  # middle — yellow
    (230, 80, 230),  # ring — magenta/pink
    (80, 255, 140),  # pinky — spring green
]


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b between vectors (a-b) and (c-b), degrees."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cos = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def flexion(lm: np.ndarray, p: int, j: int, c: int) -> float:
    """180° when straight; larger when curled (180 - interior angle)."""
    return 180.0 - angle_deg(lm[p], lm[j], lm[c])


def draw_skeleton(
    frame: np.ndarray,
    hand_lm_norm,
    color=(40, 70, 40),
    thickness: int = 1,
) -> None:
    """Full hand wireframe — dimmer so MCP angle bones stand out."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm_norm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, thickness)


def draw_mcp_angle_bones(
    frame: np.ndarray,
    hand_lm_norm,
    thickness: int = 4,
    vertex_radius: int = 7,
) -> None:
    """Bold lines for the two bones whose meeting angle defines each MCP flexion."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm_norm]

    for (_, p, j, c), col in zip(FINGERS, HIGHLIGHT_BGR):
        cv2.line(frame, pts[p], pts[j], col, thickness)
        cv2.line(frame, pts[j], pts[c], col, thickness)
        cv2.circle(frame, pts[j], vertex_radius, col, -1)
        cv2.circle(frame, pts[j], vertex_radius + 2, col, 1)


def landmarks_3d_for_angles(result, idx: int = 0) -> np.ndarray:
    """Metric-ish 3D for angles: world landmarks if present, else normalized xyz."""
    if result.hand_world_landmarks:
        wl = result.hand_world_landmarks[idx]
        return np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float64)
    hl = result.hand_landmarks[idx]
    return np.array([[p.x, p.y, p.z] for p in hl], dtype=np.float64)


def draw_mcp_overlay(
    frame_bgr: np.ndarray,
    result,
    *,
    show_missing: bool = False,
    text_y0: int = 30,
) -> None:
    """Draw skeleton, MCP highlights, and angle labels when a hand is present."""
    if not result.hand_landmarks:
        if show_missing:
            h, w = frame_bgr.shape[:2]
            cv2.putText(
                frame_bgr,
                "no hand (landmarks empty)",
                (w // 2 - 180, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 50, 255),
                2,
            )
        return
    hl = result.hand_landmarks[0]
    draw_skeleton(frame_bgr, hl)
    draw_mcp_angle_bones(frame_bgr, hl)
    lm = landmarks_3d_for_angles(result, 0)
    for i, (name, p, j, c) in enumerate(FINGERS):
        deg = flexion(lm, p, j, c)
        cv2.putText(
            frame_bgr,
            f"{name:6s} {deg:5.1f} deg",
            (10, text_y0 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            HIGHLIGHT_BGR[i],
            2,
        )


def build_hand_options(
    model_path_str: str,
    *,
    min_detection: float,
    min_presence: float,
    min_tracking: float,
) -> HandLandmarkerOptions:
    """Same shape as HandPoseExtractor — presence is independent (sliders can diverge)."""
    return HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path_str),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MP_HAND_NUM_HANDS,
        min_hand_detection_confidence=min_detection,
        min_hand_presence_confidence=min_presence,
        min_tracking_confidence=min_tracking,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live MCP flexion angles (MediaPipe)")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to hand_landmarker.task (default: same as HandPoseExtractor)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Side-by-side A/B: L_det/L_pres/L_trk drive left panel, R_* drive right panel",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Missing model: {model_path}\nDownload or copy hand_landmarker.task to project root.")
        sys.exit(1)

    if args.compare:
        main_compare(str(model_path), args.camera)
    else:
        main_single(str(model_path), args.camera)


def main_single(model_path_str: str, camera: int) -> None:
    options = build_hand_options(
        model_path_str,
        min_detection=MP_HAND_MIN_DETECTION_CONFIDENCE,
        min_presence=MP_HAND_MIN_DETECTION_CONFIDENCE,
        min_tracking=MP_HAND_MIN_TRACKING_CONFIDENCE,
    )
    landmarker = HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Cannot open camera {camera}")
        landmarker.close()
        sys.exit(1)

    frame_ts_ms = 0
    print("MCP flexion live — ESC to quit")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)
            frame_ts_ms += FRAME_DELTA_MS

            draw_mcp_overlay(frame, result)

            cv2.imshow("MCP flexion (ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


def main_compare(model_path_str: str, camera: int) -> None:
    """Two independent HandLandmarkers; each side has its own det/pres/trk sliders."""
    win = "MCP compare — L_* = left | R_* = right — ESC quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def initial_pos(v: float) -> int:
        return int(round(float(np.clip(v, 0.0, 1.0)) * TRACK_MAX))

    # Both sides start at HandPoseExtractor defaults; change either set to A/B compare.
    d0 = initial_pos(MP_HAND_MIN_DETECTION_CONFIDENCE)
    t0 = initial_pos(MP_HAND_MIN_TRACKING_CONFIDENCE)
    for name, pos in (
        ("L_det", d0),
        ("L_pres", d0),
        ("L_trk", t0),
        ("R_det", d0),
        ("R_pres", d0),
        ("R_trk", t0),
    ):
        cv2.createTrackbar(name, win, pos, TRACK_MAX, lambda *_: None)

    def read_lr_confs() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        l = (
            cv2.getTrackbarPos("L_det", win) / TRACK_MAX,
            cv2.getTrackbarPos("L_pres", win) / TRACK_MAX,
            cv2.getTrackbarPos("L_trk", win) / TRACK_MAX,
        )
        r = (
            cv2.getTrackbarPos("R_det", win) / TRACK_MAX,
            cv2.getTrackbarPos("R_pres", win) / TRACK_MAX,
            cv2.getTrackbarPos("R_trk", win) / TRACK_MAX,
        )
        return l, r

    left_confs, right_confs = read_lr_confs()

    left_lm = HandLandmarker.create_from_options(
        build_hand_options(
            model_path_str,
            min_detection=left_confs[0],
            min_presence=left_confs[1],
            min_tracking=left_confs[2],
        )
    )
    right_lm = HandLandmarker.create_from_options(
        build_hand_options(
            model_path_str,
            min_detection=right_confs[0],
            min_presence=right_confs[1],
            min_tracking=right_confs[2],
        )
    )

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Cannot open camera {camera}")
        left_lm.close()
        right_lm.close()
        sys.exit(1)

    ts_left = 0
    ts_right = 0
    print(
        "6 trackbars: L_det L_pres L_trk (left view) | R_det R_pres R_trk (right view). ESC quits."
    )
    print(
        "Note: min_tracking_confidence mostly affects frames where tracking is marginal "
        "(fast motion, blur). Steady hands often keep internal scores high — try L_trk 0 vs "
        "100 while waving fast; watch 'hand%' and red 'no hand' flashes."
    )

    recent_n = 72  # ~2.5s at 30fps — rolling hand detection rate
    left_hits: deque[int] = deque(maxlen=recent_n)
    right_hits: deque[int] = deque(maxlen=recent_n)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            new_left, new_right = read_lr_confs()
            if new_left != left_confs:
                left_lm.close()
                left_lm = HandLandmarker.create_from_options(
                    build_hand_options(
                        model_path_str,
                        min_detection=new_left[0],
                        min_presence=new_left[1],
                        min_tracking=new_left[2],
                    )
                )
                left_confs = new_left
                ts_left = 0
                left_hits.clear()

            if new_right != right_confs:
                right_lm.close()
                right_lm = HandLandmarker.create_from_options(
                    build_hand_options(
                        model_path_str,
                        min_detection=new_right[0],
                        min_presence=new_right[1],
                        min_tracking=new_right[2],
                    )
                )
                right_confs = new_right
                ts_right = 0
                right_hits.clear()

            res_left = left_lm.detect_for_video(mp_image, ts_left)
            res_right = right_lm.detect_for_video(mp_image, ts_right)
            ts_left += FRAME_DELTA_MS
            ts_right += FRAME_DELTA_MS

            left_hits.append(1 if res_left.hand_landmarks else 0)
            right_hits.append(1 if res_right.hand_landmarks else 0)
            pct_l = 100.0 * sum(left_hits) / len(left_hits) if left_hits else 0.0
            pct_r = 100.0 * sum(right_hits) / len(right_hits) if right_hits else 0.0

            panel_l = frame.copy()
            panel_r = frame.copy()

            cv2.putText(
                panel_l,
                f"hand% last {len(left_hits)}f: {pct_l:.0f}%",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 255, 200),
                2,
            )
            cv2.putText(
                panel_r,
                f"hand% last {len(right_hits)}f: {pct_r:.0f}%",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 200, 255),
                2,
            )

            cv2.putText(
                panel_l,
                "LEFT  (L_det / L_pres / L_trk)",
                (10, panel_l.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 255, 200),
                2,
            )
            cv2.putText(
                panel_l,
                f"det={left_confs[0]:.2f} pres={left_confs[1]:.2f} trk={left_confs[2]:.2f}",
                (10, panel_l.shape[0] - 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 220, 180),
                1,
            )

            cv2.putText(
                panel_r,
                "RIGHT (R_det / R_pres / R_trk)",
                (10, panel_r.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 255),
                2,
            )
            cv2.putText(
                panel_r,
                f"det={right_confs[0]:.2f} pres={right_confs[1]:.2f} trk={right_confs[2]:.2f}",
                (10, panel_r.shape[0] - 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 220),
                1,
            )

            draw_mcp_overlay(panel_l, res_left, show_missing=True, text_y0=56)
            draw_mcp_overlay(panel_r, res_right, show_missing=True, text_y0=56)

            combined = np.hstack([panel_l, panel_r])
            cv2.imshow(win, combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        left_lm.close()
        right_lm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
