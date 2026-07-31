"""Live angle visualizer — HandPose landmarks + full 3D angle math for overlays.

Uses HandPoseExtractor (same landmark normalization as recorder). Angle overlay
here uses **true 3D angles** between bone vectors (no z-zeroing): flexion/extension
angles are ``arccos(u·v)`` on unit 3D segments; MCP/CMC **ab/adduction** angles are
signed angles between projected vectors in the plane perpendicular to an estimated
palm normal (index MCP × pinky MCP). Landmark z still comes from the vision model —
these are geometrically ``3D`` but not necessarily metric-clinical angles.

Run:
    python3 -m src.angle_visualizer
    python3 -m src.angle_visualizer --camera 1   # if webcam is not index 0
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from src.ground_truth import HandPoseExtractor, HAND_CONNECTIONS
from src.landmarks_to_angles import ANGLE_NAMES

_NUM_ANGLES = 20
_WRIST = 0
_THUMB = (1, 2, 3, 4)
_INDEX = (5, 6, 7, 8)
_MIDDLE = (9, 10, 11, 12)
_RING = (13, 14, 15, 16)
_PINKY = (17, 18, 19, 20)


def _unit3(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return np.asarray(v, dtype=np.float64) / (n + eps)


def _angle_fe_3d(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    """Unsigned angle in [0, π] between 3D directions: arccos((a·b) / (|a||b|))."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    denom = (na) * (nb)
    cos_th = float(np.dot(a, b)) / denom
    cos_th = float(np.clip(cos_th, -1.0 + eps, 1.0 - eps))
    return float(np.arccos(cos_th))


def _palm_normal(lm: np.ndarray) -> np.ndarray:
    """Rough palm normal at wrist: (index MCP − wrist) × (pinky MCP − wrist)."""
    vi = lm[_INDEX[0]] - lm[_WRIST]
    vp = lm[_PINKY[0]] - lm[_WRIST]
    n = np.cross(vi, vp)
    if np.linalg.norm(n) < 1e-9:
        vm = lm[_MIDDLE[0]] - lm[_WRIST]
        n = np.cross(vi, vm)
    return _unit3(n)


def _signed_angle_in_plane(
    v1: np.ndarray,
    v2: np.ndarray,
    plane_normal: np.ndarray,
    eps: float = 1e-9,
) -> float:
    """Signed angle from v1 to v2 after projecting both onto plane ⊥ plane_normal."""
    n = _unit3(plane_normal, eps)
    p1 = v1 - n * np.dot(v1, n)
    p2 = v2 - n * np.dot(v2, n)
    u1n = np.linalg.norm(p1)
    u2n = np.linalg.norm(p2)
    if u1n < eps or u2n < eps:
        return 0.0
    u1, u2 = p1 / u1n, p2 / u2n
    sin_th = float(np.dot(n, np.cross(u1, u2)))
    cos_th = float(np.clip(np.dot(u1, u2), -1.0 + eps, 1.0 - eps))
    return float(np.arctan2(sin_th, cos_th))


def landmarks_to_angles_3d(lm: np.ndarray) -> np.ndarray:
    """Same 20-angle schema as ``landmarks_to_angles``, but **full 3D** geometry.

    - FE chains: angle between successive bone vectors in ℝ³ (arccos).
    - AA at CMC / MCP: signed angle in plane ⊥ palm normal (index×pinky at wrist).

    Args:
        lm: (21, 3) wrist-centered normalized landmarks (from HandPose).
    """
    assert lm.shape == (21, 3), f"expected (21,3), got {lm.shape}"
    lm = lm.astype(np.float64, copy=True)
    out = np.zeros(_NUM_ANGLES, dtype=np.float64)

    palm_n = _palm_normal(lm)

    thumb_mc = lm[_THUMB[0]] - lm[_WRIST]
    cmc_bone = lm[_THUMB[1]] - lm[_THUMB[0]]
    mcp_bone = lm[_THUMB[2]] - lm[_THUMB[1]]
    ip_bone = lm[_THUMB[3]] - lm[_THUMB[2]]

    out[0] = _angle_fe_3d(thumb_mc, cmc_bone)
    out[1] = _signed_angle_in_plane(thumb_mc, cmc_bone, palm_n)
    out[2] = _angle_fe_3d(cmc_bone, mcp_bone)
    out[3] = _angle_fe_3d(mcp_bone, ip_bone)

    fingers = (
        (_INDEX, 4, 5, 6, 7),
        (_MIDDLE, 8, 9, 10, 11),
        (_RING, 12, 13, 14, 15),
        (_PINKY, 16, 17, 18, 19),
    )

    for F, i_aa, i_fe, i_pip, i_dip in fingers:
        mcp, pip, dip, tip = F
        metacarpal = lm[mcp] - lm[_WRIST]
        proximal = lm[pip] - lm[mcp]
        middle_bone = lm[dip] - lm[pip]
        distal_bone = lm[tip] - lm[dip]

        out[i_aa] = _signed_angle_in_plane(metacarpal, proximal, palm_n)
        out[i_fe] = _angle_fe_3d(metacarpal, proximal)
        out[i_pip] = _angle_fe_3d(proximal, middle_bone)
        out[i_dip] = _angle_fe_3d(middle_bone, distal_bone)

    return out.astype(np.float32)

# Which angle index to show at each MediaPipe landmark index.
# Format: { mediapipe_lm_index: [(angle_idx, short_label), ...] }
JOINT_LABELS = {
    # Thumb
    1:  [(0, "CFE"), (1, "CAA")],   # CMC → CMC_FE, CMC_AA
    2:  [(2, "MFE")],               # MCP → MCP_FE
    3:  [(3, "IFE")],               # IP  → IP_FE
    # Index
    5:  [(4, "AA"),  (5, "FE")],    # MCP
    6:  [(6, "PIP")],
    7:  [(7, "DIP")],
    # Middle
    9:  [(8, "AA"),  (9, "FE")],
    10: [(10, "PIP")],
    11: [(11, "DIP")],
    # Ring
    13: [(12, "AA"), (13, "FE")],
    14: [(14, "PIP")],
    15: [(15, "DIP")],
    # Pinky
    17: [(16, "AA"), (17, "FE")],
    18: [(18, "PIP")],
    19: [(19, "DIP")],
}


def draw_hand(frame, uv_21: np.ndarray, connections, color=(0, 200, 0)):
    """uv_21: (21, 2) normalized image x,y in [0,1]."""
    h, w = frame.shape[:2]
    pts = [(int(uv_21[i, 0] * w), int(uv_21[i, 1] * h)) for i in range(21)]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, color, 1)


def draw_angles(frame, uv_21: np.ndarray, angles_deg):
    """angles_deg: values already in degrees (after ``np.degrees``)."""
    h, w = frame.shape[:2]
    for lm_idx, entries in JOINT_LABELS.items():
        px = int(uv_21[lm_idx, 0] * w)
        py = int(uv_21[lm_idx, 1] * h)

        parts = []
        for angle_idx, label in entries:
            val = angles_deg[angle_idx]
            parts.append(f"{label}:{val:+.0f}°")
        text = "  ".join(parts)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, thick = 0.38, 1
        (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
        ox, oy = px + 6, py - 4
        if ox + tw > w:
            ox = px - tw - 6
        if oy - th < 0:
            oy = py + th + 4
        cv2.rectangle(frame, (ox - 2, oy - th - 2), (ox + tw + 2, oy + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (ox, oy), font, fs, (0, 255, 200), thick,
                    cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true", default=True)
    parser.add_argument("--with-pred", action="store_true",
                        help="Also run the 5-angle EMG model; press 'r' to log GT+Pred.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic EMG (testing without MindRove).")
    parser.add_argument("--model", default="models/neuropose_angles5_best.pt")
    parser.add_argument("--config",
                        default="emg2pose/config/network/neuropose_angles5.yaml")
    parser.add_argument("--dataset", default="data/dataset_angles_no_imu.hdf5")
    parser.add_argument("--out-dir", default="recordings/angle_logs")
    args = parser.parse_args()

    # Optional EMG predictor (only imported when requested, so visualizer still runs
    # without torch / MindRove installed).
    predictor = stream = None
    KEEP_INDICES = [0, 5, 9, 13, 17]
    KEEP_NAMES = [ANGLE_NAMES[i] for i in KEEP_INDICES]
    if args.with_pred:
        import torch  # noqa: F401
        from src.acquire import EMGStream
        from src.live_compare5 import AngleEMGPredictor

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ab = lambda p: p if os.path.isabs(p) else os.path.join(root, p)
        device = ("cuda" if __import__("torch").cuda.is_available()
                  else "mps" if __import__("torch").backends.mps.is_available()
                  else "cpu")
        predictor = AngleEMGPredictor(ab(args.model), ab(args.config),
                                       ab(args.dataset), device)
        stream = EMGStream(synthetic=args.synthetic)
        stream.start(); predictor.start(stream)
        print(f"EMG predictor running on {device}.")

    extractor = HandPoseExtractor(running_mode="video")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        if predictor is not None: predictor.stop(); stream.stop()
        return

    print("Controls: ESC/Q=quit,  R=toggle record GT vs Pred")

    # Recording state
    recording = False
    rec_rows = []
    rec_path = None
    t_start = time.time()

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = extractor.extract(rgb)

            pred5_deg = None
            if predictor is not None:
                pred5_deg = np.degrees(predictor.predict())  # (5,)

            if pose is not None:
                angles_deg = np.degrees(landmarks_to_angles_3d(pose.landmarks))
                draw_hand(frame, pose.image_uv, HAND_CONNECTIONS)
                draw_angles(frame, pose.image_uv, angles_deg)

                # Overlay 5 predicted angles in red next to their joint lm
                if pred5_deg is not None:
                    h, w = frame.shape[:2]
                    lm_at = [1, 5, 9, 13, 17]
                    for k, lmi in enumerate(lm_at):
                        px = int(pose.image_uv[lmi, 0] * w)
                        py = int(pose.image_uv[lmi, 1] * h)
                        txt = f"P{pred5_deg[k]:+.0f}\u00b0"
                        cv2.putText(frame, txt, (px + 6, py + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                    (60, 60, 255), 1, cv2.LINE_AA)

                # Log a row while recording
                if recording and pred5_deg is not None:
                    gt5 = angles_deg[KEEP_INDICES]
                    rec_rows.append(
                        [f"{time.time() - t_start:.4f}"]
                        + [f"{v:.3f}" for v in gt5]
                        + [f"{v:.3f}" for v in pred5_deg]
                    )

                if frame_idx % 30 == 0:
                    print("\n--- Angles (degrees) ---")
                    for i, name in enumerate(ANGLE_NAMES):
                        tag = "*" if i in KEEP_INDICES else " "
                        extra = ""
                        if pred5_deg is not None and i in KEEP_INDICES:
                            pi = KEEP_INDICES.index(i)
                            extra = f"   pred {pred5_deg[pi]:+6.1f}"
                        print(f" {tag} {name:<20}: {angles_deg[i]:+6.1f}{extra}")
            else:
                cv2.putText(frame, "No hand detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Status line
            fh = frame.shape[0]
            status = "Angles in degrees"
            if predictor is not None:
                status += "  |  green=GT  red=Pred (5 palm-to-finger FE)"
            cv2.putText(frame, status, (10, fh - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 255, 200), 1, cv2.LINE_AA)
            if recording:
                cv2.circle(frame, (20, fh - 55), 7, (0, 0, 255), -1)
                cv2.putText(frame, f"REC  {len(rec_rows)} rows",
                            (34, fh - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)

            frame_idx += 1
            cv2.imshow("Angle visualizer — R record, ESC quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('r'):
                if predictor is None:
                    print("Recording requires --with-pred.")
                elif not recording:
                    os.makedirs(args.out_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = os.path.join(args.out_dir, f"gt_vs_pred_{ts}.csv")
                    rec_rows = []
                    t_start = time.time()
                    recording = True
                    print(f"[REC] start -> {rec_path}")
                else:
                    recording = False
                    header = (["t_s"]
                              + [f"gt_{n}" for n in KEEP_NAMES]
                              + [f"pred_{n}" for n in KEEP_NAMES])
                    with open(rec_path, "w", newline="") as f:
                        w = csv.writer(f); w.writerow(header); w.writerows(rec_rows)
                    print(f"[REC] stop, {len(rec_rows)} rows -> {rec_path}")
    finally:
        # Auto-save if still recording on exit
        if recording and rec_path is not None and rec_rows:
            header = (["t_s"]
                      + [f"gt_{n}" for n in KEEP_NAMES]
                      + [f"pred_{n}" for n in KEEP_NAMES])
            with open(rec_path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(header); w.writerows(rec_rows)
            print(f"[REC] auto-saved on exit, {len(rec_rows)} rows -> {rec_path}")
        cap.release()
        extractor.close()
        if predictor is not None:
            predictor.stop(); stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
