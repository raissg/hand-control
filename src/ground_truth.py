"""
Ground truth hand pose extraction from camera frames using MediaPipe Hands.

Outputs 21 normalized 3D landmarks per frame. Landmarks are:
  - Translated so wrist = origin
  - Scaled so wrist-to-middle-MCP distance = 1.0

This makes the representation invariant to hand distance from camera
and position in frame. The regression target is 20 x 3 = 60 values
(excluding wrist which is always [0,0,0]).

Usage:
    from src.ground_truth import HandPoseExtractor

    extractor = HandPoseExtractor()
    landmarks = extractor.extract(frame_rgb)
    # landmarks.landmarks: np.ndarray (21, 3) normalized
    # landmarks.landmarks[0] = [0, 0, 0] (wrist, always origin)

CLI:
    python src/ground_truth.py                    # live webcam
    python src/ground_truth.py --video path.mov   # from video file
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Task API imports
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe landmark indices
WRIST = 0
MIDDLE_MCP = 9
NUM_LANDMARKS = 21

# Default model path (relative to project root)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "hand_landmarker.task")

# HandLandmarkerOptions defaults — single source for ground-truth extraction, live demos, etc.
MP_HAND_NUM_HANDS = 1
MP_HAND_MIN_DETECTION_CONFIDENCE = 0.5
MP_HAND_MIN_TRACKING_CONFIDENCE = 0.5
# HandPoseExtractor sets min_hand_presence_confidence = min_detection_confidence (API parity).

# Hand connections for drawing (same as mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]


@dataclass(slots=True)
class HandPose:
    """Single frame of hand pose data."""
    landmarks: np.ndarray        # (21, 3) normalized landmarks
    landmarks_raw: np.ndarray    # (21, 3) world xyz if available, else image-plane xyz
    handedness: str              # "Left" or "Right"
    scale: float                 # wrist-to-middle-MCP distance used for normalization
    image_uv: np.ndarray        # (21, 2) normalized image x,y in [0,1] for overlay


class HandPoseExtractor:
    """Extract normalized 3D hand landmarks from RGB frames using MediaPipe.

    Parameters
    ----------
    model_path : str
        Path to hand_landmarker.task model file.
    max_hands : int
        Maximum number of hands to detect.
    min_detection_confidence : float
        Minimum confidence for hand detection.
    min_tracking_confidence : float
        Minimum confidence for landmark tracking.
    running_mode : str
        "image" for single frames, "video" for sequential frames.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_hands: int = MP_HAND_NUM_HANDS,
        min_detection_confidence: float = MP_HAND_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MP_HAND_MIN_TRACKING_CONFIDENCE,
        running_mode: str = "video",
    ):
        mode = (
            VisionRunningMode.VIDEO
            if running_mode == "video"
            else VisionRunningMode.IMAGE
        )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._mode = mode
        self._frame_ts = 0  # monotonic timestamp for video mode (ms)

    def extract(self, frame_rgb: np.ndarray) -> Optional[HandPose]:
        """Extract hand pose from an RGB frame.

        Returns the first detected hand, or None if no hand found.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._mode == VisionRunningMode.VIDEO:
            result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
            self._frame_ts += 33  # ~30 FPS
        else:
            result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        return self._parse_hand(result, 0)

    def extract_all(self, frame_rgb: np.ndarray) -> list[HandPose]:
        """Extract all detected hands from an RGB frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._mode == VisionRunningMode.VIDEO:
            result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
            self._frame_ts += 33
        else:
            result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        return [
            self._parse_hand(result, i)
            for i in range(len(result.hand_landmarks))
        ]

    def _parse_hand(self, result, idx: int) -> HandPose:
        """Parse a single hand from MediaPipe result."""
        landmarks = result.hand_landmarks[idx]
        handedness = result.handedness[idx][0].category_name

        # Use world landmarks (metric 3D) if available, else normalized
        if result.hand_world_landmarks:
            world = result.hand_world_landmarks[idx]
            raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in world],
                dtype=np.float32,
            )
        else:
            raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float32,
            )

        normalized, scale = _normalize_landmarks(raw)

        image_uv = np.array(
            [[lm.x, lm.y] for lm in landmarks],
            dtype=np.float32,
        )

        return HandPose(
            landmarks=normalized,
            landmarks_raw=raw,
            handedness=handedness,
            scale=scale,
            image_uv=image_uv,
        )

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _normalize_landmarks(raw: np.ndarray) -> tuple[np.ndarray, float]:
    """Translate to wrist origin and scale by wrist-to-middle-MCP distance.

    Returns (normalized_landmarks, scale_factor).
    """
    centered = raw - raw[WRIST]

    scale = float(np.linalg.norm(centered[MIDDLE_MCP]))
    if scale < 1e-6:
        return np.zeros_like(centered), 0.0

    normalized = centered / scale
    return normalized, scale


def landmarks_to_target(landmarks: np.ndarray) -> np.ndarray:
    """Convert (21, 3) normalized landmarks to (60,) regression target.

    Drops the wrist (always [0,0,0]) and flattens.
    """
    return landmarks[1:].flatten()


def target_to_landmarks(target: np.ndarray) -> np.ndarray:
    """Convert (60,) regression target back to (21, 3) landmarks.

    Prepends the wrist at origin.
    """
    joints = target.reshape(20, 3)
    wrist = np.zeros((1, 3), dtype=joints.dtype)
    return np.concatenate([wrist, joints], axis=0)


# ---------------------------------------------------------------------------
# CLI: visualize landmarks from webcam or video
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Hand pose ground truth extraction")
    parser.add_argument("--video", type=str, default=None, help="Video file (default: webcam)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to .task model")
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open: {source}")
        exit(1)

    extractor = HandPoseExtractor(model_path=args.model)
    frame_count = 0
    t_start = time.time()

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose = extractor.extract(frame_rgb)

        if pose is not None:
            h, w = frame.shape[:2]

            # Draw skeleton using raw image-space landmarks
            # (for world landmarks, we project; for normalized, scale by image)
            # Re-detect to get image-space landmarks for drawing
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            draw_result = extractor._landmarker.detect_for_video(
                mp_image, extractor._frame_ts
            )
            extractor._frame_ts += 33

            if draw_result.hand_landmarks:
                img_lms = draw_result.hand_landmarks[0]
                for i, lm in enumerate(img_lms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                    if i == WRIST:
                        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), 2)

                for a, b in HAND_CONNECTIONS:
                    ax, ay = int(img_lms[a].x * w), int(img_lms[a].y * h)
                    bx, by = int(img_lms[b].x * w), int(img_lms[b].y * h)
                    cv2.line(frame, (ax, ay), (bx, by), (0, 200, 200), 2)

            target = landmarks_to_target(pose.landmarks)
            cv2.putText(
                frame,
                f"hand={pose.handedness} scale={pose.scale:.4f} target={target.shape}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        frame_count += 1
        fps = frame_count / (time.time() - t_start + 1e-6)
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )

        cv2.imshow("Hand Pose Ground Truth", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"Processed {frame_count} frames at {fps:.1f} FPS")
