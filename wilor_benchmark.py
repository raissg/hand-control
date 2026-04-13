import os
import cv2
import torch
import numpy as np
import time
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

VIDEO_PATH = os.environ.get("WILOR_VIDEO", "cam_test.mov")


def pick_device() -> torch.device:
    """WILOR_DEVICE=cpu|cuda|mps|auto — auto prefers CUDA, else CPU (MPS optional, unstable with wilor_mini)."""
    pref = os.environ.get("WILOR_DEVICE", "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")


device = pick_device()
print(f"Device: {device}")
print("Loading WiLoR model...")
t0 = time.time()
pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32, verbose=False)
print(f"Model loaded in {time.time() - t0:.1f}s")

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames, {video_fps:.0f} FPS, {total_frames/video_fps:.1f}s duration")
print("-" * 60)

frame_times = []
hands_detected = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t_start = time.time()
    results = pipe.predict(frame_rgb)
    t_elapsed = time.time() - t_start
    frame_times.append(t_elapsed)

    n_hands = len(results)
    if n_hands > 0:
        hands_detected += 1

    frame_idx += 1
    if frame_idx % 10 == 0 or frame_idx == 1:
        avg_so_far = np.mean(frame_times)
        eta = avg_so_far * (total_frames - frame_idx)
        print(f"Frame {frame_idx}/{total_frames} | "
              f"this: {t_elapsed:.3f}s | "
              f"avg: {avg_so_far:.3f}s | "
              f"hands: {n_hands} | "
              f"ETA: {eta:.0f}s")

cap.release()

avg_time = np.mean(frame_times)
total_time = sum(frame_times)
effective_fps = len(frame_times) / total_time

print("-" * 60)
print(f"Processed {len(frame_times)} frames in {total_time:.1f}s")
print(f"Average per frame: {avg_time:.3f}s")
print(f"Effective FPS: {effective_fps:.2f}")
print(f"Hands detected in {hands_detected}/{len(frame_times)} frames ({100*hands_detected/len(frame_times):.0f}%)")
print(f"Realtime ratio: {total_time / (total_frames/video_fps):.1f}x (1.0 = realtime)")
