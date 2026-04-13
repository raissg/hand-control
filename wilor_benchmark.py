"""
Benchmark WiLoR inference (full repo, not wilor_mini): detector + ViTDetDataset + model.

Uses the same stack as Desktop WiLoR run_realtime.py / demo.py. Set WILOR_ROOT to your
WiLoR repo root (contains wilor/ and pretrained_models/). Default: wilor_realtime/WiLoR
next to this script. Video: cam_test.mov beside this script unless WILOR_VIDEO is set.
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# --- paths: resolve video before chdir(WILOR_ROOT) ---
_here = Path(__file__).resolve().parent
_default_root = _here / "wilor_realtime" / "WiLoR"
WILOR_ROOT = Path(os.environ.get("WILOR_ROOT", str(_default_root))).expanduser().resolve()
_raw_video = os.environ.get("WILOR_VIDEO", str(_here / "cam_test.mov"))
VIDEO_PATH = str(Path(_raw_video).expanduser().resolve())

if not WILOR_ROOT.is_dir():
    print(f"WILOR_ROOT is not a directory: {WILOR_ROOT}")
    print("Set WILOR_ROOT to your WiLoR repo root (contains wilor/ and pretrained_models/).")
    sys.exit(1)

sys.path.insert(0, str(WILOR_ROOT))
_prev_cwd = os.getcwd()
os.chdir(WILOR_ROOT)

# PyTorch 2.4+ defaults weights_only=True; WiLoR checkpoints need full load.
torch.serialization.add_safe_globals([])
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from wilor.models import load_wilor  # noqa: E402
from wilor.utils import recursive_to  # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def pick_device() -> torch.device:
    """WILOR_DEVICE=cpu|cuda|mps|auto — auto prefers CUDA, then MPS, else CPU."""
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
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def run_frame(detector, model, model_cfg, device, frame_bgr: np.ndarray) -> int:
    """Detector + WiLoR forward only (no rendering). Returns number of hands processed."""
    detections = detector(frame_bgr, conf=0.3, verbose=False)[0]
    bboxes = []
    is_right = []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())

    if len(bboxes) == 0:
        return 0

    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    dataset = ViTDetDataset(model_cfg, frame_bgr, boxes, right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    n_hands = 0
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            model(batch)
        n_hands += int(batch["img"].shape[0])
    return n_hands


def main() -> None:
    device = pick_device()
    print(f"Device: {device}")
    print(f"WiLoR root: {WILOR_ROOT}")
    print("Loading WiLoR model...")
    t0 = time.time()
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    model = model.to(device)
    model.eval()
    print(f"WiLoR weights loaded in {time.time() - t0:.1f}s")

    print("Loading hand detector...")
    t1 = time.time()
    detector = YOLO("./pretrained_models/detector.pt")
    detector.to(device)
    print(f"Detector loaded in {time.time() - t1:.1f}s")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        os.chdir(_prev_cwd)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {VIDEO_PATH}")
    print(f"Frames: {total_frames}, {video_fps:.0f} FPS, {total_frames / video_fps:.1f}s duration")
    print("-" * 60)

    frame_times = []
    hands_detected = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()
        n_hands = run_frame(detector, model, model_cfg, device, frame)
        t_elapsed = time.time() - t_start
        frame_times.append(t_elapsed)

        if n_hands > 0:
            hands_detected += 1

        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == 1:
            avg_so_far = float(np.mean(frame_times))
            eta = avg_so_far * (total_frames - frame_idx)
            print(
                f"Frame {frame_idx}/{total_frames} | "
                f"this: {t_elapsed:.3f}s | "
                f"avg: {avg_so_far:.3f}s | "
                f"hands: {n_hands} | "
                f"ETA: {eta:.0f}s"
            )

    cap.release()
    os.chdir(_prev_cwd)

    if not frame_times:
        print("No frames processed.")
        return

    avg_time = float(np.mean(frame_times))
    total_time = sum(frame_times)
    effective_fps = len(frame_times) / total_time

    print("-" * 60)
    print(f"Processed {len(frame_times)} frames in {total_time:.1f}s")
    print(f"Average per frame: {avg_time:.3f}s")
    print(f"Effective FPS: {effective_fps:.2f}")
    print(
        f"Hands detected in {hands_detected}/{len(frame_times)} frames "
        f"({100 * hands_detected / len(frame_times):.0f}%)"
    )
    print(f"Realtime ratio: {total_time / (total_frames / video_fps):.1f}x (1.0 = realtime)")


if __name__ == "__main__":
    main()
