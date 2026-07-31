"""Live EMG-predicted angles vs. MediaPipe ground-truth angles.

Styled like ``src/angle_visualizer.py``: single webcam pane with the hand
skeleton. At each joint we overlay the GT angle in **green** and the
predicted error (|pred − gt|) in **red**, right next to it.

Usage:
    python -m src.live_compare                          # real MindRove + cam 0
    python -m src.live_compare --synthetic              # synthetic EMG
    python -m src.live_compare --camera 1
"""

from __future__ import annotations

import argparse
import collections
import os
import threading
import time

import cv2
import h5py
import numpy as np
import torch

from src.acquire import EMGStream
from src.convert_to_hdf5 import filter_emg
from src.ground_truth import HandPoseExtractor, HAND_CONNECTIONS
from src.angle_visualizer import landmarks_to_angles_3d
from src.landmarks_to_angles import ANGLE_NAMES
from src.train import get_model


NUM_EMG = 8
WINDOW_LENGTH = 1000
SAMPLE_RATE = 500

# MediaPipe lm idx -> list of (angle_idx, short_label) — same as angle_visualizer
JOINT_LABELS = {
    1:  [(0, "CFE"), (1, "CAA")],
    2:  [(2, "MFE")],
    3:  [(3, "IFE")],
    5:  [(4, "AA"),  (5, "FE")],
    6:  [(6, "PIP")], 7: [(7, "DIP")],
    9:  [(8, "AA"),  (9, "FE")],
    10: [(10, "PIP")], 11: [(11, "DIP")],
    13: [(12, "AA"), (13, "FE")],
    14: [(14, "PIP")], 15: [(15, "DIP")],
    17: [(16, "AA"), (17, "FE")],
    18: [(18, "PIP")], 19: [(19, "DIP")],
}


class AngleEMGPredictor:
    """Rolling EMG buffer + angles model. Thread-safe push from stream thread."""
    def __init__(self, model_path, config_path, dataset_path, device,
                 ema_alpha=0.5):
        self.device = torch.device(device)
        self.ema_alpha = ema_alpha

        self.model = get_model(str(config_path)).to(self.device)
        state = torch.load(str(model_path), map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

        with h5py.File(str(dataset_path), "r") as f:
            self.input_mean = np.asarray(f.attrs["input_mean"], dtype=np.float32)
            self.input_std = np.asarray(f.attrs["input_std"], dtype=np.float32)

        assert len(self.input_mean) == NUM_EMG, \
            f"expected {NUM_EMG}ch norm stats, got {len(self.input_mean)}"

        self.buffer = collections.deque(maxlen=WINDOW_LENGTH)
        for _ in range(WINDOW_LENGTH):
            self.buffer.append(np.zeros(NUM_EMG, dtype=np.float32))
        self._lock = threading.Lock()
        self.last_pred = None

        self.running = False
        self.thread = None

    def push(self, samples):
        with self._lock:
            for s in samples:
                self.buffer.append(s)

    def _snapshot(self):
        with self._lock:
            return np.array(self.buffer, dtype=np.float32)

    @torch.no_grad()
    def predict(self):
        arr = self._snapshot()
        arr[:, :NUM_EMG] = filter_emg(arr[:, :NUM_EMG], fs=SAMPLE_RATE)
        arr = (arr - self.input_mean) / self.input_std
        x = torch.from_numpy(arr).float().transpose(0, 1).unsqueeze(0).to(self.device)
        out = self.model(x)
        latest = out[0, :, -1].cpu().numpy()
        if self.last_pred is not None:
            latest = self.ema_alpha * latest + (1 - self.ema_alpha) * self.last_pred
        self.last_pred = latest
        return latest

    def start(self, stream):
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(stream,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _loop(self, stream):
        while self.running:
            batch = stream.get_batch(max_items=512)
            if batch:
                rows = [s.emg.astype(np.float32) for s in batch]
                self.push(rows)
            else:
                time.sleep(0.002)


def draw_hand(frame, uv_21, color=(0, 200, 0)):
    h, w = frame.shape[:2]
    pts = [(int(uv_21[i, 0] * w), int(uv_21[i, 1] * h)) for i in range(21)]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, color, 1)


def draw_angles_gt_pred(frame, uv_21, gt_deg, pred_deg):
    """Green GT angle, red predicted angle next to it — angle_visualizer style."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.38, 1
    green = (0, 255, 200)
    red = (60, 60, 255)

    for lm_idx, entries in JOINT_LABELS.items():
        px = int(uv_21[lm_idx, 0] * w)
        py = int(uv_21[lm_idx, 1] * h)

        pieces = []
        for k, (ai, lbl) in enumerate(entries):
            if k > 0:
                pieces.append(("  ", green))
            pieces.append((f"{lbl}:{gt_deg[ai]:+.0f}\u00b0", green))
            pieces.append((f"/{pred_deg[ai]:+.0f}\u00b0", red))

        sizes = [cv2.getTextSize(t, font, fs, thick)[0] for t, _ in pieces]
        tw = sum(s[0] for s in sizes)
        th = max(s[1] for s in sizes)

        ox, oy = px + 6, py - 4
        if ox + tw > w:
            ox = px - tw - 6
        if oy - th < 0:
            oy = py + th + 4

        cv2.rectangle(frame, (ox - 2, oy - th - 2),
                      (ox + tw + 2, oy + 2), (0, 0, 0), -1)

        cx = ox
        for (t, col), (sw, _) in zip(pieces, sizes):
            cv2.putText(frame, t, (cx, oy), font, fs, col, thick, cv2.LINE_AA)
            cx += sw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/neuropose_angles_best.pt")
    parser.add_argument("--config",
                        default="emg2pose/config/network/neuropose_angles.yaml")
    parser.add_argument("--dataset", default="data/dataset_angles_no_imu.hdf5")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    print(f"Device: {args.device}")

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(_root, p)

    predictor = AngleEMGPredictor(
        model_path=abspath(args.model),
        config_path=abspath(args.config),
        dataset_path=abspath(args.dataset),
        device=args.device,
    )

    stream = EMGStream(synthetic=args.synthetic)
    stream.start()
    predictor.start(stream)

    extractor = HandPoseExtractor(running_mode="video")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        predictor.stop(); stream.stop()
        return

    print("Green = GT angle (deg). Red = |pred − gt| (deg). ESC/Q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = extractor.extract(rgb)

            pred_rad = predictor.predict()
            pred_deg = np.degrees(pred_rad)

            if pose is not None:
                gt_rad = landmarks_to_angles_3d(pose.landmarks)
                gt_deg = np.degrees(gt_rad)
                draw_hand(frame, pose.image_uv)
                draw_angles_gt_pred(frame, pose.image_uv, gt_deg, pred_deg)

                mean_err = float(np.mean(np.abs(gt_deg - pred_deg)))
                cv2.putText(frame, f"mean |err| = {mean_err:5.1f} deg",
                            (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 255, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No hand detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Live compare — green GT / red err (deg) — ESC/Q", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cap.release()
        extractor.close()
        predictor.stop()
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
