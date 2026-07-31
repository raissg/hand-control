"""Live 5-angle EMG model vs. MediaPipe GT.

Overlays the 5 palm-to-finger flexion angles on the hand (angle_visualizer style):
green = GT (MediaPipe), red = predicted (EMG model).

Targets: THUMB_CMC_FE (lm 1), INDEX_MCP_FE (lm 5), MIDDLE_MCP_FE (lm 9),
         RING_MCP_FE (lm 13), PINKY_MCP_FE (lm 17).

Usage:
    python -m src.live_compare5
    python -m src.live_compare5 --synthetic
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import threading
import time
from datetime import datetime

import cv2
import h5py
import numpy as np
import torch

from src.acquire import EMGStream
from src.angle_visualizer import landmarks_to_angles_3d
from src.convert_to_hdf5 import filter_emg
from src.ground_truth import HandPoseExtractor, HAND_CONNECTIONS
from src.landmarks_to_angles import ANGLE_NAMES
from src.train import get_model


NUM_EMG = 8
WINDOW_LENGTH = 1000
SAMPLE_RATE = 500

KEEP_INDICES = [0, 5, 9, 13, 17]                        # into 20-angle vector
JOINT_LM = [1, 5, 9, 13, 17]                            # MediaPipe lm to anchor text
LABELS = ["T-CMC", "I-MCP", "M-MCP", "R-MCP", "P-MCP"]


class AngleEMGPredictor:
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
        assert len(self.input_mean) == NUM_EMG
        self.buffer = collections.deque(maxlen=WINDOW_LENGTH)
        for _ in range(WINDOW_LENGTH):
            self.buffer.append(np.zeros(NUM_EMG, dtype=np.float32))
        self._lock = threading.Lock()
        self.last_pred = None
        self.running = False; self.thread = None

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
        out = self.model(x)                         # (1, 5, T)
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
                self.push([s.emg.astype(np.float32) for s in batch])
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


def draw_gt_pred(frame, uv_21, gt5_deg, pred5_deg):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.42, 1
    green = (0, 255, 200); red = (60, 60, 255)

    for k, lm_idx in enumerate(JOINT_LM):
        px = int(uv_21[lm_idx, 0] * w)
        py = int(uv_21[lm_idx, 1] * h)
        pieces = [
            (f"{LABELS[k]}:{gt5_deg[k]:+.0f}\u00b0", green),
            (f"/{pred5_deg[k]:+.0f}\u00b0", red),
        ]
        sizes = [cv2.getTextSize(t, font, fs, thick)[0] for t, _ in pieces]
        tw = sum(s[0] for s in sizes); th = max(s[1] for s in sizes)
        ox, oy = px + 8, py - 6
        if ox + tw > w: ox = px - tw - 8
        if oy - th < 0: oy = py + th + 6
        cv2.rectangle(frame, (ox - 2, oy - th - 2),
                      (ox + tw + 2, oy + 2), (0, 0, 0), -1)
        cx = ox
        for (t, col), (sw, _) in zip(pieces, sizes):
            cv2.putText(frame, t, (cx, oy), font, fs, col, thick, cv2.LINE_AA)
            cx += sw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/neuropose_angles5_best.pt")
    p.add_argument("--config",
                   default="emg2pose/config/network/neuropose_angles5.yaml")
    p.add_argument("--dataset", default="data/dataset_angles_no_imu.hdf5")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    print(f"Device: {args.device}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ab = lambda q: q if os.path.isabs(q) else os.path.join(root, q)

    predictor = AngleEMGPredictor(ab(args.model), ab(args.config),
                                   ab(args.dataset), args.device)
    stream = EMGStream(synthetic=args.synthetic)
    stream.start(); predictor.start(stream)

    extractor = HandPoseExtractor(running_mode="video")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        predictor.stop(); stream.stop(); return

    print("Green = GT, Red = Pred (5 palm-to-finger FE, degrees).")
    print("R = start/stop recording. ESC/Q = quit.")

    recording = False
    rec_rows = []
    rec_path = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = extractor.extract(rgb)

            pred5_deg = np.degrees(predictor.predict())   # (5,)
            ts = time.time()

            if pose is not None:
                gt20_rad = landmarks_to_angles_3d(pose.landmarks)
                gt5_deg = np.degrees(gt20_rad[KEEP_INDICES])
                draw_hand(frame, pose.image_uv)
                draw_gt_pred(frame, pose.image_uv, gt5_deg, pred5_deg)
                mean_err = float(np.mean(np.abs(gt5_deg - pred5_deg)))
                cv2.putText(frame, f"mean |err| = {mean_err:5.1f} deg",
                            (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 255, 200), 1, cv2.LINE_AA)
                if recording:
                    rec_rows.append([f"{ts:.4f}"] +
                                    [f"{v:.3f}" for v in gt5_deg] +
                                    [f"{v:.3f}" for v in pred5_deg])
            else:
                gt5_deg = None
                cv2.putText(frame, "No hand detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Recording indicator
            if recording:
                cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 0, 255), -1)
                cv2.putText(frame, f"REC {len(rec_rows)}",
                            (frame.shape[1] - 80, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            cv2.imshow("Live compare 5 — R=rec, ESC/Q=quit", frame)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('r'):
                if not recording:
                    recording = True
                    rec_rows = []
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = os.path.join(root, f"recordings/live_rec_{stamp}.csv")
                    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
                    print(f"Recording started → {rec_path}")
                else:
                    recording = False
                    header = (["timestamp"] +
                              [f"gt_{n}" for n in LABELS] +
                              [f"pred_{n}" for n in LABELS])
                    with open(rec_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(header)
                        w.writerows(rec_rows)
                    print(f"Recording saved: {rec_path}  ({len(rec_rows)} frames)")
                    rec_rows = []

            elif k in (27, ord('q')):
                break
    finally:
        if recording and rec_rows and rec_path:
            header = (["timestamp"] +
                      [f"gt_{n}" for n in LABELS] +
                      [f"pred_{n}" for n in LABELS])
            with open(rec_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rec_rows)
            print(f"Recording saved on exit: {rec_path}  ({len(rec_rows)} frames)")
        cap.release(); extractor.close()
        predictor.stop(); stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
