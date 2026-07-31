"""Real-time EMG -> 3D hand landmarks inference.

Mirrors the training-time preprocessing exactly:
  1. Raw EMG+IMU streamed from MindRove (filter disabled in stream).
  2. Rolling 1000-sample window (2s @ 500 Hz).
  3. Offline-style bandpass+notch filter on the 8 EMG channels (filtfilt).
  4. Per-channel z-score normalization using stats from dataset.hdf5.
  5. Forward pass -> (1, 60, 1000); take last timestep -> (20, 3).
  6. EMA smoothing over time.

Used by src/visualizer.py for live rendering.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

from src.acquire import EMGStream
from src.convert_to_hdf5 import filter_emg
from src.train import get_model

logger = logging.getLogger(__name__)

# First 8 channels of the 14-channel input are EMG; remaining 6 are IMU (accel+gyro).
NUM_EMG = 8
NUM_CHANNELS = 14


class EMGPredictor:
    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        dataset_path: str | Path,
        device: str = "cpu",
        window_length: int = 1000,
        sample_rate: int = 500,
        ema_alpha: float = 0.5,
        filter_low: float = 20.0,
        filter_high: float = 240.0,
        filter_notch: float = 50.0,
        filter_enable: bool = True,
        emg_only: bool = False,
    ):
        self.device = torch.device(device)
        self.window_length = window_length
        self.sample_rate = sample_rate
        self.ema_alpha = ema_alpha
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        # filter_enable=False: model was trained on raw EMG; skip offline filter.
        # emg_only=True: model expects 8ch (no IMU). Stream still delivers 14ch;
        # we slice before buffering.
        self.filter_enable = filter_enable
        self.emg_only = emg_only
        self.num_channels = NUM_EMG if emg_only else NUM_CHANNELS

        # --- Model ---
        logger.info(f"Loading model from {model_path}")
        self.model = get_model(str(config_path)).to(self.device)
        state = torch.load(str(model_path), map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

        # --- Normalization stats (must match training) ---
        with h5py.File(str(dataset_path), "r") as f:
            if "input_mean" not in f.attrs or "input_std" not in f.attrs:
                raise RuntimeError(
                    f"{dataset_path} has no input_mean/input_std attrs; "
                    "rebuild dataset with convert_to_hdf5.py."
                )
            self.input_mean = np.asarray(f.attrs["input_mean"], dtype=np.float32)
            self.input_std = np.asarray(f.attrs["input_std"], dtype=np.float32)
        if len(self.input_mean) != self.num_channels:
            raise RuntimeError(
                f"Expected {self.num_channels} channels of norm stats "
                f"(emg_only={emg_only}), got {len(self.input_mean)} in {dataset_path}. "
                "The dataset and the --emg-only flag must match the model."
            )

        # --- Rolling buffer (T, C) ---
        self.buffer = collections.deque(maxlen=window_length)
        for _ in range(window_length):
            self.buffer.append(np.zeros(self.num_channels, dtype=np.float32))

        self.last_pred: Optional[np.ndarray] = None  # (20, 3)
        self._lock = threading.Lock()

        self.running = False
        self.thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def push_samples(self, samples: List[np.ndarray]) -> None:
        """Append one or more (14,) rows to the rolling buffer."""
        with self._lock:
            for s in samples:
                self.buffer.append(s)

    def _window_array(self) -> np.ndarray:
        """Snapshot the buffer as a (T, C) float32 array."""
        with self._lock:
            return np.array(self.buffer, dtype=np.float32)

    @torch.no_grad()
    def predict(self) -> np.ndarray:
        """Run one forward pass on the current buffer. Returns (20, 3)."""
        arr = self._window_array()  # (T, C) raw — C is 8 or 14

        # Offline-style filter on the 8 EMG channels, matching training preprocessing.
        # Skipped when the model was trained on raw EMG (filter_enable=False).
        if self.filter_enable:
            emg = arr[:, :NUM_EMG]
            arr[:, :NUM_EMG] = filter_emg(
                emg,
                fs=self.sample_rate,
                low=self.filter_low,
                high=self.filter_high,
                notch=self.filter_notch,
            )

        # Per-channel z-score.
        arr = (arr - self.input_mean) / self.input_std

        # (T, C) -> (1, C, T)
        x = torch.from_numpy(arr).float().transpose(0, 1).unsqueeze(0).to(self.device)
        out = self.model(x)               # (1, 60, T)
        latest = out[0, :, -1].cpu().numpy()  # (60,)
        landmarks = latest.reshape(-1, 3)     # (20, 3)

        if self.last_pred is not None:
            landmarks = (
                self.ema_alpha * landmarks + (1.0 - self.ema_alpha) * self.last_pred
            )
        self.last_pred = landmarks
        return landmarks

    # ------------------------------------------------------------------
    def start_background_loop(self, stream: EMGStream, fps: int = 30) -> None:
        """Spawn a thread that drains `stream` into the rolling buffer.

        Prediction itself is cheap — let the visualizer call predict() on demand
        so it always uses the freshest buffer.
        """
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(stream,), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def _loop(self, stream: EMGStream) -> None:
        while self.running:
            batch = stream.get_batch(max_items=512)
            if batch:
                if self.emg_only:
                    # Drop IMU — model was trained on EMG-only input.
                    rows = [s.emg.astype(np.float32) for s in batch]
                else:
                    rows = [
                        np.concatenate([s.emg, s.accel, s.gyro]).astype(np.float32)
                        for s in batch
                    ]
                self.push_samples(rows)
            else:
                time.sleep(0.002)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="EMGPredictor smoke test.")
    parser.add_argument("--model", default="models/neuropose_mindrove_best.pt")
    parser.add_argument("--config", default="emg2pose/config/network/neuropose_mindrove.yaml")
    parser.add_argument("--dataset", default="data/dataset.hdf5")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    p = EMGPredictor(
        model_path=args.model,
        config_path=args.config,
        dataset_path=args.dataset,
        device=args.device,
    )
    t0 = time.time()
    out = p.predict()
    print(f"Output {out.shape}, took {(time.time()-t0)*1000:.1f} ms")
    print(out)
