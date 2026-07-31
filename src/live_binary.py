"""Live 5-bit finger-bend inference.

Displays 5 LEDs (green=bent, grey=extended), updated every ``hop`` ms.
Window: 200 ms (100 samples @ 500Hz). Hop: 50 ms by default → 20 Hz updates.

Smoothing pipeline (operates in *logit* space when ``logit_ema=True``):
  raw logit -> median filter (K frames) -> EMA (alpha) -> sigmoid
              -> [snap to supported pattern] OR [per-bit hysteresis]
              -> N-frame debounce (vector-level)
              -> per-bit minimum-dwell debounce (time-domain)
              -> bit
- Median kills 1-2 frame spikes at the source.
- Logit-space EMA softens jitter without the non-linearity that makes
  prob-space EMA wobble near the decision threshold.
- Hysteresis (ON >on_thresh, OFF <off_thresh) prevents threshold-hover chatter.
- Optional pattern snap forces the output into the canonical set the model
  was trained on (kills single-bit chatter that takes you out of vocabulary).
- ``min_consecutive`` frames: new bit-vector only commits after this many
  identical frames in a row (vector-level frame debounce).
- ``min_dwell_s`` seconds: per finger, a state flip is only committed after
  the new value has persisted for this long. Independent per bit so e.g.
  the thumb can update while the ring finger is still settling.
Displayed % on each LED is the post-smoothing prob.

Usage:
    python -m src.live_binary
    python -m src.live_binary --synthetic
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
from src.binary_model import EMG5Bit, EMG5BitMulti  # === IMU ADDITION ===
from src.convert_to_hdf5 import filter_emg
from src.dataset_bits import get_supported_patterns


NUM_EMG = 8
NUM_IMU = 6                          # === IMU ADDITION === accel(3) + gyro(3)
SAMPLE_RATE = 500
LABELS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]


# ============================================================================
# CANONICAL LIVE-INFERENCE CONFIG
# ----------------------------------------------------------------------------
# Single source of truth for the smoothing/temporal-stability knobs used by
# the live predictor. ``live_binary.main()`` uses these as argparse defaults;
# downstream callers (e.g. ``hand_control_emg_arduino``) should consume them
# via :func:`build_predictor` rather than re-stating the kwargs themselves.
# ============================================================================
LIVE_INFERENCE_DEFAULTS: dict = {
    # Hysteresis (only used when no supported_patterns are passed).
    "on_thresh": 0.7,
    "off_thresh": 0.3,
    # Smoothing time constants.
    "ema_alpha": 0.18,        # ~6-frame EMA TC at 50 ms hop -> ~300 ms lag.
    "median_k": 9,            # 9 frames * 50 ms = 450 ms median window.
    "logit_ema": True,        # EMA in logit space, sigmoid afterwards.
    # Frame-level debounce: a new bit/pattern must persist for N frames
    # in a row before it is committed by the predictor.
    "min_consecutive": 3,
    # Per-finger minimum dwell (seconds). After the vector debounce above,
    # each bit individually must hold its new value for at least this long
    # before the change is published. Suppresses sub-blip flips even when
    # the snapped pattern briefly oscillates.
    "min_dwell_s": 0.2,
    # Extra raw EMG samples kept beyond the model window so filtfilt has
    # run-up; matches training where each segment was filtered as one block.
    "filter_pad": 200,
}


def build_predictor(
    model_path: str,
    dataset_path: str | None,
    device,
    *,
    supported_patterns: list[str] | None = None,
    window: int = 100,
    **overrides,
) -> "BinaryPredictor":
    """Build a :class:`BinaryPredictor` with the canonical live-inference config.

    All knobs default to ``LIVE_INFERENCE_DEFAULTS``; pass any subset as
    ``overrides`` (e.g. ``ema_alpha=0.25``) to tweak individual values.
    Pass ``supported_patterns=...`` to enable snap-to-canonical-pattern.
    """
    cfg = {**LIVE_INFERENCE_DEFAULTS, **overrides}
    return BinaryPredictor(
        model_path,
        dataset_path,
        device,
        window=window,
        supported_patterns=supported_patterns,
        **cfg,
    )


def add_inference_args(ap):
    """Register the live-inference smoothing/temporal-stability flags.

    Defaults read from ``LIVE_INFERENCE_DEFAULTS``. Use
    :func:`inference_overrides_from_args` to convert the parsed Namespace
    back into a kwargs dict for :func:`build_predictor`.
    """
    D = LIVE_INFERENCE_DEFAULTS
    ap.add_argument(
        "--on-thresh", type=float, default=D["on_thresh"], metavar="P",
        help=f"Hysteresis: latch bit ON when probability > P (default {D['on_thresh']}). "
             "Ignored when --snap is on.",
    )
    ap.add_argument(
        "--off-thresh", type=float, default=D["off_thresh"], metavar="P",
        help=f"Hysteresis: latch bit OFF when probability < P (default {D['off_thresh']}). "
             "Ignored when --snap is on.",
    )
    ap.add_argument(
        "--ema-alpha", type=float, default=D["ema_alpha"], metavar="A",
        help=f"EMA weight on the newest sample (default {D['ema_alpha']}). "
             "Lower = smoother + more latency. 1.0 disables smoothing. "
             "Runs in logit space unless --no-logit-ema is set.",
    )
    ap.add_argument(
        "--median-k", type=int, default=D["median_k"], metavar="K",
        help=f"Per-bit median filter window (default {D['median_k']} frames). "
             "Kills 1-2 frame prob spikes. Set 1 to disable.",
    )
    ap.add_argument(
        "--no-logit-ema", dest="logit_ema", action="store_false",
        help="Run EMA on probabilities (legacy) instead of on logits.",
    )
    ap.set_defaults(logit_ema=D["logit_ema"])
    ap.add_argument(
        "--min-consecutive", type=int, default=D["min_consecutive"], metavar="N",
        help=f"Vector-level frame debounce: only commit a new bit/pattern after "
             f"the same value persists for N consecutive frames "
             f"(default {D['min_consecutive']}). 1 disables.",
    )
    ap.add_argument(
        "--min-dwell-s", type=float, default=D["min_dwell_s"], metavar="S",
        help=f"Per-finger time-domain debounce: a state flip is only "
             f"published after the new value persists for S seconds "
             f"(default {D['min_dwell_s']}). 0 disables.",
    )
    ap.add_argument(
        "--filter-pad", type=int, default=D["filter_pad"], metavar="S",
        help=f"Extra raw EMG samples kept beyond the model window so filtfilt "
             f"has run-up (default {D['filter_pad']} = "
             f"{D['filter_pad'] * 1000 // SAMPLE_RATE}ms at {SAMPLE_RATE}Hz). 0 disables.",
    )


def inference_overrides_from_args(args) -> dict:
    """Convert an argparse Namespace produced by :func:`add_inference_args`
    into the kwargs dict accepted by :func:`build_predictor`."""
    return {
        "on_thresh": args.on_thresh,
        "off_thresh": args.off_thresh,
        "ema_alpha": args.ema_alpha,
        "median_k": args.median_k,
        "logit_ema": args.logit_ema,
        "min_consecutive": args.min_consecutive,
        "min_dwell_s": args.min_dwell_s,
        "filter_pad": args.filter_pad,
    }


class BinaryPredictor:
    def __init__(
        self,
        ckpt_path,
        dataset_path,
        device,
        window=100,
        on_thresh: float = 0.6,
        off_thresh: float = 0.4,
        ema_alpha: float = 0.35,
        median_k: int = 5,
        # === TIER 1 SMOOTHING =================================================
        # Opt-in temporal-stability features. Defaults preserve legacy behavior
        # so existing callers (e.g. ``python -m src.live_binary``) are unchanged.
        # ====================================================================
        logit_ema: bool = True,
        min_consecutive: int = 1,
        min_dwell_s: float = 0.0,
        filter_pad: int = 0,
        supported_patterns: list[str] | None = None,
    ):
        self.device = torch.device(device)
        self.on_thresh = float(on_thresh)
        self.off_thresh = float(off_thresh)
        self.ema_alpha = float(ema_alpha)
        self.median_k = int(median_k)
        self.logit_ema = bool(logit_ema)
        self.min_consecutive = max(1, int(min_consecutive))
        self.min_dwell_s = max(0.0, float(min_dwell_s))
        self.filter_pad = max(0, int(filter_pad))
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1] (got {self.ema_alpha})"
            )
        if self.median_k < 1:
            raise ValueError(
                f"median_k must be >= 1 (got {self.median_k})"
            )
        # Rolling window of recent raw outputs for the median filter.
        # Stores logits when logit_ema=True, probs otherwise. Median is
        # invariant to the sigmoid so either is fine — we just keep the
        # buffer aligned with whichever space the EMA runs in.
        self._smooth_buf: collections.deque = collections.deque(maxlen=self.median_k)
        self._probs_ema = None   # used when logit_ema=False
        self._logits_ema = None  # used when logit_ema=True
        if not (0.0 < self.off_thresh < self.on_thresh < 1.0):
            raise ValueError(
                "hysteresis requires 0 < off_thresh < on_thresh < 1 "
                f"(got off={self.off_thresh}, on={self.on_thresh})"
            )

        # Snap-to-supported-pattern setup.
        if supported_patterns is not None:
            sup = []
            for p in supported_patterns:
                if not (len(p) == 5 and set(p) <= {"0", "1"}):
                    raise ValueError(f"supported_patterns: bad entry {p!r}")
                sup.append([int(c) for c in p])
            if not sup:
                raise ValueError("supported_patterns is empty")
            self._sup_patterns = np.asarray(sup, dtype=np.float32)  # (N, 5)
            self._sup_strings = list(supported_patterns)
        else:
            self._sup_patterns = None
            self._sup_strings = None

        # N-frame debounce state: candidate bits seen for `_pending_count`
        # consecutive frames; we only commit (return) once the count clears
        # min_consecutive. Until then we keep returning the last committed bits.
        self._pending_bits: np.ndarray | None = None
        self._pending_count: int = 0
        self._committed_bits = np.zeros(5, dtype=np.int32)

        # Per-finger time-domain dwell state. After the vector debounce
        # commits, each bit independently has to hold its new value for at
        # least ``min_dwell_s`` seconds before being published to callers.
        # ``_dwell_pending[i]`` is the latest candidate, ``_dwell_since[i]``
        # the monotonic timestamp at which it became different from the
        # currently-published value.
        self._dwell_published = np.zeros(5, dtype=np.int32)
        self._dwell_pending = np.zeros(5, dtype=np.int32)
        self._dwell_since = np.zeros(5, dtype=np.float64)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        is_dict = isinstance(ckpt, dict)
        state = ckpt["state_dict"] if is_dict and "state_dict" in ckpt else ckpt
        # === IMU ADDITION =================================================
        # Pick architecture from the checkpoint. New multistream checkpoints
        # carry ``multistream=True`` and ``imu_mean``/``imu_std``. Older
        # EMG-only checkpoints are detected by the absence of the flag.
        # ===================================================================
        self.multistream = bool(is_dict and ckpt.get("multistream", False))
        if self.multistream:
            self.model = EMG5BitMulti(
                emg_channels=NUM_EMG, imu_channels=NUM_IMU, n_out=5,
            ).to(self.device)
            print("BinaryPredictor: multistream model (EMG + IMU).")
        else:
            self.model = EMG5Bit(in_channels=NUM_EMG, n_out=5).to(self.device)
            print("BinaryPredictor: single-stream model (EMG only).")
        self.model.load_state_dict(state)
        self.model.eval()

        # Prefer norm stats stored in the checkpoint (self-labeled training).
        # Fall back to the HDF5 attrs (HDF5-trained checkpoint).
        if is_dict and "input_mean" in ckpt and "input_std" in ckpt:
            self.input_mean = np.asarray(ckpt["input_mean"], dtype=np.float32)[:NUM_EMG]
            self.input_std = np.asarray(ckpt["input_std"], dtype=np.float32)[:NUM_EMG]
            print("Loaded norm stats from checkpoint.")
        elif dataset_path:
            with h5py.File(dataset_path, "r") as f:
                self.input_mean = np.asarray(f.attrs["input_mean"], dtype=np.float32)[:NUM_EMG]
                self.input_std = np.asarray(f.attrs["input_std"], dtype=np.float32)[:NUM_EMG]
            print(f"Loaded norm stats from dataset {dataset_path}.")
        else:
            raise ValueError("No norm stats in checkpoint and no --dataset provided.")

        # === IMU ADDITION =================================================
        # IMU norm stats are kept separate (different scale than EMG). Only
        # required when running a multistream checkpoint.
        # ===================================================================
        self.imu_mean = self.imu_std = None
        if self.multistream:
            if is_dict and "imu_mean" in ckpt and "imu_std" in ckpt:
                self.imu_mean = np.asarray(ckpt["imu_mean"], dtype=np.float32)[:NUM_IMU]
                self.imu_std = np.asarray(ckpt["imu_std"], dtype=np.float32)[:NUM_IMU]
                print("Loaded IMU norm stats from checkpoint.")
            else:
                raise ValueError(
                    "Multistream checkpoint missing imu_mean/imu_std — "
                    "retrain with --multistream after recording IMU sessions."
                )

        # Window length: checkpoint wins if it knows.
        self.window = int(ckpt["window"]) if is_dict and "window" in ckpt else window
        print(f"Window: {self.window} samples ({self.window * 1000 // SAMPLE_RATE} ms)")

        # EMG ring buffer holds `window + filter_pad` raw samples. We filter
        # the longer buffer once per predict() and then slice the most recent
        # `window` samples — this dramatically shrinks the per-window filtfilt
        # edge transients that show up when you filter just 100 samples at a
        # time. ``filter_pad=0`` reproduces the legacy behavior exactly.
        emg_buf_len = self.window + self.filter_pad
        self.buffer = collections.deque(maxlen=emg_buf_len)
        for _ in range(emg_buf_len):
            self.buffer.append(np.zeros(NUM_EMG, dtype=np.float32))
        # === IMU ADDITION === parallel ring buffer for the IMU stream so the
        # multistream model gets a (6, W) window aligned to the EMG window.
        # IMU is not filtered, so it doesn't need the pad — keep it at `window`.
        self.imu_buffer = collections.deque(maxlen=self.window)
        for _ in range(self.window):
            self.imu_buffer.append(np.zeros(NUM_IMU, dtype=np.float32))
        self._lock = threading.Lock()
        self.running = False
        self.thread = None
        # Per-finger latched bit state for hysteresis (updated in predict).
        self._bits_hyst = np.zeros(5, dtype=np.int32)

    def push(self, samples):
        """Append EMG samples to the rolling window.

        ``samples`` is the existing list of (T,) EMG arrays for backward
        compatibility with EMG-only callers. The multistream path uses
        ``push_emg_imu`` instead so it can keep the two buffers aligned.
        """
        with self._lock:
            for s in samples:
                self.buffer.append(s)

    # === IMU ADDITION =====================================================
    # Aligned push for multistream mode: takes a list of (emg, imu) arrays.
    # Both buffers advance in lockstep so the (8, W) and (6, W) windows
    # always cover the exact same time range.
    # ======================================================================
    def push_emg_imu(self, emg_samples, imu_samples):
        with self._lock:
            for e, i in zip(emg_samples, imu_samples):
                self.buffer.append(e)
                self.imu_buffer.append(i)

    def _snap(self):
        with self._lock:
            return np.array(self.buffer, dtype=np.float32)

    # === IMU ADDITION === snapshot helper for the IMU window.
    def _snap_imu(self):
        with self._lock:
            return np.array(self.imu_buffer, dtype=np.float32)

    @torch.no_grad()
    def predict(self):
        arr_full = self._snap()                          # (W + pad, 8)
        # Filter once on the longer buffer; the trailing `window` samples
        # have far smaller filtfilt edge transients than filtering 100
        # samples in isolation. This matches the training distribution
        # better, where each segment was filtered as one long block.
        arr_full = filter_emg(arr_full, fs=SAMPLE_RATE)
        arr = arr_full[-self.window:]                    # (W, 8)
        arr = (arr - self.input_mean) / self.input_std
        x = torch.from_numpy(arr).float().transpose(0, 1).unsqueeze(0).to(self.device)
        # === IMU ADDITION =================================================
        # Multistream: also build the IMU tensor from its own buffer + stats
        # and run the two-input forward. Single-stream path is unchanged.
        # ===================================================================
        if self.multistream:
            imu_arr = self._snap_imu()                    # (W, 6)
            imu_arr = (imu_arr - self.imu_mean) / self.imu_std
            x_imu = (torch.from_numpy(imu_arr).float()
                     .transpose(0, 1).unsqueeze(0).to(self.device))
            logits_t = self.model(x, x_imu)
        else:
            logits_t = self.model(x)                     # (1, 5)
        logits_raw = logits_t[0].cpu().numpy().astype(np.float32)  # (5,)

        # (1) Median filter over the last K raw frames (per bit). Median is
        # invariant under sigmoid, so it doesn't matter if we median in
        # logit or prob space — we keep it in whichever space the EMA runs.
        if self.logit_ema:
            smooth_in = logits_raw
        else:
            smooth_in = (1.0 / (1.0 + np.exp(-logits_raw))).astype(np.float32)

        if self.median_k > 1:
            self._smooth_buf.append(smooth_in)
            smooth_med = np.median(
                np.stack(self._smooth_buf, axis=0), axis=0
            ).astype(np.float32)
        else:
            smooth_med = smooth_in

        # (2) EMA. Doing it in *logit* space avoids the non-linear blur
        # near the decision boundary that makes prob-space EMA wobble.
        a = self.ema_alpha
        if self.logit_ema:
            if self._logits_ema is None:
                self._logits_ema = smooth_med.copy()
            else:
                self._logits_ema = a * smooth_med + (1.0 - a) * self._logits_ema
            probs = (1.0 / (1.0 + np.exp(-self._logits_ema))).astype(np.float32)
        else:
            if self._probs_ema is None:
                self._probs_ema = smooth_med.copy()
            else:
                self._probs_ema = a * smooth_med + (1.0 - a) * self._probs_ema
            probs = self._probs_ema

        # (3) Decode probs -> candidate bits.
        if self._sup_patterns is not None:
            # Snap to the canonical pattern with min L2 distance to probs.
            # Equivalent to picking the most likely class under an
            # independent-bit Bernoulli model among the supported set.
            diffs = self._sup_patterns - probs[None, :]
            dists = np.sum(diffs * diffs, axis=1)
            cand = self._sup_patterns[int(np.argmin(dists))].astype(np.int32)
        else:
            # Per-finger hysteresis (latched across calls).
            for i in range(5):
                p = float(probs[i])
                if p > self.on_thresh:
                    self._bits_hyst[i] = 1
                elif p < self.off_thresh:
                    self._bits_hyst[i] = 0
                # else: keep self._bits_hyst[i] (dead zone)
            cand = self._bits_hyst.copy()

        # (4) N-frame debounce: only commit a new bit vector once the
        # exact same candidate has been seen `min_consecutive` times in
        # a row. Until then we keep returning the last committed bits.
        if self.min_consecutive <= 1:
            self._committed_bits = cand
            committed = cand
        else:
            if (self._pending_bits is None
                    or not np.array_equal(cand, self._pending_bits)):
                self._pending_bits = cand
                self._pending_count = 1
            else:
                self._pending_count += 1
            if self._pending_count >= self.min_consecutive:
                self._committed_bits = cand
            committed = self._committed_bits.copy()

        # (5) Per-finger time-domain dwell. For each bit independently, a
        # new value has to persist for at least ``min_dwell_s`` seconds
        # before it replaces the published value. If the candidate flips
        # back before the timer expires, the timer restarts and the
        # published value never changes. Disabled when min_dwell_s == 0.
        if self.min_dwell_s > 0.0:
            now = time.monotonic()
            for i in range(5):
                target = int(committed[i])
                if target == int(self._dwell_published[i]):
                    # No pending change for this finger; keep timer reset.
                    self._dwell_pending[i] = target
                    self._dwell_since[i] = now
                    continue
                if target != int(self._dwell_pending[i]):
                    # Candidate flipped while waiting; restart the timer.
                    self._dwell_pending[i] = target
                    self._dwell_since[i] = now
                    continue
                # Same candidate as last frame and still differs from what
                # we've published — commit if held long enough.
                if (now - self._dwell_since[i]) >= self.min_dwell_s:
                    self._dwell_published[i] = target
            bits = self._dwell_published.copy()
        else:
            self._dwell_published = committed.copy()
            bits = committed
        return bits, probs

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
                # === IMU ADDITION === in multistream mode, push aligned IMU
                # samples (accel xyz + gyro xyz) alongside EMG. Otherwise
                # only EMG is buffered, exactly as before.
                if self.multistream:
                    emg_list = [s.emg.astype(np.float32) for s in batch]
                    imu_list = [
                        np.concatenate([
                            s.accel.astype(np.float32),
                            s.gyro.astype(np.float32),
                        ])
                        for s in batch
                    ]
                    self.push_emg_imu(emg_list, imu_list)
                else:
                    self.push([s.emg.astype(np.float32) for s in batch])
            else:
                time.sleep(0.002)


def render_panel(
    bits, probs, hop_ms, infer_ms, on_thresh: float, off_thresh: float,
    ema_alpha: float = 1.0, median_k: int = 1, width=520, height=260,
):
    canvas = np.full((height, width, 3), 25, dtype=np.uint8)
    radius = 30
    gap = (width - 2 * 40) // 5
    cy = 100
    for i, lbl in enumerate(LABELS):
        cx = 40 + gap // 2 + i * gap
        colour = (0, 220, 0) if bits[i] else (60, 60, 60)
        cv2.circle(canvas, (cx, cy), radius, colour, -1)
        cv2.circle(canvas, (cx, cy), radius, (220, 220, 220), 2)
        cv2.putText(canvas, lbl, (cx - 32, cy + radius + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{probs[i]*100:.0f}%",
                    (cx - 18, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2, cv2.LINE_AA)

    bitstr = "".join(str(b) for b in bits)
    cv2.putText(canvas, f"bits: {bitstr}", (40, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"hop {hop_ms:.0f}ms  inf {infer_ms:.1f}ms  "
        f"med K={median_k}  ema {ema_alpha:.2f}  "
        f"hyst >{on_thresh} <{off_thresh}",
        (40, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
    )
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/emg5bit_selflabel_best.pt",
                    help="Checkpoint path. Self-labeled ckpts embed norm stats; "
                         "HDF5-trained ckpts need --dataset.")
    ap.add_argument("--dataset", default=None,
                    help="HDF5 for norm stats (only needed if checkpoint lacks them).")
    ap.add_argument("--hop_ms", type=float, default=50.0)
    add_inference_args(ap)
    ap.add_argument(
        "--snap", action="store_true",
        help="Snap output to the nearest canonical pattern from "
             "dataset_bits.get_supported_patterns(). Recommended when the "
             "checkpoint was trained with --remap.",
    )
    ap.add_argument(
        "--no-snap-remap", dest="snap_remap", action="store_false",
        help="With --snap, use the full 15 recorded patterns instead of "
             "the 8 LABEL_REMAP-collapsed ones. Use this if your checkpoint "
             "was trained WITHOUT --remap.",
    )
    ap.set_defaults(snap_remap=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    print(f"Device: {args.device}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ab = lambda q: q if (q is None or os.path.isabs(q)) else os.path.join(root, q)

    supported = None
    if args.snap:
        supported = get_supported_patterns(remap=args.snap_remap)
        print(f"Snap-to-pattern enabled ({len(supported)} canonical patterns"
              f"{', LABEL_REMAP collapsed' if args.snap_remap else ''}): "
              f"{supported}")

    predictor = build_predictor(
        ab(args.model),
        ab(args.dataset),
        args.device,
        supported_patterns=supported,
        **inference_overrides_from_args(args),
    )

    # Raw EMG — filter_emg is applied on each prediction window (matches training).
    stream = EMGStream(synthetic=args.synthetic, enable_filter=False)
    stream.start()

    # Refuse to run if the board isn't delivering samples (mirrors record_bits).
    MIN_WARMUP_S = 1.0
    MIN_SAMPLES = 250
    time.sleep(MIN_WARMUP_S)
    warmup = stream.get_batch(max_items=4096)
    if len(warmup) < MIN_SAMPLES:
        stream.stop()
        raise SystemExit(
            f"EMG stream is not delivering data: got {len(warmup)} samples in "
            f"{MIN_WARMUP_S:.1f}s (expected >={MIN_SAMPLES}). "
            f"Check that the MindRove is powered on, Wi-Fi connected "
            f"(ip {'(synthetic)' if args.synthetic else '192.168.4.1'}), "
            f"and the armband is on your forearm. Aborting."
        )
    arr = np.asarray([s.emg for s in warmup], dtype=np.float32)
    if np.allclose(arr, 0.0):
        stream.stop()
        raise SystemExit(
            "EMG stream delivered only zeros during warm-up — electrodes "
            "may not be making contact. Aborting."
        )
    print(f"Warm-up OK: {len(warmup)} samples in {MIN_WARMUP_S:.1f}s, "
          f"per-channel std: {np.round(arr.std(axis=0), 2).tolist()}")
    # Feed warm-up samples into the predictor so we don't waste them.
    # === IMU ADDITION === route through the dual-stream push when needed.
    if predictor.multistream:
        emg_list = [s.emg.astype(np.float32) for s in warmup]
        imu_list = [
            np.concatenate([
                s.accel.astype(np.float32),
                s.gyro.astype(np.float32),
            ])
            for s in warmup
        ]
        predictor.push_emg_imu(emg_list, imu_list)
    else:
        predictor.push([s.emg.astype(np.float32) for s in warmup])
    predictor.start(stream)

    print("5-bit live predictor. ESC/Q to quit.")
    hop_s = args.hop_ms / 1000.0
    try:
        while True:
            t0 = time.time()
            bits, probs = predictor.predict()
            infer_ms = (time.time() - t0) * 1000
            panel = render_panel(
                bits,
                probs,
                args.hop_ms,
                infer_ms,
                args.on_thresh,
                args.off_thresh,
                ema_alpha=args.ema_alpha,
                median_k=args.median_k,
            )
            cv2.imshow("5-bit live (ESC/Q)", panel)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            # Pace loop at hop_ms
            sleep_left = hop_s - (time.time() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)
    finally:
        predictor.stop(); stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
