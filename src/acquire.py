"""
EMG acquisition from MindRove armband (8ch, 500 Hz).

Provides EMGStream — a thread-safe wrapper around MindRove BoardShim that
handles connection, streaming, real-time filtering, and delivers samples
via a queue or callback.

Usage:
    from src.acquire import EMGStream

    stream = EMGStream(synthetic=True)  # or False for real hardware
    stream.start()

    while True:
        sample = stream.get()  # blocks until data available
        # sample.emg: np.ndarray (8,)
        # sample.accel: np.ndarray (3,)
        # sample.gyro: np.ndarray (3,)
        # sample.timestamp: float (seconds since stream start)

    stream.stop()

Or with a callback:

    def on_samples(samples: list[EMGSample]):
        for s in samples:
            print(s.emg)

    stream = EMGStream(synthetic=True, callback=on_samples)
    stream.start()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Callable, Optional

import numpy as np
from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams
from mindrove.data_filter import DataFilter, DetrendOperations, FilterTypes


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
BANDPASS_LOW = 20.0       # Hz — below this is motion artifact
BANDPASS_HIGH = 450.0     # Hz — stay within Nyquist for 500 Hz
NOTCH_FREQ = 50.0         # power-line (60 Hz US, 50 Hz EU, Qatar uses 50 Hz)
FILTER_ORDER = 4
POLL_INTERVAL = 0.004     # ~250 Hz polling, well above sample delivery rate


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class EMGSample:
    """Single timestep of EMG + IMU data."""
    timestamp: float          # seconds since stream.start()
    emg: np.ndarray           # (num_channels,) filtered EMG
    accel: np.ndarray         # (3,) accelerometer
    gyro: np.ndarray          # (3,) gyroscope


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class EMGStream:
    """Thread-safe MindRove EMG streamer with real-time filtering.

    Parameters
    ----------
    synthetic : bool
        If True, use BrainFlow's synthetic board (no hardware needed).
    ip : str
        Armband IP address (ignored if synthetic).
    port : int
        Armband port (ignored if synthetic).
    queue_maxsize : int
        Max items in the output queue. Oldest samples are dropped when full.
    notch_hz : float
        Power-line notch frequency (60 Hz US, 50 Hz EU).
    bandpass_low : float
        Bandpass lower cutoff in Hz.
    bandpass_high : float
        Bandpass upper cutoff in Hz.
    filter_order : int
        Butterworth filter order.
    enable_filter : bool
        If False, deliver raw (unfiltered) EMG.
    callback : callable, optional
        Called with a list[EMGSample] each time new data arrives.
        If provided, samples are NOT put on the queue (use one or the other).
    """

    def __init__(
        self,
        synthetic: bool = False,
        ip: str = "192.168.4.1",
        port: int = 4210,
        queue_maxsize: int = 4096,
        notch_hz: float = NOTCH_FREQ,
        bandpass_low: float = BANDPASS_LOW,
        bandpass_high: float = BANDPASS_HIGH,
        filter_order: int = FILTER_ORDER,
        enable_filter: bool = True,
        callback: Optional[Callable[[list[EMGSample]], None]] = None,
    ):
        # Board selection
        if synthetic:
            self._board_id = BoardIds.SYNTHETIC_BOARD
            self._params = MindRoveInputParams()
        else:
            self._board_id = BoardIds.MINDROVE_WIFI_BOARD
            self._params = MindRoveInputParams()
            self._params.ip_address = ip
            self._params.ip_port = port

        # Channel indices (resolved after prepare_session)
        self._emg_channels: list[int] = []
        self._accel_channels: list[int] = []
        self._gyro_channels: list[int] = []

        # Filter config
        self._notch_hz = notch_hz
        self._bandpass_low = bandpass_low
        self._bandpass_high = bandpass_high
        self._filter_order = filter_order
        self._enable_filter = enable_filter

        # Output
        self._callback = callback
        self._queue: Queue[EMGSample] = Queue(maxsize=queue_maxsize)

        # Internal state
        self._board: Optional[BoardShim] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._t0: float = 0.0  # epoch time at start()

        # Public read-only properties set after start()
        self.sampling_rate: int = 0
        self.num_channels: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to the board and begin streaming."""
        if self._running:
            raise RuntimeError("EMGStream is already running")

        BoardShim.enable_board_logger()

        self._board = BoardShim(self._board_id, self._params)
        self._board.prepare_session()
        self._board.start_stream()

        # Resolve channel indices
        self._emg_channels = list(BoardShim.get_emg_channels(self._board_id))[:8]
        self._accel_channels = list(BoardShim.get_accel_channels(self._board_id))
        self._gyro_channels = list(BoardShim.get_gyro_channels(self._board_id))
        self.sampling_rate = BoardShim.get_sampling_rate(self._board_id)
        self.num_channels = len(self._emg_channels)

        self._t0 = time.time()
        self._running = True

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        board_name = "SYNTHETIC" if self._board_id == BoardIds.SYNTHETIC_BOARD else "MindRove"
        print(
            f"EMGStream started: {board_name} | "
            f"{self.num_channels}ch @ {self.sampling_rate} Hz | "
            f"filter={'on' if self._enable_filter else 'off'}"
        )

    def stop(self) -> None:
        """Stop streaming and release the board."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._board is not None:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception as e:
                print(f"EMGStream cleanup error: {e}")
            self._board = None
        print("EMGStream stopped.")

    def get(self, timeout: Optional[float] = None) -> EMGSample:
        """Get the next sample from the queue (blocks).

        Raises queue.Empty if timeout expires.
        """
        return self._queue.get(timeout=timeout)

    def get_batch(self, max_items: int = 256) -> list[EMGSample]:
        """Drain up to max_items from the queue without blocking."""
        batch: list[EMGSample] = []
        for _ in range(max_items):
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break
        return batch

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def start_time(self) -> float:
        """Epoch time when start() was called."""
        return self._t0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread: poll board, filter, enqueue."""
        while self._running:
            try:
                n_available = self._board.get_board_data_count()
                if n_available == 0:
                    time.sleep(POLL_INTERVAL)
                    continue

                # Pull all available samples: shape (num_rows, n_available)
                data = self._board.get_board_data(n_available)

                # --- Filter each EMG channel in-place ---
                if self._enable_filter:
                    for ch in self._emg_channels:
                        row = data[ch]
                        DataFilter.detrend(row, DetrendOperations.LINEAR.value)
                        DataFilter.perform_bandpass(
                            row,
                            self.sampling_rate,
                            self._bandpass_low,
                            self._bandpass_high,
                            self._filter_order,
                            FilterTypes.BUTTERWORTH.value,
                            0,
                        )
                        DataFilter.remove_environmental_noise(
                            row, self.sampling_rate, 1  # 60 Hz
                        )

                # --- Build samples ---
                # Compute timestamps: last sample = now, earlier samples spaced by 1/fs
                now = time.time() - self._t0
                dt = 1.0 / self.sampling_rate
                # timestamps for each column, oldest first
                timestamps = now - (n_available - 1 - np.arange(n_available)) * dt

                emg_block = data[self._emg_channels]    # (num_ch, n)
                accel_block = data[self._accel_channels] if self._accel_channels else None
                gyro_block = data[self._gyro_channels] if self._gyro_channels else None

                samples: list[EMGSample] = []
                for i in range(n_available):
                    sample = EMGSample(
                        timestamp=timestamps[i],
                        emg=emg_block[:, i].copy(),
                        accel=(
                            accel_block[:, i].copy()
                            if accel_block is not None
                            else np.zeros(3)
                        ),
                        gyro=(
                            gyro_block[:, i].copy()
                            if gyro_block is not None
                            else np.zeros(3)
                        ),
                    )
                    samples.append(sample)

                # --- Deliver ---
                if self._callback is not None:
                    self._callback(samples)
                else:
                    for sample in samples:
                        try:
                            self._queue.put_nowait(sample)
                        except Full:
                            # Drop oldest to make room
                            try:
                                self._queue.get_nowait()
                            except Empty:
                                pass
                            try:
                                self._queue.put_nowait(sample)
                            except Full:
                                pass

                time.sleep(POLL_INTERVAL)

            except Exception as e:
                if self._running:
                    print(f"EMGStream poll error: {e}")
                    time.sleep(0.1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EMGStream smoke test")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic board")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to run")
    parser.add_argument("--no-filter", action="store_true", help="Disable filtering")
    args = parser.parse_args()

    stream = EMGStream(synthetic=args.synthetic, enable_filter=not args.no_filter)
    stream.start()

    t_end = time.time() + args.duration
    total_samples = 0
    last_print = 0.0

    try:
        while time.time() < t_end:
            batch = stream.get_batch(max_items=512)
            total_samples += len(batch)

            if batch and time.time() - last_print > 1.0:
                s = batch[-1]
                print(
                    f"t={s.timestamp:6.2f}s | "
                    f"emg=[{', '.join(f'{v:+8.1f}' for v in s.emg)}] | "
                    f"accel=[{', '.join(f'{v:+6.2f}' for v in s.accel)}] | "
                    f"queue={stream.queue_size}"
                )
                last_print = time.time()
            elif not batch:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()

    elapsed = args.duration
    print(f"\nReceived {total_samples} samples in {elapsed:.1f}s "
          f"({total_samples / elapsed:.0f} samples/sec, "
          f"expected {stream.sampling_rate})")
