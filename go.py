"""
Full pipeline: MindRove EMG -> 5 bits -> Arduino servos (per finger).

Auto-compiles + uploads the sketch, opens the port, streams per-finger bits.
`--lock-s` is a single user-facing lock knob:

  * host side: optional extra cooldown after each serial flip (spam guard)
  * Arduino side: same value pushed as ``L####`` (firmware lock on F flips)

Omit ``--lock-s`` (default) to disable both host + firmware lock (sends L0000).

Press 'c' or ESC to quit. On exit (including Ctrl+C) the hand parks at
DEFAULT_ANGLE on the Arduino. Send 'S' manually only if you need detach.

  bit 0 -> position 20  (open)
  bit 1 -> position 143 (closed)

Inference smoothing flags are inherited from src.live_binary so this script
and ``python -m src.live_binary`` always share the same brain.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import serial
import torch
from pynput import keyboard

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.acquire import EMGStream  # noqa: E402
from src.dataset_bits import get_supported_patterns  # noqa: E402
from src.live_binary import (  # noqa: E402
    add_inference_args,
    build_predictor,
    inference_overrides_from_args,
    render_panel,
)

# ---------------- defaults ----------------
DEFAULT_PORT = "/dev/cu.usbmodemCC8DA22028202"
DEFAULT_BAUD = 115200
DEFAULT_FQBN = "arduino:renesas_uno:unor4wifi"
DEFAULT_MODEL = "models/imu_remap.pt"
DEFAULT_HOP_MS = 50.0
SKETCH_DIR = os.path.join(SCRIPT_DIR, "arduino", "hand_servos")
# ------------------------------------------


def _configure_arduino_lock_ms(ser: serial.Serial, lock_s: float | None) -> None:
    """Apply unified lock setting on firmware via ``L####``.

    ``lock_s=None`` sends ``L0000`` so both host+firmware are effectively
    unlocked by default.
    """
    if lock_s is None:
        lock_ms = 0
    else:
        if lock_s < 0:
            raise SystemExit("--lock-s must be >= 0")
        lock_ms = int(round(lock_s * 1000.0))
    if lock_ms < 0 or lock_ms > 9999:
        raise SystemExit(
            "--lock-s out of range: Arduino protocol supports 0..9.999 seconds"
        )
    ser.reset_input_buffer()
    payload = b"L" + f"{lock_ms:04d}".encode("ascii")
    ser.write(payload)
    ser.flush()
    print(
        f"Unified lock: sent L{lock_ms:04d} "
        f"({lock_ms} ms servo lock on F flips)."
    )


def _parse_args() -> argparse.Namespace:
    """All inference smoothing knobs are inherited from src.live_binary so
    flags like --ema-alpha, --min-dwell-s, etc. behave identically here and
    in ``python -m src.live_binary``."""
    ap = argparse.ArgumentParser(
        description="MindRove EMG -> Arduino servo bridge (5-bit inference).",
    )
    # Arduino-side flags.
    ap.add_argument("--port", default=DEFAULT_PORT,
                    help=f"Serial port to the Arduino (default {DEFAULT_PORT}).")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                    help=f"Serial baud rate (default {DEFAULT_BAUD}).")
    ap.add_argument("--fqbn", default=DEFAULT_FQBN,
                    help=f"arduino-cli FQBN (default {DEFAULT_FQBN}).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Checkpoint path under repo root (default {DEFAULT_MODEL}).")
    ap.add_argument("--hop-ms", type=float, default=DEFAULT_HOP_MS,
                    help=f"Inference loop period in ms (default {DEFAULT_HOP_MS}).")
    ap.add_argument(
        "--lock-s",
        type=float,
        default=None,
        metavar="S",
        help="Unified per-finger lock duration in seconds. Applied to BOTH: "
             "(1) Python-side cooldown after each serial flip, and "
             "(2) Arduino firmware lock via L####. Omit (default) => no lock "
             "(sends L0000 to firmware). Arduino protocol supports 0..9.999 s.",
    )
    ap.add_argument("--no-flash", dest="flash", action="store_false",
                    help="Skip arduino-cli compile + upload (use whatever "
                         "sketch is already on the board).")
    ap.set_defaults(flash=True)
    # Snap-to-pattern flags (mirror src.live_binary).
    ap.add_argument("--no-snap", dest="snap", action="store_false",
                    help="Disable snap-to-canonical-pattern (model can then "
                         "emit any of 32 raw bit combinations).")
    ap.set_defaults(snap=True)
    ap.add_argument("--no-snap-remap", dest="snap_remap", action="store_false",
                    help="With snap on, use the full 15 recorded patterns "
                         "instead of the 8 LABEL_REMAP-collapsed ones.")
    ap.set_defaults(snap_remap=True)
    # All inference smoothing flags come from live_binary so the two scripts
    # share one source of truth.
    add_inference_args(ap)
    return ap.parse_args()


def render_supported_strip(supported: list[str], active_bits: np.ndarray,
                            width: int, cell_h: int = 36, cols: int = 4,
                            title_h: int = 26) -> np.ndarray:
    """Strip listing the patterns the trained model can emit.

    The active row (matching the model's current raw bits, pre-invert) is
    highlighted so you can see at a glance which canonical class fired.
    """
    rows = (len(supported) + cols - 1) // cols
    height = title_h + rows * cell_h + 8
    strip = np.full((height, width, 3), 18, dtype=np.uint8)

    cv2.putText(strip, f"supported model outputs ({len(supported)})",
                (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)

    active = "".join(str(int(b)) for b in active_bits)
    cell_w = width // cols
    for i, pat in enumerate(supported):
        r, c = divmod(i, cols)
        x0 = c * cell_w + 6
        y0 = title_h + r * cell_h + 2
        x1 = x0 + cell_w - 12
        y1 = y0 + cell_h - 4
        is_on = pat == active
        bg = (40, 90, 40) if is_on else (35, 35, 35)
        fg = (220, 255, 220) if is_on else (210, 210, 210)
        cv2.rectangle(strip, (x0, y0), (x1, y1), bg, -1)
        cv2.rectangle(strip, (x0, y0), (x1, y1),
                      (90, 200, 90) if is_on else (80, 80, 80), 1)
        # Bit string with per-bit color so it reads at a glance.
        char_x = x0 + 10
        char_y = y0 + cell_h // 2 + 5
        for ch in pat:
            col = (90, 230, 90) if ch == "1" else (110, 110, 110)
            if is_on:
                col = (140, 255, 140) if ch == "1" else (180, 180, 180)
            cv2.putText(strip, ch, (char_x, char_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            char_x += 16
        # Inverted (Arduino-side) form, smaller, on the right.
        inv = "".join("1" if c == "0" else "0" for c in pat)
        cv2.putText(strip, f"->{inv}", (char_x + 6, char_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, fg, 1, cv2.LINE_AA)
    return strip


def _tty_to_cu(p: str) -> str:
    return "/dev/cu." + p[len("/dev/tty."):] if p.startswith("/dev/tty.") else p


def _release_serial(*paths: str) -> None:
    killed: list[int] = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            r = subprocess.run(["lsof", path], capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        for line in r.stdout.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            if pid == os.getpid() or pid in killed:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
    if killed:
        print(f"Released serial port (killed {killed})")
        time.sleep(0.35)


def _build_and_upload(port: str, fqbn: str) -> None:
    print(f"Compiling {SKETCH_DIR} ...", flush=True)
    r = subprocess.run(["arduino-cli", "compile", "--fqbn", fqbn, "."], cwd=SKETCH_DIR)
    if r.returncode != 0:
        raise SystemExit(f"arduino-cli compile failed (exit {r.returncode})")

    print(f"Uploading to {port} ...", flush=True)
    r = subprocess.run(
        ["arduino-cli", "upload", "-p", port, "--fqbn", fqbn, "."],
        cwd=SKETCH_DIR,
    )
    if r.returncode != 0:
        raise SystemExit(f"arduino-cli upload failed (exit {r.returncode})")
    time.sleep(1.5)  # let board finish auto-reset + setup()


def main():
    args = _parse_args()
    os.chdir(SCRIPT_DIR)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # 1. Free port, flash sketch (unless --no-flash), free port again, open serial.
    _release_serial(_tty_to_cu(args.port), args.port)
    if args.flash:
        _build_and_upload(args.port, args.fqbn)
        _release_serial(_tty_to_cu(args.port), args.port)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        if getattr(e, "errno", None) == 16 or "busy" in str(e).lower():
            print(
                "Serial port is busy. Quit Arduino Serial Monitor / other scripts.\n"
                f"Find who:  lsof {args.port}"
            )
        raise
    time.sleep(2)
    print(f"Serial open: {args.port}")
    _configure_arduino_lock_ms(ser, args.lock_s)

    # 2. Background reader -- print Arduino echoes.
    stop_reader = threading.Event()

    def reader():
        while not stop_reader.is_set():
            try:
                line = ser.readline()
            except (serial.SerialException, OSError):
                break
            if not line:
                continue
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                print(f"[arduino] {text}", flush=True)

    threading.Thread(target=reader, daemon=True).start()

    # 3. Predictor + EMG stream. Smoothing knobs default to
    #    src.live_binary.LIVE_INFERENCE_DEFAULTS but every one of them is
    #    overridable from the CLI (see add_inference_args). This script is
    #    a thin Arduino-bridge layer over the live inference pipeline.
    supported = (get_supported_patterns(remap=args.snap_remap)
                 if args.snap else None)
    if supported is not None:
        print(f"Snap-to-pattern: {len(supported)} canonical patterns "
              f"({'remap' if args.snap_remap else 'all 15'}): {supported}")
    else:
        print("Snap-to-pattern: disabled (--no-snap).")
    predictor = build_predictor(
        os.path.join(SCRIPT_DIR, args.model), None, device,
        supported_patterns=supported,
        **inference_overrides_from_args(args),
    )
    stream = EMGStream(synthetic=False, enable_filter=False)
    stream.start()

    time.sleep(1.0)
    warmup = stream.get_batch(max_items=4096)
    if len(warmup) < 250:
        stream.stop(); ser.close()
        raise SystemExit(f"EMG stream weak: {len(warmup)} samples in 1s. Check armband.")
    arr = np.asarray([s.emg for s in warmup], dtype=np.float32)
    if np.allclose(arr, 0.0):
        stream.stop(); ser.close()
        raise SystemExit("EMG warm-up all zeros -- electrodes not in contact.")
    print(f"Warm-up OK ({len(warmup)} samples).")
    # === IMU ADDITION === multistream checkpoints need IMU pushed too so the
    # IMU ring buffer is primed before the first predict() call.
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

    # 4. Global keyboard listener: 'c' -> stop, ESC -> quit.
    stop_flag = {"v": False}

    def send_park_default():
        """Park at Arduino DEFAULT_ANGLE before closing (cmd=D)."""
        try:
            ser.write(b"D")
            ser.flush()
            time.sleep(0.2)
        except Exception:
            pass

    def on_press(key):
        try:
            k = key.char.lower() if key.char else None
        except AttributeError:
            k = None
        if k == "c":
            stop_flag["v"] = True
            return False
        if key == keyboard.Key.esc:
            stop_flag["v"] = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Streaming EMG -> Arduino. Press 'c' to stop, ESC to quit.")
    hop_s = args.hop_ms / 1000.0
    lock_s: float | None = args.lock_s
    lock_label = (
        "none (host + firmware unlocked)"
        if lock_s is None
        else f"{lock_s}s"
    )
    print(f"Unified lock (--lock-s): {lock_label}")

    # Strip is only meaningful when snap-to-pattern is on; show whichever
    # set the predictor was actually built with.
    strip_supported = supported

    # Per-finger commanded state mirror (matches Arduino).
    # Start at 0 (rest), matching the Arduino's setup() where bit 0 = REST.
    commanded = np.zeros(5, dtype=np.int32)
    last_change = [0.0] * 5  # python-side cooldown to avoid serial spam

    # EMG bit polarity is inverted vs. our Arduino convention -- flip before send.
    INVERT_BITS = True

    try:
        while not stop_flag["v"]:
            t0 = time.time()
            bits, probs = predictor.predict()
            infer_ms = (time.time() - t0) * 1000

            # Build the next commanded bits, applying optional per-finger cooldown.
            next_cmd = commanded.copy()
            for i in range(5):
                raw = int(bits[i])
                target = (1 - raw) if INVERT_BITS else raw
                if target == commanded[i]:
                    continue
                if lock_s is not None and (t0 - last_change[i]) < lock_s:
                    continue
                next_cmd[i] = target
                last_change[i] = t0

            if not np.array_equal(next_cmd, commanded):
                commanded = next_cmd
                payload = b"F" + bytes("".join(str(int(b)) for b in commanded), "ascii")
                ser.write(payload)

            # Visualization. Read smoothing config off the predictor so the
            # on-screen footer always reflects the *actual* live values
            # (which come from live_binary.LIVE_INFERENCE_DEFAULTS or the
            # CLI overrides).
            panel = render_panel(
                bits, probs, args.hop_ms, infer_ms,
                predictor.on_thresh, predictor.off_thresh,
                ema_alpha=predictor.ema_alpha, median_k=predictor.median_k,
            )
            now = time.time()
            cmd_str = "".join(str(int(b)) for b in commanded)
            cv2.putText(
                panel, f"cmd {cmd_str}  ({os.path.basename(args.port)})",
                (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA,
            )
            # Per-finger Python-side lock countdown bar (only if --lock-s set).
            if lock_s is not None:
                for i in range(5):
                    remain = max(0.0, lock_s - (now - last_change[i]))
                    cx = 40 + (panel.shape[1] - 80) // 5 // 2 + i * ((panel.shape[1] - 80) // 5)
                    bar_y = 175
                    if remain > 0:
                        frac = remain / lock_s
                        cv2.rectangle(panel, (cx - 25, bar_y),
                                      (cx - 25 + int(50 * frac), bar_y + 5),
                                      (40, 80, 220), -1)
                    cv2.rectangle(panel, (cx - 25, bar_y), (cx + 25, bar_y + 5),
                                  (90, 90, 90), 1)

            if strip_supported is not None:
                strip = render_supported_strip(strip_supported, bits,
                                                width=panel.shape[1])
                display = cv2.vconcat([panel, strip])
            else:
                display = panel
            cv2.imshow("EMG -> Arduino per-finger (c/ESC/Ctrl+C=quit, parks on exit)", display)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord("c"):
                break

            sleep_left = hop_s - (time.time() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)
    finally:
        send_park_default()
        print("Parked at DEFAULT_ANGLE.")
        stop_reader.set()
        listener.stop()
        predictor.stop()
        stream.stop()
        cv2.destroyAllWindows()
        ser.close()
        print("Closed.")


if __name__ == "__main__":
    main()
