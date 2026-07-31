"""
Manual keyboard control for the 5 finger servos + the pin 11 extra servo.

Auto-compiles + uploads the sketch, then listens globally for:
  w   HOLD: step every finger toward CLOSED, stops each at its CLOSED_ANGLE
  s   HOLD: step every finger toward REST,  stops each at its REST_ANGLE
  a   HOLD: step pin 11 toward MAX_ANGLE_P11
  d   HOLD: step pin 11 toward MIN_ANGLE_P11
  c / ESC   detach all servos and quit

Sends raw 'A<i><3 digits>' commands so the per-finger 2 s lockout is bypassed.
Pin 11 motion is hard-clamped to [MIN_ANGLE_P11, MAX_ANGLE_P11].
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time

import serial
from pynput import keyboard

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PORT = "/dev/cu.usbmodemCC8DA22028202"
BAUD = 115200
SKETCH_DIR = os.path.join(SCRIPT_DIR, "arduino", "hand_servos")
FQBN = "arduino:renesas_uno:unor4wifi"

# ---------------- 5 fingers (indices 0..4) ---------------------------------
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
PINS         = [3, 5, 6, 9, 10]

# Mirrors the sketch (per-finger calibration, physical degrees, 0..300).
REST_ANGLE   = [ 95, 148, 124, 126, 126]
# 69, 148, 124, 126, 126
CLOSED_ANGLE = [ 37,   0, 250, 260, 260]

# ---------------- Pin 11 extra servo (index 5) -----------------------------
PIN11_INDEX     = 5
PIN11_PIN       = 11
DEFAULT_P11     = 100   # starting position for pin 11
MIN_ANGLE_P11   =  70   # 'd' (open) ramps down to this
MAX_ANGLE_P11   = 200   # 'a' (close) ramps up to this

# ---------------- Ramp speed (shared) --------------------------------------
TICK_HZ          = 60.0
STEP_DEG_PER_SEC = 120.0
STEP_DEG         = STEP_DEG_PER_SEC / TICK_HZ


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


def _build_and_upload() -> None:
    print(f"Compiling {SKETCH_DIR} ...", flush=True)
    r = subprocess.run(["arduino-cli", "compile", "--fqbn", FQBN, "."], cwd=SKETCH_DIR)
    if r.returncode != 0:
        raise SystemExit(f"arduino-cli compile failed (exit {r.returncode})")
    print(f"Uploading to {PORT} ...", flush=True)
    r = subprocess.run(
        ["arduino-cli", "upload", "-p", PORT, "--fqbn", FQBN, "."],
        cwd=SKETCH_DIR,
    )
    if r.returncode != 0:
        raise SystemExit(f"arduino-cli upload failed (exit {r.returncode})")
    time.sleep(1.5)


def main():
    _release_serial(_tty_to_cu(PORT), PORT)
    _build_and_upload()
    _release_serial(_tty_to_cu(PORT), PORT)

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print(f"Serial open: {PORT}")

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

    # Per-servo current angle (float for sub-degree integration).
    finger_current = [float(a) for a in REST_ANGLE]
    pin11_current  = float(DEFAULT_P11)
    last_sent = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, PIN11_INDEX: -1}

    def send_finger(i: int, angle: float) -> None:
        a = int(round(max(0.0, min(300.0, angle))))
        if a == last_sent[i]:
            return
        last_sent[i] = a
        try:
            ser.write(f"A{i}{a:03d}".encode("ascii"))
        except (serial.SerialException, OSError) as e:
            print(f"serial write failed: {e}")

    def send_pin11(angle: float) -> None:
        # Hard-clamp pin 11 motion to its allowed range.
        a = int(round(max(float(MIN_ANGLE_P11),
                          min(float(MAX_ANGLE_P11), angle))))
        if a == last_sent[PIN11_INDEX]:
            return
        last_sent[PIN11_INDEX] = a
        try:
            ser.write(f"A{PIN11_INDEX}{a:03d}".encode("ascii"))
        except (serial.SerialException, OSError) as e:
            print(f"serial write failed: {e}")

    # Initialize: fingers at REST, pin 11 at its default.
    for i in range(5):
        send_finger(i, finger_current[i])
    send_pin11(pin11_current)

    pressed: set[str] = set()
    stop_flag = {"v": False}

    def on_press(key):
        nonlocal pin11_current
        try:
            k = key.char.lower() if key.char else None
        except AttributeError:
            k = None
        if k in ("w", "s"):
            if k not in pressed:
                print(f"[{k}] press     fingers -> {'CLOSED' if k == 'w' else 'REST'}")
            pressed.add(k)
            return
        if k == "a":
            pin11_current = float(MAX_ANGLE_P11)
            send_pin11(pin11_current)
            print(f"[a] -> pin{PIN11_PIN}={MAX_ANGLE_P11}")
            return
        if k == "d":
            pin11_current = float(MIN_ANGLE_P11)
            send_pin11(pin11_current)
            print(f"[d] -> pin{PIN11_PIN}={MIN_ANGLE_P11}")
            return
        if k == "c" or key == keyboard.Key.esc:
            stop_flag["v"] = True
            return False

    def on_release(key):
        try:
            k = key.char.lower() if key.char else None
        except AttributeError:
            k = None
        if k in pressed:
            pressed.remove(k)
            if k in ("w", "s"):
                snapshot = [int(round(v)) for v in finger_current]
                print(f"[{k}] release   fingers={snapshot}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Manual control ready.")
    print("  w = hold to close fingers   s = hold to open fingers")
    print(f"  a = snap pin{PIN11_PIN} to {MAX_ANGLE_P11}   "
          f"d = snap pin{PIN11_PIN} to {MIN_ANGLE_P11}")
    print(f"  pin{PIN11_PIN} range [{MIN_ANGLE_P11}, {MAX_ANGLE_P11}], default {DEFAULT_P11}")
    print("  c / ESC = stop & quit\n")

    tick_dt = 1.0 / TICK_HZ
    arrived_finger_target = None   # 'CLOSED' / 'REST' / None
    try:
        while not stop_flag["v"]:
            t0 = time.time()

            # ---- Fingers (w/s) ramp ----
            f_target_label = None
            if "w" in pressed:
                f_target_label = "CLOSED"
            elif "s" in pressed:
                f_target_label = "REST"

            if f_target_label is not None:
                still_moving = False
                for i in range(5):
                    target = float(CLOSED_ANGLE[i] if f_target_label == "CLOSED"
                                   else REST_ANGLE[i])
                    diff = target - finger_current[i]
                    if abs(diff) <= STEP_DEG:
                        if finger_current[i] != target:
                            finger_current[i] = target
                            send_finger(i, finger_current[i])
                            still_moving = True
                    else:
                        finger_current[i] += STEP_DEG if diff > 0 else -STEP_DEG
                        send_finger(i, finger_current[i])
                        still_moving = True
                if not still_moving and arrived_finger_target != f_target_label:
                    snapshot = [int(round(v)) for v in finger_current]
                    print(f"      arrived  fingers={snapshot}")
                    arrived_finger_target = f_target_label
            else:
                arrived_finger_target = None

            # Pin 11 (a/d) is snap-on-press in on_press; nothing to do per-tick.

            sleep_left = tick_dt - (time.time() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)
    finally:
        try:
            ser.write(b"S"); ser.flush()
            print("STOP sent -- servos detached.")
        except Exception:
            pass
        stop_reader.set()
        listener.stop()
        time.sleep(0.2)
        ser.close()


if __name__ == "__main__":
    main()
