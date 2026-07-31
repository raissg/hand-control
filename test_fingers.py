"""
Sequential finger test. Auto-compiles + uploads the sketch, then for each
finger: moves it to active (angle 20) and waits for 'y' to advance to the
next finger. Press 'c' or ESC to stop and detach.

Order: thumb, index, middle, ring, pinky.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import serial
from pynput import keyboard

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PORT = "/dev/cu.usbmodemCC8DA22028202"
BAUD = 115200
SKETCH_DIR = os.path.join(SCRIPT_DIR, "arduino", "hand_servos")
FQBN = "arduino:renesas_uno:unor4wifi"

FINGER_NAMES  = ["thumb", "index", "middle", "ring", "pinky"]
PINS          = [3, 5, 6, 9, 10]
SERVO_MAX_DEG = 300
# Mirrors sketch REST_ANGLE / CLOSED_ANGLE.
REST_ANGLE    = [ 95, 174,  98, 100, 100]   # bit 0 (open / relaxed)
CLOSED_ANGLE  = [ 37,   0, 250, 260, 260]   # bit 1 (flexed)
LOCK_S        = 2.0


def report(bits: list[int]) -> None:
    print("    name    pin  bit  state    -> angle (deg)")
    for i, name in enumerate(FINGER_NAMES):
        angle = CLOSED_ANGLE[i] if bits[i] else REST_ANGLE[i]
        state = "closed" if bits[i] else "rest  "
        print(f"    {name:<7} {PINS[i]:>3}   {bits[i]}    {state}    -> {angle:>3}")


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

    stop_flag = {"v": False}
    advance_flag = {"v": False}

    def on_press(key):
        try:
            k = key.char.lower() if key.char else None
        except AttributeError:
            k = None
        if k == "c" or key == keyboard.Key.esc:
            stop_flag["v"] = True
            return False
        if k == "y":
            advance_flag["v"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    def send_bits(bits: list[int]) -> None:
        payload = b"F" + bytes("".join(str(b) for b in bits), "ascii")
        ser.write(payload)
        ser.flush()
        report(bits)

    def wait_for_advance() -> bool:
        """Block until 'y' is pressed. Returns False if user wants to stop."""
        advance_flag["v"] = False
        while not advance_flag["v"]:
            if stop_flag["v"]:
                return False
            time.sleep(0.05)
        return True

    def cooldown(since: float) -> bool:
        """Sleep until the per-servo lockout has elapsed since `since`."""
        remain = LOCK_S - (time.time() - since)
        while remain > 0:
            if stop_flag["v"]:
                return False
            time.sleep(min(0.1, remain))
            remain = LOCK_S - (time.time() - since)
        return True

    print("Test plan: each finger goes ACTIVE; press 'y' to move on.")
    print("Press 'c' or ESC to stop.\n")

    # Start at rest (sketch inits to all 0s, but be explicit).
    send_bits([0, 0, 0, 0, 0])
    time.sleep(0.5)

    try:
        for i, name in enumerate(FINGER_NAMES):
            if stop_flag["v"]:
                break
            print(f"--> Testing finger {i}: {name.upper()} -> CLOSED ({CLOSED_ANGLE[i]} deg).  Press 'y' for next.")
            bits = [0, 0, 0, 0, 0]
            bits[i] = 1
            send_bits(bits)
            t_active = time.time()

            if not wait_for_advance():
                break

            # Honor sketch lockout before sending the rest command.
            if not cooldown(t_active):
                break

            print(f"    {name} -> rest ({REST_ANGLE[i]} deg)")
            send_bits([0, 0, 0, 0, 0])
            t_rest = time.time()

            # Don't violate this finger's lockout on the next iteration either.
            if i < len(FINGER_NAMES) - 1:
                if not cooldown(t_rest):
                    break

        print("\nDone.")
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
