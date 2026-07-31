"""
Manual snap control for the pin 11 servo (sketch index 5).

  a   snap pin 11 to MAX_ANGLE
  d   snap pin 11 to MIN_ANGLE
  c / ESC   detach all servos and quit
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

SERVO_INDEX   = 5
PIN           = 11
DEFAULT_ANGLE = 100
MIN_ANGLE     =  70
MAX_ANGLE     = 200


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

    def send_angle(angle: int) -> None:
        a = max(MIN_ANGLE, min(MAX_ANGLE, int(angle)))
        try:
            ser.write(f"A{SERVO_INDEX}{a:03d}".encode("ascii"))
        except (serial.SerialException, OSError) as e:
            print(f"serial write failed: {e}")

    send_angle(DEFAULT_ANGLE)

    stop_flag = {"v": False}

    def on_press(key):
        try:
            k = key.char.lower() if key.char else None
        except AttributeError:
            k = None
        if k == "a":
            print(f"[a] -> {MAX_ANGLE}")
            send_angle(MAX_ANGLE)
            return
        if k == "d":
            print(f"[d] -> {MIN_ANGLE}")
            send_angle(MIN_ANGLE)
            return
        if k == "c" or key == keyboard.Key.esc:
            stop_flag["v"] = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print(f"Pin {PIN} snap control. a={MAX_ANGLE}  d={MIN_ANGLE}  default={DEFAULT_ANGLE}")
    print("c / ESC = stop & quit\n")

    try:
        while not stop_flag["v"]:
            time.sleep(0.05)
    finally:
        try:
            ser.write(b"S")
            ser.flush()
            print("STOP sent -- servos detached.")
        except Exception:
            pass
        stop_reader.set()
        listener.stop()
        time.sleep(0.2)
        ser.close()


if __name__ == "__main__":
    main()
