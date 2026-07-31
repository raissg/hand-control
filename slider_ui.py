"""
Slider UI: 5 sliders (0..180) -> 5 servo angles, one per finger.

Auto-compiles + uploads the sketch, then opens a tkinter window with one
slider per finger. Each slider directly sets the raw servo angle on the
Arduino (bypasses direction + the 2 s lockout -- calibration mode).

Buttons:
  STOP   -> detach all servos (no torque)
  REST   -> all sliders to 140
  ACTIVE -> all sliders to 20
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk

import serial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PORT = "/dev/cu.usbmodemCC8DA22028202"
BAUD = 115200
SKETCH_DIR = os.path.join(SCRIPT_DIR, "arduino", "hand_servos")
FQBN = "arduino:renesas_uno:unor4wifi"

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
PINS = [3, 5, 6, 9, 10]
SERVO_MAX_DEG = 300            # PDI-6225MG
INITIAL_ANGLE = 150            # mid of 0..300


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

    last_sent = [-1] * 5

    def send_angle(i: int, angle: int) -> None:
        a = max(0, min(SERVO_MAX_DEG, int(angle)))
        if a == last_sent[i]:
            return
        last_sent[i] = a
        try:
            ser.write(f"A{i}{a:03d}".encode("ascii"))
        except (serial.SerialException, OSError) as e:
            print(f"serial write failed: {e}")

    # Push the initial angle to every servo so the UI matches reality.
    for i in range(5):
        send_angle(i, INITIAL_ANGLE)

    root = tk.Tk()
    root.title("Hand servos")

    sliders: list[tk.Scale] = []
    vars_: list[tk.IntVar] = []

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack()

    for i, name in enumerate(FINGER_NAMES):
        col = tk.Frame(frame, padx=6)
        col.grid(row=0, column=i)
        tk.Label(col, text=f"{name}\npin {PINS[i]}", font=("Helvetica", 11, "bold")).pack()
        v = tk.IntVar(value=INITIAL_ANGLE)
        s = tk.Scale(
            col, from_=SERVO_MAX_DEG, to=0, length=380, variable=v,
            orient=tk.VERTICAL, resolution=1, showvalue=True,
            tickinterval=50,
            command=lambda val, idx=i: send_angle(idx, int(float(val))),
        )
        s.pack()
        sliders.append(s)
        vars_.append(v)

    btns = tk.Frame(root, pady=8)
    btns.pack()

    def set_all(angle: int):
        for i, v in enumerate(vars_):
            v.set(angle)
            send_angle(i, angle)

    def stop_servos():
        try:
            ser.write(b"S"); ser.flush()
            print("STOP sent -- servos detached.")
        except Exception:
            pass

    tk.Button(btns, text="REST (140)", width=12, command=lambda: set_all(140)).pack(side=tk.LEFT, padx=4)
    tk.Button(btns, text="ACTIVE (20)", width=12, command=lambda: set_all(20)).pack(side=tk.LEFT, padx=4)
    tk.Button(btns, text="MID (150)", width=12, command=lambda: set_all(150)).pack(side=tk.LEFT, padx=4)
    tk.Button(btns, text="MAX (300)", width=12, command=lambda: set_all(300)).pack(side=tk.LEFT, padx=4)
    tk.Button(btns, text="STOP (detach)", width=14, fg="red", command=stop_servos).pack(side=tk.LEFT, padx=4)

    def on_close():
        stop_servos()
        stop_reader.set()
        time.sleep(0.2)
        try:
            ser.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    print("Slider UI ready. Close window or press STOP to detach.")
    root.mainloop()


if __name__ == "__main__":
    main()
