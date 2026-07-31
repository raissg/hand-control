"""Auto-labeling EMG recorder for 5-bit finger-bend classifier.

Shows a target bit pattern on screen (e.g. ``10000`` = thumb bent, others
extended). You hold that pose while it records EMG for ``hold_s`` seconds,
then it switches to the next pattern. Every EMG sample is tagged with the
pattern shown at its timestamp.

Output: one CSV per session under ``recordings/bits_YYYYMMDD_HHMMSS/emg.csv``
with columns: ``timestamp, ch0..ch7, b0,b1,b2,b3,b4``.

Default protocol — 15 target patterns (thumb→pinky bits), each × --repeats:
    00000 11111 10000 01000 00100 11000 01100 10001 01110 11001
    11100 01111 10111 11011 11101

Usage:
    python -m src.record_bits
    python -m src.record_bits --hold 4 --rest 2 --repeats 3
    python -m src.record_bits --synthetic      # test without hardware
    python -m src.record_bits --patterns 00000,11111,01000   # custom set
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.acquire import EMGStream

DEFAULT_PATTERNS = [
    "00000",
    "11111",
    "10000",
    "01000",
    "11000",
    "01100",
    "11100"
]

# Calibration uses the SAME pattern set as training (DEFAULT_PATTERNS) so the
# head fine-tune sees the deployment distribution, not a synthetic one. The
# only thing --calibrate changes is timing: 1 rep + slightly longer hold so a
# new band session can be normalized + head fine-tuned in ~50s of user time.
CALIB_PATTERNS = DEFAULT_PATTERNS

# DEFAULT_PATTERNS = [
#     "00000", keep
#     "11111", keep
#     "10000", keep
#     "01000", keep
#     "11000",k
#     "01100",k
#     "10001", consider "10000"
#     "01110", consider "11111"
#     "11001", consider "11000"
#     "11100",k
#     "01111",consider "11111"
#     "10111",consider "11111"
#     "11011",consider "11111"
#     "11101",consider "11111"
# ]
FINGER_NAMES = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]


def render_panel(pattern: str, phase: str, t_left: float,
                 rep: int, total_rep: int, idx: int, n_patterns: int,
                 width: int = 720, height: int = 420) -> np.ndarray:
    """Full-screen pose prompt + countdown."""
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)

    # Header
    header = f"pattern {idx+1}/{n_patterns}   rep {rep}/{total_rep}   {phase}"
    cv2.putText(canvas, header, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

    # Big bit string
    big_text = " ".join(list(pattern))
    (tw, th), _ = cv2.getTextSize(big_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 4)
    cv2.putText(canvas, big_text, ((width - tw) // 2, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (240, 240, 240), 4, cv2.LINE_AA)

    # Finger LEDs
    radius = 28
    gap = (width - 80) // 5
    cy = 230
    for i, ch in enumerate(pattern):
        cx = 40 + gap // 2 + i * gap
        on = ch == "1"
        col = (0, 220, 0) if on else (55, 55, 55)
        cv2.circle(canvas, (cx, cy), radius, col, -1)
        cv2.circle(canvas, (cx, cy), radius, (220, 220, 220), 2)
        cv2.putText(canvas, FINGER_NAMES[i][:3],
                    (cx - 22, cy + radius + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Phase colour bar + countdown
    bar_y = height - 70
    bar_col = {"HOLD": (0, 200, 0), "REST": (0, 150, 220),
               "READY": (0, 180, 220)}.get(phase, (100, 100, 100))
    cv2.rectangle(canvas, (30, bar_y), (width - 30, bar_y + 30), bar_col, -1)
    cv2.putText(canvas, f"{phase}  {t_left:4.1f}s",
                (45, bar_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2, cv2.LINE_AA)

    cv2.putText(canvas, "ESC to abort   S to skip current pattern",
                (30, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return canvas


def parse_patterns(s: str) -> list[str]:
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        assert len(p) == 5 and set(p) <= {"0", "1"}, \
            f"pattern '{p}' must be 5 chars of 0/1"
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    # Defaults are None so we can pick different values for --calibrate mode
    # without overriding what the user explicitly passed.
    ap.add_argument("--hold", type=float, default=None,
                    help="Seconds to hold each pose (default 4.0; "
                         "with --calibrate: 5.0).")
    ap.add_argument("--rest", type=float, default=None,
                    help="Seconds of rest between poses, not recorded "
                         "(default 2.0; with --calibrate: 1.0).")
    ap.add_argument("--ready", type=float, default=1.5,
                    help="Pre-hold warning countdown (not recorded).")
    ap.add_argument("--repeats", type=int, default=None,
                    help="Cycles through pattern list (default 3; "
                         "with --calibrate: 1).")
    ap.add_argument("--patterns", default=None,
                    help="Comma-separated 5-bit patterns. Default: full "
                         "DEFAULT_PATTERNS set; with --calibrate: single-"
                         "finger CALIB_PATTERNS set.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Calibration mode: short ~50s recording with one "
                         "rep over single-finger patterns. Output goes to "
                         "recordings/calib_<timestamp>/. Pair with "
                         "`train_bits --resume <ckpt> --freeze-backbone`.")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out_dir", default=None,
                    help="Override output directory "
                         "(default: recordings/bits_<timestamp> or "
                         "recordings/calib_<timestamp> with --calibrate).")
    # === IMU ADDITION =====================================================
    # Off by default to keep CSV identical to legacy recordings. Pass --imu
    # to also persist accel + gyro columns for the two-stream model.
    # ======================================================================
    ap.add_argument("--imu", action="store_true",
                    help="Also record accel_x/y/z, gyro_x/y/z columns "
                         "(needed for --multistream training).")
    args = ap.parse_args()

    # Resolve mode-dependent defaults (only fills in what the user didn't pass).
    if args.calibrate:
        if args.hold is None:     args.hold = 5.0
        if args.rest is None:     args.rest = 1.0
        if args.repeats is None:  args.repeats = 1
        if args.patterns is None: args.patterns = ",".join(CALIB_PATTERNS)
        # Always record IMU during calibration. EMG-only training silently
        # ignores those columns; IMU-model fine-tune NEEDS them. Recording
        # them costs a few KB and removes a footgun.
        args.imu = True
        print("Calibration mode: short single-rep recording over "
              "CALIB_PATTERNS, IMU forced on.")
    else:
        if args.hold is None:     args.hold = 4.0
        if args.rest is None:     args.rest = 2.0
        if args.repeats is None:  args.repeats = 3
        if args.patterns is None: args.patterns = ",".join(DEFAULT_PATTERNS)

    patterns = parse_patterns(args.patterns)
    n_patterns = len(patterns)
    print(f"Patterns ({n_patterns}): {patterns}")
    print(f"Hold {args.hold}s | Rest {args.rest}s | Ready {args.ready}s | "
          f"{args.repeats} reps  -> {args.repeats * n_patterns * args.hold:.0f}s recorded.")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "calib" if args.calibrate else "bits"
    out_dir = Path(args.out_dir or os.path.join(root, "recordings",
                                                  f"{prefix}_{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "emg.csv"
    meta_path = out_dir / "meta.txt"

    # Meta
    with open(meta_path, "w") as f:
        f.write(f"created: {stamp}\n")
        f.write(f"hold_s: {args.hold}\n")
        f.write(f"rest_s: {args.rest}\n")
        f.write(f"ready_s: {args.ready}\n")
        f.write(f"repeats: {args.repeats}\n")
        f.write(f"patterns: {patterns}\n")
        f.write(f"synthetic: {args.synthetic}\n")
        f.write(f"imu: {args.imu}\n")  # === IMU ADDITION ===
        f.write(f"calibrate: {args.calibrate}\n")

    # Record RAW EMG — filtering is done once at training time in dataset_bits.
    # Avoids double-filtering and lets us pick the correct 50 Hz notch offline.
    stream = EMGStream(synthetic=args.synthetic, enable_filter=False)
    stream.start()

    # Sanity: refuse to record if the board is silent. Expect ~500 Hz sampling;
    # after 1.0 s we should see at least ~250 samples.
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
            f"and the armband is on your forearm. Aborting — no file written."
        )
    # Second check: non-zero signal (dead channels / unplugged electrodes).
    arr = np.asarray([s.emg for s in warmup], dtype=np.float32)
    if np.allclose(arr, 0.0):
        stream.stop()
        raise SystemExit(
            "EMG stream delivered only zeros during warm-up — electrodes "
            "may not be making contact. Aborting."
        )
    print(f"Warm-up OK: {len(warmup)} samples in {MIN_WARMUP_S:.1f}s, "
          f"per-channel std: {np.round(arr.std(axis=0), 2).tolist()}")

    window_name = "Auto-label recorder (ESC to abort)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 720, 420)

    rows = []                           # accumulate labeled rows
    aborted = False

    def drain_batch(label_bits):
        """Pull everything the stream has and label with the current bit pattern."""
        batch = stream.get_batch(max_items=4096)
        for s in batch:
            row = [f"{s.timestamp:.6f}"] + [f"{v:.4f}" for v in s.emg]
            # === IMU ADDITION =====================================
            # Only when --imu is passed, append accel + gyro columns
            # so the two-stream model can train on them later. Without
            # the flag, the CSV layout is identical to legacy recordings.
            # ======================================================
            if args.imu:
                row += [f"{v:.6f}" for v in s.accel]
                row += [f"{v:.6f}" for v in s.gyro]
            row += list(label_bits)
            rows.append(row)

    def run_phase(phase: str, duration: float, pattern: str,
                  record: bool, rep: int) -> bool:
        """Runs phase for ``duration``. Returns False if aborted."""
        nonlocal aborted
        t_end = time.time() + duration
        while True:
            t_left = t_end - time.time()
            if t_left <= 0:
                break
            panel = render_panel(pattern, phase, t_left, rep, args.repeats,
                                  run_phase.idx, n_patterns)
            cv2.imshow(window_name, panel)
            k = cv2.waitKey(10) & 0xFF
            if k == 27:        # ESC
                aborted = True
                return False
            if k == ord("s"):
                return True    # skip this phase
            if record:
                drain_batch(pattern)
            else:
                # drop incoming samples (don't let queue pile up)
                stream.get_batch(max_items=4096)
        return True

    run_phase.idx = 0

    try:
        for rep in range(1, args.repeats + 1):
            for idx, pat in enumerate(patterns):
                run_phase.idx = idx
                # READY (no record)
                if not run_phase("READY", args.ready, pat, record=False, rep=rep):
                    if aborted: raise KeyboardInterrupt
                # HOLD (record)
                if not run_phase("HOLD", args.hold, pat, record=True, rep=rep):
                    if aborted: raise KeyboardInterrupt
                # REST (no record) - next pattern if not last
                if idx < n_patterns - 1 or rep < args.repeats:
                    if not run_phase("REST", args.rest, pat,
                                      record=False, rep=rep):
                        if aborted: raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("Aborted.")

    stream.stop()
    cv2.destroyAllWindows()

    # === IMU ADDITION =========================================================
    # With --imu, columns are: timestamp, ch0..7, accel_x/y/z, gyro_x/y/z,
    # b0..b4. Without --imu, the layout is the legacy one (no IMU columns).
    # dataset_bits.py reads columns by name so both layouts coexist.
    # ==========================================================================
    header = ["timestamp"] + [f"ch{i}" for i in range(8)]
    if args.imu:
        header += ["accel_x", "accel_y", "accel_z",
                   "gyro_x", "gyro_y", "gyro_z"]
    header += [f"b{i}" for i in range(5)]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Saved {len(rows)} labeled EMG samples ({len(rows)/500:.1f}s) to {csv_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
