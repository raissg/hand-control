"""Run a repeatable, pre-cued video demonstration of the EMG hand decoder.

The normal ``go.py`` loop continuously mirrors whatever pose the model emits.
That is useful for development, but it is hard to film consistently: the
operator does not know which pose to perform next, the robot may move while the
operator is still transitioning, and there is no trial-level timing record.

This script turns the same live model and Arduino protocol into a deterministic
sequence of trials:

1. The operator first returns to the all-extended baseline.
2. A large on-screen cue reveals the next target pose.
3. The model keeps running during a visible ``cue_lead_s`` preparation period,
   while robot commands are intentionally gated.
4. Once the gate opens, the robot is commanded only if the *live model output*
   matches the target. The script never sends the target merely because it was
   scheduled.
5. Cue-to-detection time, command time, and hold stability are saved to CSV.

The preparation period makes a polished video look responsive because the
model's median/EMA/debounce pipeline can settle before the visible robot reveal.
It is choreography, not a latency benchmark. The display therefore carries a
persistent "PRE-CUED" label and reports both the cue lead and the real measured
times. Set ``--cue-lead-s 0`` when measuring unassisted response latency.

The hardware setup intentionally reuses ``go.py``'s compile/upload, serial-port,
and inference configuration helpers. This keeps the benchmark on the exact same
firmware and model path as the normal demo instead of creating a subtly
different second implementation.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial
import torch

# Running this file directly places the repository root on sys.path on most
# Python installations, but adding it explicitly also makes invocation from a
# different working directory predictable.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# These are internal helpers in go.py, but importing them here is deliberate:
# the benchmark should flash and configure the board exactly like the normal
# live bridge. Keeping one implementation prevents demo-only hardware drift.
from go import (  # noqa: E402
    DEFAULT_BAUD,
    DEFAULT_FQBN,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    _build_and_upload,
    _configure_arduino_lock_ms,
    _release_serial,
    _tty_to_cu,
)
from src.acquire import EMGStream  # noqa: E402
from src.live_binary import (  # noqa: E402
    add_inference_args,
    build_predictor,
    inference_overrides_from_args,
)
from src.record_bits import DEFAULT_PATTERNS, FINGER_NAMES  # noqa: E402


# The current recorder is the source of truth for poses this checkpoint can
# reasonably have learned. In particular, this avoids the stale legacy
# ``RECORDED_PATTERNS`` list used elsewhere, which still contains ``00100`` even
# though the current recording protocol never captures it.
CURRENT_PATTERN_SET = tuple(DEFAULT_PATTERNS)
DEFAULT_DEMO_PATTERNS = tuple(p for p in CURRENT_PATTERN_SET if p != "00000")

WINDOW_NAME = "Pre-cued EMG robot benchmark"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
BASELINE_PATTERN = "00000"


@dataclass
class TrialResult:
    """One CSV row with enough context to compare runs honestly.

    ``recognition_ms`` is measured from the visual cue, so it includes human
    reaction and hand-transition time in addition to model/smoothing delay. It
    must not be presented as neural-network inference latency. ``command_ms`` is
    when the serial command was dispatched; actual servo completion is not
    measured because the hand has no position sensor.
    """

    run_id: str
    trial: int
    target: str
    pose: str
    outcome: str
    cue_lead_ms: float
    recognition_ms: float | None
    command_ms: float | None
    target_score_at_command: float | None
    hold_match_rate: float | None
    last_prediction: str
    arduino_command: str
    model: str
    hop_ms: float
    median_k: int
    ema_alpha: float
    min_consecutive: int
    min_dwell_s: float


class ScreenRecorder:
    """Optionally save the generated UI as a synchronized MP4.

    This recording proves what the cue and model display showed, but it does not
    contain the physical robot unless an external camera also films the setup.
    OpenCV's widely available ``mp4v`` codec is used for portability.
    """

    def __init__(self, path: Path | None, fps: float):
        self.path = path
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._disabled = path is None

    def write(self, frame: np.ndarray) -> None:
        if self._disabled:
            return
        if self._writer is None:
            assert self.path is not None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.path),
                fourcc,
                self.fps,
                (frame.shape[1], frame.shape[0]),
            )
            if not self._writer.isOpened():
                print(
                    f"Warning: could not open screen recording {self.path}; "
                    "continuing without MP4 output.",
                    flush=True,
                )
                self._writer.release()
                self._writer = None
                self._disabled = True
                return
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class ArduinoHand:
    """Small serial wrapper that preserves ``go.py``'s bit convention.

    Model bits mean "finger bent" in the documentation. ``go.py`` currently
    complements every bit before sending it to the Arduino, so this benchmark
    defaults to the same behavior for an apples-to-apples demo. The conversion
    is explicit and displayed on screen because the firmware itself documents
    bit 1 as CLOSED. ``--no-invert-bits`` is available if the physical hand
    confirms that the extra complement is no longer required.
    """

    def __init__(self, ser: serial.Serial, invert_bits: bool):
        self.ser = ser
        self.invert_bits = invert_bits
        self.last_model_pose: str | None = None
        self.last_arduino_pose: str | None = None

    def encode(self, model_pose: str) -> str:
        if self.invert_bits:
            return "".join("0" if bit == "1" else "1" for bit in model_pose)
        return model_pose

    def send_model_pose(self, model_pose: str, *, force: bool = False) -> str:
        """Send one atomic five-finger command and return its Arduino bits."""
        if not force and model_pose == self.last_model_pose:
            return self.last_arduino_pose or self.encode(model_pose)
        arduino_pose = self.encode(model_pose)
        self.ser.write(b"F" + arduino_pose.encode("ascii"))
        self.ser.flush()
        self.last_model_pose = model_pose
        self.last_arduino_pose = arduino_pose
        print(
            f"[benchmark] model pose {model_pose} -> Arduino F{arduino_pose}",
            flush=True,
        )
        return arduino_pose

    def park(self) -> None:
        """Return all servos to firmware DEFAULT_ANGLE while keeping torque."""
        try:
            self.ser.write(b"D")
            self.ser.flush()
            time.sleep(0.2)
        except (serial.SerialException, OSError):
            # Cleanup should continue even if the board was unplugged mid-run.
            pass


def _parse_patterns(value: str) -> list[str]:
    """Validate a deterministic trial sequence against current training poses."""
    patterns = [item.strip() for item in value.split(",") if item.strip()]
    if not patterns:
        raise argparse.ArgumentTypeError("at least one pattern is required")
    for pattern in patterns:
        if len(pattern) != 5 or set(pattern) - {"0", "1"}:
            raise argparse.ArgumentTypeError(
                f"{pattern!r} must be exactly five 0/1 digits"
            )
        if pattern not in CURRENT_PATTERN_SET:
            raise argparse.ArgumentTypeError(
                f"{pattern!r} is not in the current recorder vocabulary "
                f"{list(CURRENT_PATTERN_SET)}"
            )
        if pattern == BASELINE_PATTERN:
            raise argparse.ArgumentTypeError(
                "00000 is reserved for the between-trial baseline"
            )
    return patterns


def _parse_args() -> argparse.Namespace:
    """Build CLI flags while inheriting all canonical inference controls."""
    parser = argparse.ArgumentParser(
        description=(
            "Pre-cued, model-driven EMG-to-robot demo with trial timing and CSV."
        )
    )

    # Keep hardware defaults aligned with go.py. No separate set of demo-only
    # firmware assumptions should emerge.
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--fqbn", default=DEFAULT_FQBN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--no-flash",
        dest="flash",
        action="store_false",
        help="Use the sketch already on the board instead of compiling/uploading.",
    )
    parser.set_defaults(flash=True)
    parser.add_argument(
        "--no-invert-bits",
        dest="invert_bits",
        action="store_false",
        help=(
            "Send model bits directly to firmware. By default this script "
            "matches go.py and complements them before sending."
        ),
    )
    parser.set_defaults(invert_bits=True)

    # The sequence intentionally excludes the baseline. The all-open pose is
    # inserted automatically before every measured trial so smoothing state
    # starts from the same place.
    parser.add_argument(
        "--patterns",
        type=_parse_patterns,
        default=list(DEFAULT_DEMO_PATTERNS),
        help=(
            "Comma-separated measured targets. Current valid non-baseline "
            f"patterns: {list(DEFAULT_DEMO_PATTERNS)}."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--cue-lead-s",
        type=float,
        default=0.9,
        help=(
            "Visible preparation time before robot commands are allowed. "
            "Use 0 for an unassisted response benchmark."
        ),
    )
    parser.add_argument(
        "--rest-s",
        type=float,
        default=1.5,
        help="Minimum all-open baseline time before each target is revealed.",
    )
    parser.add_argument(
        "--baseline-timeout-s",
        type=float,
        default=5.0,
        help="Abort if the smoothed model cannot return to 00000 in this time.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=4.0,
        help="Maximum time from cue until a matching live prediction.",
    )
    parser.add_argument(
        "--hold-s",
        type=float,
        default=1.5,
        help="Keep the robot latched after a match and score model stability.",
    )
    parser.add_argument(
        "--failure-s",
        type=float,
        default=1.5,
        help="How long to display a missed-trial result.",
    )
    parser.add_argument(
        "--summary-s",
        type=float,
        default=6.0,
        help="How long the final benchmark summary remains on screen.",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=50.0,
        help="Prediction/display period. 50 ms means 20 Hz.",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark_results",
        help="Directory for CSV and optional screen recording.",
    )
    parser.add_argument(
        "--record-screen",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "Record the generated UI to MP4. With no PATH, save beside the CSV. "
            "Use an external camera as well to capture the physical hand."
        ),
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Show a resizable window instead of the default full-screen cue.",
    )

    # The benchmark defaults to snap against exactly the current recorder
    # vocabulary. This avoids allowing a trial result outside the set on which
    # the current model could have been trained.
    parser.add_argument(
        "--no-snap",
        dest="snap",
        action="store_false",
        help="Disable snap-to-current-pattern vocabulary.",
    )
    parser.set_defaults(snap=True)

    add_inference_args(parser)
    args = parser.parse_args()

    for name in (
        "cue_lead_s",
        "rest_s",
        "baseline_timeout_s",
        "timeout_s",
        "hold_s",
        "failure_s",
        "summary_s",
        "hop_ms",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.hop_ms == 0:
        parser.error("--hop-ms must be > 0")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.timeout_s < args.cue_lead_s:
        parser.error("--timeout-s must be >= --cue-lead-s")
    if args.baseline_timeout_s < args.rest_s:
        parser.error("--baseline-timeout-s must be >= --rest-s")
    return args


def _pattern_string(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in bits)


def _pose_name(pattern: str) -> str:
    """Turn a bit string into a phrase readable while moving a hand."""
    bent = [FINGER_NAMES[index].title() for index, bit in enumerate(pattern) if bit == "1"]
    if not bent:
        return "Open hand"
    if len(bent) == len(FINGER_NAMES):
        return "Close all fingers"
    return "Bend " + " + ".join(bent)


def _target_score(pattern: str, probs: np.ndarray) -> float:
    """Mean probability assigned to the target's five requested bit values."""
    per_bit = [
        float(prob) if target == "1" else 1.0 - float(prob)
        for target, prob in zip(pattern, probs)
    ]
    return float(np.mean(per_bit))


def _centered_text(
    canvas: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw one ASCII line centered horizontally in an OpenCV canvas."""
    (width, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = max(20, (canvas.shape[1] - width) // 2)
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_finger_cards(
    canvas: np.ndarray,
    target: str,
    predicted: str,
    probs: np.ndarray,
    *,
    y: int = 210,
) -> None:
    """Render target instructions and live probabilities finger by finger."""
    margin = 55
    gap = 14
    card_width = (canvas.shape[1] - 2 * margin - 4 * gap) // 5
    card_height = 260

    for index, name in enumerate(FINGER_NAMES):
        x0 = margin + index * (card_width + gap)
        x1 = x0 + card_width
        target_on = target[index] == "1"
        predicted_on = predicted[index] == "1"

        # Green cards are fingers the operator should bend; gray cards are
        # fingers that should remain straight. The live-prediction border turns
        # green only when that bit currently agrees with the target.
        background = (35, 82, 42) if target_on else (40, 40, 44)
        border = (55, 220, 90) if target[index] == predicted[index] else (60, 80, 235)
        cv2.rectangle(canvas, (x0, y), (x1, y + card_height), background, -1)
        cv2.rectangle(canvas, (x0, y), (x1, y + card_height), border, 4)

        _card_text = name[:3]
        (tw, _), _ = cv2.getTextSize(
            _card_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        cv2.putText(
            canvas,
            _card_text,
            (x0 + (card_width - tw) // 2, y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (225, 225, 225),
            2,
            cv2.LINE_AA,
        )

        instruction = "BEND" if target_on else "STRAIGHT"
        (tw, _), _ = cv2.getTextSize(
            instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
        )
        cv2.putText(
            canvas,
            instruction,
            (x0 + (card_width - tw) // 2, y + 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (145, 255, 165) if target_on else (190, 190, 190),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"target {target[index]}",
            (x0 + 18, y + 127),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"model  {predicted[index]}",
            (x0 + 18, y + 157),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (130, 245, 160) if target[index] == predicted[index] else (110, 150, 255),
            1,
            cv2.LINE_AA,
        )

        # A probability bar gives engineers more information than the binary
        # output while remaining readable in a video frame.
        bar_x0 = x0 + 18
        bar_x1 = x1 - 18
        bar_y0 = y + 187
        bar_y1 = bar_y0 + 18
        cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1), (25, 25, 25), -1)
        fill_x = bar_x0 + int((bar_x1 - bar_x0) * float(np.clip(probs[index], 0, 1)))
        cv2.rectangle(canvas, (bar_x0, bar_y0), (fill_x, bar_y1), (70, 205, 100), -1)
        cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1), (115, 115, 115), 1)
        cv2.putText(
            canvas,
            f"P(bent) {probs[index] * 100:5.1f}%",
            (x0 + 18, y + 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def _render_trial(
    *,
    phase: str,
    trial: int,
    total_trials: int,
    target: str,
    predicted: str,
    probs: np.ndarray,
    headline: str,
    subhead: str,
    cue_lead_s: float,
    recognition_ms: float | None = None,
    command_ms: float | None = None,
    hold_match_rate: float | None = None,
    arduino_command: str = "",
) -> np.ndarray:
    """Build the single frame seen by the operator and captured in the video."""
    canvas = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (18, 18, 21), dtype=np.uint8)
    phase_colors = {
        "BASELINE": (175, 105, 20),
        "CUE": (20, 155, 225),
        "LIVE": (45, 185, 220),
        "MATCH": (45, 175, 70),
        "MISSED": (45, 55, 210),
    }
    cv2.rectangle(
        canvas,
        (0, 0),
        (FRAME_WIDTH, 72),
        phase_colors.get(phase, (75, 75, 75)),
        -1,
    )
    cv2.putText(
        canvas,
        f"{phase}   trial {trial}/{total_trials}",
        (32, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (12, 12, 12),
        2,
        cv2.LINE_AA,
    )
    # This disclosure stays visible in every trial frame. A pre-cued clip must
    # not be mistaken for a zero-latency reaction measurement.
    disclosure = (
        f"PRE-CUED MODEL DEMO | visible lead {cue_lead_s:.2f}s"
        if cue_lead_s > 0
        else "ZERO-LEAD RESPONSE BENCHMARK"
    )
    (tw, _), _ = cv2.getTextSize(
        disclosure, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    cv2.putText(
        canvas,
        disclosure,
        (FRAME_WIDTH - tw - 30, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )

    _centered_text(canvas, headline, 122, 1.15, (245, 245, 245), 3)
    _centered_text(canvas, subhead, 164, 0.62, (190, 205, 220), 1)
    _draw_finger_cards(canvas, target, predicted, probs)

    recognition = (
        "detection: waiting"
        if recognition_ms is None
        else f"detection: +{recognition_ms:.0f} ms from cue"
    )
    command = (
        "robot command: gated"
        if command_ms is None
        else f"robot command: +{command_ms:.0f} ms  F{arduino_command}"
    )
    cv2.putText(
        canvas,
        f"target {target}   model {predicted}   score {_target_score(target, probs):.3f}",
        (55, 515),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (225, 225, 225),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        recognition,
        (55, 558),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (155, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        command,
        (520, 558),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (155, 220, 255),
        1,
        cv2.LINE_AA,
    )
    if hold_match_rate is not None:
        cv2.putText(
            canvas,
            f"hold stability: {hold_match_rate * 100:5.1f}% matching frames",
            (55, 596),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (160, 245, 170),
            1,
            cv2.LINE_AA,
        )

    cv2.line(canvas, (45, 630), (FRAME_WIDTH - 45, 630), (65, 65, 70), 1)
    cv2.putText(
        canvas,
        "Measured target moves only after live model output equals target. Q / ESC / C stops and parks.",
        (55, 668),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (155, 155, 165),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _render_summary(results: list[TrialResult], cue_lead_s: float) -> np.ndarray:
    """Show the outcomes long enough for a clean final shot."""
    canvas = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (18, 18, 21), dtype=np.uint8)
    successes = [result for result in results if result.outcome == "matched"]
    recognition_values = [
        result.recognition_ms
        for result in successes
        if result.recognition_ms is not None
    ]
    command_values = [
        result.command_ms for result in successes if result.command_ms is not None
    ]
    hold_values = [
        result.hold_match_rate
        for result in successes
        if result.hold_match_rate is not None
    ]

    _centered_text(canvas, "BENCHMARK COMPLETE", 90, 1.35, (235, 235, 235), 3)
    _centered_text(
        canvas,
        (
            f"PRE-CUED ({cue_lead_s:.2f}s visible lead)"
            if cue_lead_s > 0
            else "ZERO-LEAD"
        ),
        135,
        0.62,
        (120, 205, 255),
        2,
    )
    _centered_text(
        canvas,
        f"{len(successes)}/{len(results)} targets matched",
        220,
        1.1,
        (115, 240, 145) if len(successes) == len(results) else (105, 160, 255),
        3,
    )

    median_recognition = (
        float(np.median(recognition_values)) if recognition_values else None
    )
    p90_recognition = (
        float(np.percentile(recognition_values, 90)) if recognition_values else None
    )
    median_command = float(np.median(command_values)) if command_values else None
    p90_command = (
        float(np.percentile(command_values, 90)) if command_values else None
    )
    mean_hold = float(np.mean(hold_values)) if hold_values else None
    rows = [
        (
            "Median cue-to-detection",
            "n/a" if median_recognition is None else f"{median_recognition:.0f} ms",
        ),
        (
            "P90 cue-to-detection",
            "n/a" if p90_recognition is None else f"{p90_recognition:.0f} ms",
        ),
        (
            "Median cue-to-command",
            "n/a" if median_command is None else f"{median_command:.0f} ms",
        ),
        (
            "P90 cue-to-command",
            "n/a" if p90_command is None else f"{p90_command:.0f} ms",
        ),
        (
            "Mean hold stability",
            "n/a" if mean_hold is None else f"{mean_hold * 100:.1f}%",
        ),
    ]
    for row, (label, value) in enumerate(rows):
        y = 270 + row * 55
        cv2.putText(
            canvas,
            label,
            (315, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (190, 190, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            value,
            (810, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (230, 230, 235),
            2,
            cv2.LINE_AA,
        )

    _centered_text(
        canvas,
        "Cue-to-detection includes the person's reaction and movement.",
        560,
        0.57,
        (150, 175, 205),
        1,
    )
    _centered_text(
        canvas,
        "It is not the neural network's sub-millisecond forward-pass time.",
        596,
        0.57,
        (150, 175, 205),
        1,
    )
    _centered_text(canvas, "Q / ESC / C to close", 670, 0.53, (145, 145, 155), 1)
    return canvas


def _show_frame(frame: np.ndarray, recorder: ScreenRecorder) -> bool:
    """Display/record one frame and return True when the operator aborts."""
    recorder.write(frame)
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord("q"), ord("c"))


def _pace(loop_started: float, hop_s: float) -> None:
    """Keep prediction cadence stable without accumulating schedule drift."""
    remaining = hop_s - (time.monotonic() - loop_started)
    if remaining > 0:
        time.sleep(remaining)


def _start_arduino_reader(
    ser: serial.Serial,
) -> tuple[threading.Event, threading.Thread]:
    """Print firmware acknowledgements without blocking the visual loop."""
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                line = ser.readline()
            except (serial.SerialException, OSError):
                return
            if line:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    print(f"[arduino] {text}", flush=True)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return stop, thread


def _open_arduino(args: argparse.Namespace) -> serial.Serial:
    """Use the same port-release and firmware-upload sequence as go.py."""
    _release_serial(_tty_to_cu(args.port), args.port)
    if args.flash:
        _build_and_upload(args.port, args.fqbn)
        _release_serial(_tty_to_cu(args.port), args.port)
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        if getattr(exc, "errno", None) == 16 or "busy" in str(exc).lower():
            print(
                "Serial port is busy. Quit Arduino Serial Monitor / other scripts.\n"
                f"Inspect the owner with: lsof {args.port}",
                flush=True,
            )
        raise
    time.sleep(2.0)
    # Benchmark commands must execute when dispatched. Explicitly disabling the
    # firmware's per-finger lock prevents a previous trial from silently
    # rejecting the next command and invalidating the recorded command time.
    _configure_arduino_lock_ms(ser, None)
    return ser


def _prime_predictor(predictor, stream: EMGStream) -> None:
    """Require real, non-zero armband data and fill the predictor buffers."""
    stream.start()
    time.sleep(1.0)
    warmup = stream.get_batch(max_items=4096)
    if len(warmup) < 250:
        raise RuntimeError(
            f"EMG stream weak: {len(warmup)} samples in 1 second; "
            "check armband power, Wi-Fi, and electrode contact."
        )
    emg = np.asarray([sample.emg for sample in warmup], dtype=np.float32)
    if np.allclose(emg, 0.0):
        raise RuntimeError(
            "EMG warm-up was all zeros; electrodes are probably not in contact."
        )
    print(
        f"Warm-up OK: {len(warmup)} samples; "
        f"channel std={np.round(emg.std(axis=0), 2).tolist()}",
        flush=True,
    )

    if predictor.multistream:
        predictor.push_emg_imu(
            [sample.emg.astype(np.float32) for sample in warmup],
            [
                np.concatenate(
                    [
                        sample.accel.astype(np.float32),
                        sample.gyro.astype(np.float32),
                    ]
                )
                for sample in warmup
            ],
        )
    else:
        predictor.push([sample.emg.astype(np.float32) for sample in warmup])
    predictor.start(stream)


def _wait_for_baseline(
    *,
    predictor,
    robot: ArduinoHand,
    recorder: ScreenRecorder,
    trial: int,
    total_trials: int,
    args: argparse.Namespace,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Return the operator and smoothing state to a comparable all-open start.

    The target remains hidden in this phase, so the cue onset in the next phase
    is unambiguous on video. At least ``rest_s`` is always spent at baseline;
    after that, the phase ends once the *published* model output is 00000.
    """
    robot.send_model_pose(BASELINE_PATTERN, force=True)
    started = time.monotonic()
    last_bits = np.zeros(5, dtype=np.int32)
    last_probs = np.zeros(5, dtype=np.float32)

    while True:
        loop_started = time.monotonic()
        last_bits, last_probs = predictor.predict()
        elapsed = loop_started - started
        predicted = _pattern_string(last_bits)
        baseline_ready = elapsed >= args.rest_s and predicted == BASELINE_PATTERN
        timed_out = elapsed >= args.baseline_timeout_s
        remaining = max(0.0, args.rest_s - elapsed)
        frame = _render_trial(
            phase="BASELINE",
            trial=trial,
            total_trials=total_trials,
            target=BASELINE_PATTERN,
            predicted=predicted,
            probs=last_probs,
            headline="OPEN AND RELAX YOUR HAND",
            subhead=(
                f"Next target stays hidden; baseline minimum {remaining:.1f}s"
                if remaining > 0
                else "Waiting for the smoothed model to publish 00000"
            ),
            cue_lead_s=args.cue_lead_s,
        )
        if _show_frame(frame, recorder):
            return False, last_bits, last_probs
        if baseline_ready:
            return True, last_bits, last_probs
        if timed_out:
            print(
                "Baseline timeout: model did not return to 00000. "
                "Stopping because later trial timings would not be comparable.",
                flush=True,
            )
            return False, last_bits, last_probs
        _pace(loop_started, args.hop_ms / 1000.0)


def _run_trial(
    *,
    predictor,
    robot: ArduinoHand,
    recorder: ScreenRecorder,
    target: str,
    trial: int,
    total_trials: int,
    run_id: str,
    args: argparse.Namespace,
) -> tuple[TrialResult, bool]:
    """Cue one target and command it only after the live model agrees."""
    cue_started = time.monotonic()
    recognition_ms: float | None = None
    command_ms: float | None = None
    command_score: float | None = None
    arduino_command = ""
    last_bits = np.zeros(5, dtype=np.int32)
    last_probs = np.zeros(5, dtype=np.float32)
    aborted = False
    outcome = "missed"

    while True:
        loop_started = time.monotonic()
        elapsed = loop_started - cue_started
        last_bits, last_probs = predictor.predict()
        predicted = _pattern_string(last_bits)
        matches = predicted == target

        if matches and recognition_ms is None:
            recognition_ms = elapsed * 1000.0

        gate_open = elapsed >= args.cue_lead_s
        if gate_open and matches:
            # Crucially, this sends the target because the live model currently
            # equals it, never merely because the trial schedule requested it.
            arduino_command = robot.send_model_pose(target, force=True)
            command_ms = (time.monotonic() - cue_started) * 1000.0
            command_score = _target_score(target, last_probs)
            outcome = "matched"
            break

        if elapsed >= args.timeout_s:
            break

        if gate_open:
            phase = "LIVE"
            headline = "KEEP HOLDING - WAITING FOR MODEL"
            subhead = "Robot gate is open; command dispatches on an exact target match"
        else:
            phase = "CUE"
            headline = "MOVE NOW: " + _pose_name(target).upper()
            subhead = (
                f"Robot command gate opens in "
                f"{max(0.0, args.cue_lead_s - elapsed):.1f}s"
            )
            if recognition_ms is not None:
                subhead += " | target already detected"

        frame = _render_trial(
            phase=phase,
            trial=trial,
            total_trials=total_trials,
            target=target,
            predicted=predicted,
            probs=last_probs,
            headline=headline,
            subhead=subhead,
            cue_lead_s=args.cue_lead_s,
            recognition_ms=recognition_ms,
        )
        if _show_frame(frame, recorder):
            aborted = True
            break
        _pace(loop_started, args.hop_ms / 1000.0)

    hold_match_rate: float | None = None
    if outcome == "matched" and not aborted:
        hold_started = time.monotonic()
        match_frames = 0
        total_frames = 0
        while time.monotonic() - hold_started < args.hold_s:
            loop_started = time.monotonic()
            last_bits, last_probs = predictor.predict()
            predicted = _pattern_string(last_bits)
            total_frames += 1
            match_frames += int(predicted == target)
            hold_match_rate = match_frames / total_frames
            remaining = max(0.0, args.hold_s - (loop_started - hold_started))
            frame = _render_trial(
                phase="MATCH",
                trial=trial,
                total_trials=total_trials,
                target=target,
                predicted=predicted,
                probs=last_probs,
                headline="MATCH - ROBOT COMMANDED",
                subhead=f"Keep holding for stability score: {remaining:.1f}s",
                cue_lead_s=args.cue_lead_s,
                recognition_ms=recognition_ms,
                command_ms=command_ms,
                hold_match_rate=hold_match_rate,
                arduino_command=arduino_command,
            )
            if _show_frame(frame, recorder):
                aborted = True
                break
            _pace(loop_started, args.hop_ms / 1000.0)
    elif not aborted:
        failure_started = time.monotonic()
        while time.monotonic() - failure_started < args.failure_s:
            loop_started = time.monotonic()
            last_bits, last_probs = predictor.predict()
            predicted = _pattern_string(last_bits)
            frame = _render_trial(
                phase="MISSED",
                trial=trial,
                total_trials=total_trials,
                target=target,
                predicted=predicted,
                probs=last_probs,
                headline="NO MATCH - ROBOT NOT COMMANDED",
                subhead=(
                    "This trial is recorded as a miss; the script does not fake the pose"
                ),
                cue_lead_s=args.cue_lead_s,
                recognition_ms=recognition_ms,
            )
            if _show_frame(frame, recorder):
                aborted = True
                break
            _pace(loop_started, args.hop_ms / 1000.0)

    result = TrialResult(
        run_id=run_id,
        trial=trial,
        target=target,
        pose=_pose_name(target),
        outcome="aborted" if aborted else outcome,
        cue_lead_ms=args.cue_lead_s * 1000.0,
        recognition_ms=recognition_ms,
        command_ms=command_ms,
        target_score_at_command=command_score,
        hold_match_rate=hold_match_rate,
        last_prediction=_pattern_string(last_bits),
        arduino_command=arduino_command,
        model=args.model,
        hop_ms=args.hop_ms,
        median_k=predictor.median_k,
        ema_alpha=predictor.ema_alpha,
        min_consecutive=predictor.min_consecutive,
        min_dwell_s=predictor.min_dwell_s,
    )
    return result, aborted


def main() -> None:
    args = _parse_args()
    os.chdir(SCRIPT_DIR)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path
    if not model_path.is_file():
        raise SystemExit(
            f"Model checkpoint not found: {model_path}\n"
            "The Git clone does not currently contain a models/ directory. "
            "Copy a trained checkpoint locally and pass it with --model."
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = SCRIPT_DIR / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"demo_benchmark_{run_id}.csv"
    if args.record_screen == "auto":
        video_path: Path | None = results_dir / f"demo_benchmark_{run_id}.mp4"
    elif args.record_screen:
        video_path = Path(args.record_screen)
        if not video_path.is_absolute():
            video_path = SCRIPT_DIR / video_path
    else:
        video_path = None

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Trials: {args.patterns} x {args.repeats}")
    print(
        f"Mode: {'PRE-CUED' if args.cue_lead_s > 0 else 'ZERO-LEAD'}; "
        f"cue lead {args.cue_lead_s:.2f}s"
    )
    print(f"CSV: {csv_path}")

    ser: serial.Serial | None = None
    reader_stop: threading.Event | None = None
    reader_thread: threading.Thread | None = None
    stream: EMGStream | None = None
    predictor = None
    robot: ArduinoHand | None = None
    recorder = ScreenRecorder(video_path, fps=1000.0 / args.hop_ms)
    results: list[TrialResult] = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.windowed:
        cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
    else:
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    try:
        ser = _open_arduino(args)
        reader_stop, reader_thread = _start_arduino_reader(ser)
        robot = ArduinoHand(ser, invert_bits=args.invert_bits)

        # Snap only to poses present in the current recorder, avoiding the stale
        # ``00100`` vocabulary mismatch documented in the architecture notes.
        supported_patterns = list(CURRENT_PATTERN_SET) if args.snap else None
        predictor = build_predictor(
            str(model_path),
            None,
            device,
            supported_patterns=supported_patterns,
            **inference_overrides_from_args(args),
        )
        stream = EMGStream(synthetic=False, enable_filter=False)
        _prime_predictor(predictor, stream)

        sequence = args.patterns * args.repeats
        total_trials = len(sequence)
        fieldnames = list(TrialResult.__dataclass_fields__.keys())
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_file.flush()

            for trial, target in enumerate(sequence, start=1):
                baseline_ok, _, _ = _wait_for_baseline(
                    predictor=predictor,
                    robot=robot,
                    recorder=recorder,
                    trial=trial,
                    total_trials=total_trials,
                    args=args,
                )
                if not baseline_ok:
                    break

                result, aborted = _run_trial(
                    predictor=predictor,
                    robot=robot,
                    recorder=recorder,
                    target=target,
                    trial=trial,
                    total_trials=total_trials,
                    run_id=run_id,
                    args=args,
                )
                results.append(result)
                writer.writerow(asdict(result))
                csv_file.flush()
                print(
                    f"Trial {trial}/{total_trials}: {target} {result.outcome}; "
                    f"detect={result.recognition_ms}ms "
                    f"command={result.command_ms}ms "
                    f"hold={result.hold_match_rate}",
                    flush=True,
                )
                if aborted:
                    break

        # A stable summary frame makes the end of a one-take video useful and
        # gives the operator time to stop the external camera cleanly.
        summary_started = time.monotonic()
        while results and time.monotonic() - summary_started < args.summary_s:
            frame = _render_summary(results, args.cue_lead_s)
            if _show_frame(frame, recorder):
                break
            time.sleep(min(0.05, args.summary_s))
    except KeyboardInterrupt:
        print("Interrupted by operator.", flush=True)
    finally:
        if robot is not None:
            robot.park()
            print("Robot parked at firmware DEFAULT_ANGLE.", flush=True)
        if reader_stop is not None:
            reader_stop.set()
        if predictor is not None:
            predictor.stop()
        if stream is not None:
            stream.stop()
        recorder.close()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()
        if reader_thread is not None:
            reader_thread.join(timeout=1.0)

    print(f"Results written to {csv_path}")
    if video_path is not None:
        print(f"Screen recording requested at {video_path}")


if __name__ == "__main__":
    main()
