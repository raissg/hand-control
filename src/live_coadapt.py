"""Live co-adaptive calibration.

Cued gesture protocol like ``record_bits.py``, but the model also fine-tunes
on the labeled stream while you do the gestures. UI:

    +--------------------------+--------------------------+
    |   MODEL PREDICTS         |   DO THIS                |
    |                          |                          |
    |   THUMB   [#####.....]   |   1 0 1 1 0              |
    |   INDEX   [..........]   |   (LED row)              |
    |   MIDDLE  [#####.....]                              |
    |   RING    [#######...]                              |
    |   PINKY   [..........]                              |
    +--------------------------+--------------------------+
       phase bar (READY/HOLD/REST)   countdown

Pipeline per 50 ms tick:
  1. Drain the EMG stream into a ring buffer.
  2. Take the latest 400 ms window, bandpass + notch filter, normalize.
  3. Forward pass through the model → 5 probabilities, drawn on the LHS
     as LEDs (after display-side smoothing: median → EMA → hysteresis).
  4. If we're in HOLD phase and past the trim window, push (window, label)
     into a replay buffer.

Every ``--steps_per_sec`` ticks, draw a mini-batch from the replay buffer
and take one SGD step on the model (head only by default). The next
inference frame uses the freshly updated weights — the display-side
smoothing pipeline (median K + EMA on probabilities + hysteresis) is
what keeps the LEDs visually stable.

This is *additive*: it does not modify ``record_bits.py``, ``live_binary.py``,
``binary_model.py``, or any checkpoint. The base ``--model`` is loaded
read-only; the adapted weights are saved to a new path via ``--save``.

Usage:
    # full co-adapt session, save adapted weights:
    python -m src.live_coadapt \
        --model models/model_20260425_202900.pt \
        --save  models/model_20260425_202900_adapted.pt

    # biofeedback only (no weight updates), useful as a poster demo:
    python -m src.live_coadapt --model models/model_20260425_202900.pt --no_train

    # short demo set, no hardware:
    python -m src.live_coadapt --synthetic --hold 2 --rest 1 --repeats 1 \
        --patterns 00000,11111,10000,01000
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from src.acquire import EMGStream
from src.binary_model import EMG5Bit
from src.convert_to_hdf5 import filter_emg

# Mirror record_bits.py defaults so the protocol is familiar.
DEFAULT_PATTERNS = [
    "00000", "10000", "01000", "11000",
]
FINGER_NAMES = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
SR = 500
HOP_SAMPLES = 25  # 50 ms hop @ 500 Hz → 20 Hz output rate


# ──────────────────────────────────────────────────────────────────────────
# Model loading + freezing
# ──────────────────────────────────────────────────────────────────────────

def load_model(path: str, device):
    """Load checkpoint. Returns (model, mean, std, window). Norm stats embedded."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EMG5Bit(in_channels=8, n_out=5).to(device)
    model.load_state_dict(ckpt["state_dict"])
    mean = np.asarray(ckpt["input_mean"], dtype=np.float32)[:8]
    std = np.asarray(ckpt["input_std"], dtype=np.float32)[:8]
    window = int(ckpt["window"])
    return model, mean, std, window, ckpt


def freeze_backbone(model: nn.Module) -> tuple[int, int]:
    """Train ``head.*`` only, freeze conv backbone (``feat.*``).

    Returns (n_trainable_params, n_total_params).
    """
    n_train = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if name.startswith("head"):
            p.requires_grad = True
            n_train += p.numel()
        else:
            p.requires_grad = False
    return n_train, n_total


# ──────────────────────────────────────────────────────────────────────────
# Window preprocessing — mirror dataset_bits / convert_to_hdf5.filter_emg
# ──────────────────────────────────────────────────────────────────────────

def preprocess_window(emg: np.ndarray, mean, std) -> np.ndarray:
    """emg: (T, 8) float raw → returns (8, T) float32 filtered+normalized."""
    x = filter_emg(emg, fs=SR)         # (T, 8) bandpass 20–240 Hz + 50 Hz notch
    x = (x - mean) / std
    return x.T.astype(np.float32, copy=False)


# ──────────────────────────────────────────────────────────────────────────
# UI rendering
# ──────────────────────────────────────────────────────────────────────────

def render(width: int, height: int, *, cue: str, probs: np.ndarray,
           bits: np.ndarray, phase: str, t_left: float, idx: int, n_pat: int,
           rep: int, total_rep: int, train_steps: int, replay_n: int,
           train_on: bool, smooth_str: str,
           wait_enter: bool = False) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)

    # Header strip
    head_txt = (f"pattern {idx+1}/{n_pat}   rep {rep}/{total_rep}   "
                f"replay={replay_n}   "
                f"{'TRAINING' if train_on else 'BIOFEEDBACK ONLY'}: steps={train_steps}")
    cv2.putText(canvas, head_txt, (24, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(canvas, (24, 46), (width - 24, 46), (60, 60, 60), 1)

    # Split mid-line
    mid = width // 2
    cv2.line(canvas, (mid, 60), (mid, height - 90), (50, 50, 50), 1)

    # ─── LHS: model predictions (LED row, matches live_binary.py) ───────
    cv2.putText(canvas, "MODEL PREDICTS", (40, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    radius_l = 48
    lhs_left = 40
    lhs_w = mid - 80
    gap_l = lhs_w // 5
    cy_l = 220
    for i, lbl in enumerate(FINGER_NAMES):
        cx = lhs_left + gap_l // 2 + i * gap_l
        on = bool(bits[i])
        col = (0, 220, 0) if on else (60, 60, 60)
        cv2.circle(canvas, (cx, cy_l), radius_l, col, -1)
        cv2.circle(canvas, (cx, cy_l), radius_l, (220, 220, 220), 2)
        # Percentage inside the LED
        pct = f"{probs[i]*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(canvas, pct, (cx - tw // 2, cy_l + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (15, 15, 15) if on else (230, 230, 230), 2, cv2.LINE_AA)
        # Finger label below
        cv2.putText(canvas, lbl[:5], (cx - 30, cy_l + radius_l + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1,
                    cv2.LINE_AA)
    # Bit string + smoothing summary
    bitstr = "".join(str(b) for b in bits)
    cv2.putText(canvas, f"bits  {bitstr}", (40, cy_l + radius_l + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, smooth_str, (40, cy_l + radius_l + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

    # ─── RHS: target gesture ──────────────────────────────────
    cv2.putText(canvas, "DO THIS", (mid + 40, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    big = " ".join(list(cue))
    (tw, _), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)
    cv2.putText(canvas, big,
                (mid + (mid - tw) // 2, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (240, 240, 240), 5, cv2.LINE_AA)
    # LED row
    radius = 32
    rhs_left = mid + 40
    rhs_w = mid - 80
    gap = rhs_w // 5
    cy = 320
    for i, ch in enumerate(cue):
        cx = rhs_left + gap // 2 + i * gap
        on = ch == "1"
        col = (0, 220, 0) if on else (55, 55, 55)
        cv2.circle(canvas, (cx, cy), radius, col, -1)
        cv2.circle(canvas, (cx, cy), radius, (220, 220, 220), 2)
        cv2.putText(canvas, FINGER_NAMES[i][:3],
                    (cx - 26, cy + radius + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1,
                    cv2.LINE_AA)

    # ─── Phase bar at bottom ─────────────────────────────────
    bar_y = height - 70
    bar_col = {"HOLD": (0, 200, 0), "REST": (60, 60, 200),
               "READY": (0, 180, 220)}.get(phase, (110, 110, 110))
    cv2.rectangle(canvas, (24, bar_y), (width - 24, bar_y + 36), bar_col, -1)
    if wait_enter:
        bar_txt = f"{phase}    press ENTER to continue   ({t_left:4.1f}s)"
    else:
        bar_txt = f"{phase}    {t_left:4.1f}s"
    cv2.putText(canvas, bar_txt,
                (40, bar_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2, cv2.LINE_AA)
    hint = ("ESC abort   ENTER next   S skip phase" if wait_enter
            else "ESC abort   S skip phase")
    cv2.putText(canvas, hint,
                (24, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)
    return canvas


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model_20260425_202900.pt",
                    help="Base checkpoint to adapt from. Loaded read-only.")
    ap.add_argument("--save", default=None,
                    help="Where to write the adapted checkpoint. "
                         "Default: do NOT save, just demo.")
    ap.add_argument("--patterns", default=",".join(DEFAULT_PATTERNS))
    ap.add_argument("--hold", type=float, default=4.0)
    ap.add_argument("--rest", type=float, default=2.0)
    ap.add_argument("--ready", type=float, default=1.5)
    ap.add_argument("--repeats", type=int, default=3)

    # Online training
    ap.add_argument("--no_train", action="store_true",
                    help="Disable weight updates (UI only — biofeedback demo).")
    ap.add_argument("--no_freeze", action="store_true",
                    help="Train all params (default: freeze conv, head only).")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps_per_sec", type=float, default=2.0)
    ap.add_argument("--replay_size", type=int, default=8000)
    ap.add_argument("--trim_start_ms", type=float, default=400.0,
                    help="Skip the first N ms of each HOLD when adding to "
                         "replay (reaction time + EMG ramp).")
    ap.add_argument("--trim_end_ms", type=float, default=200.0)

    # Display-only smoothing (matches live_binary.py defaults).
    # Training data goes into the replay buffer UNSMOOTHED — these knobs
    # only affect what the user sees on the LHS LEDs.
    ap.add_argument("--median-k", type=int, default=5,
                    help="Display median filter K (frames).")
    ap.add_argument("--ema-alpha", type=float, default=0.35,
                    help="Display EMA weight on newest prob (0,1].")
    ap.add_argument("--on-thresh", type=float, default=0.6)
    ap.add_argument("--off-thresh", type=float, default=0.4)

    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_save_csv", action="store_true",
                    help="Skip writing the cued recording CSV.")
    args = ap.parse_args()

    # Resolve device + paths
    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    device = torch.device(args.device)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abspath = lambda q: q if os.path.isabs(q) else os.path.join(root, q)
    model_path = abspath(args.model)

    # ───────── Model setup ──────────────────────────────────
    print(f"Device: {device}")
    print(f"Loading: {model_path}")
    model, mean, std, window, base_ckpt = load_model(model_path, device)
    model.eval()

    if not args.no_train:
        if args.no_freeze:
            n_train = sum(p.numel() for p in model.parameters())
            n_total = n_train
        else:
            n_train, n_total = freeze_backbone(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=args.lr) if trainable else None
        bce = nn.BCEWithLogitsLoss()
        print(f"Online training ON: {n_train:,}/{n_total:,} params trainable "
              f"({'all' if args.no_freeze else 'head only'}), lr={args.lr}, "
              f"{args.steps_per_sec} steps/s, batch={args.batch}, "
              f"replay={args.replay_size}")
    else:
        opt = None
        bce = None
        print("Online training OFF — biofeedback only.")

    # ───────── Patterns + protocol ─────────────────────────
    patterns = parse_patterns(args.patterns)
    n_pat = len(patterns)
    cued_secs = n_pat * args.repeats * args.hold
    print(f"Patterns ({n_pat}): {patterns}")
    print(f"Hold {args.hold}s | Rest {args.rest}s | Ready {args.ready}s | "
          f"{args.repeats} reps  → {cued_secs:.0f}s of cued recording.")

    # ───────── EMG stream ──────────────────────────────────
    stream = EMGStream(synthetic=args.synthetic, enable_filter=False)
    stream.start()

    # Warm-up sanity (mirrors record_bits.py)
    time.sleep(1.0)
    warm = stream.get_batch(max_items=4096)
    if len(warm) < 250:
        stream.stop()
        raise SystemExit(
            f"EMG stream not delivering data (got {len(warm)} samples in 1 s). "
            f"Check armband/Wi-Fi.")
    arr = np.asarray([s.emg for s in warm], dtype=np.float32)
    if np.allclose(arr, 0.0):
        stream.stop()
        raise SystemExit("EMG stream is all zeros — check electrode contact.")
    print(f"Warm-up OK: {len(warm)} samples, "
          f"per-channel std: {np.round(arr.std(axis=0), 2).tolist()}")

    # ───────── Buffers ─────────────────────────────────────
    # Ring buffer of the most recent ~5 s of raw samples.
    ring_t: deque[float] = deque(maxlen=int(SR * 5))
    ring_x: deque[list[float]] = deque(maxlen=int(SR * 5))
    for s in warm:
        ring_t.append(s.timestamp)
        ring_x.append(s.emg)

    # Replay buffer: list of (window (8,T) float32, label (5,) float32)
    replay: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=args.replay_size)

    # CSV log of every raw sample with its cue label (None during READY/REST)
    csv_rows: list[list] = []  # only populated during HOLD

    # ───────── State ──────────────────────────────────────
    state = {
        "probs": np.zeros(5, dtype=np.float32),  # post-smoothing (display)
        "bits": np.zeros(5, dtype=np.int32),     # post-hysteresis (display)
        "probs_buf": deque(maxlen=max(1, args.median_k)),  # for median filter
        "probs_ema": None,                       # EMA state
        "last_inf_t": 0.0,
        "last_train_t": 0.0,
        "train_steps": 0,
    }
    smooth_str = (f"med K={args.median_k}  ema {args.ema_alpha:.2f}  "
                  f"hyst >{args.on_thresh:.2f} <{args.off_thresh:.2f}")

    win_name = "live co-adapt — ESC abort"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1200, 640)

    def drain():
        b = stream.get_batch(max_items=4096)
        for s in b:
            ring_t.append(s.timestamp)
            ring_x.append(s.emg)

    def latest_window():
        if len(ring_x) < window:
            return None
        x = np.asarray(list(ring_x)[-window:], dtype=np.float32)
        return preprocess_window(x, mean, std)  # (8, T)

    def infer(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xb = torch.from_numpy(x[None]).to(device)
            return torch.sigmoid(model(xb)).cpu().numpy()[0]

    def train_step():
        if opt is None or len(replay) < args.batch:
            return
        idx = np.random.randint(0, len(replay), size=args.batch)
        items = [replay[i] for i in idx]
        X = np.stack([it[0] for it in items])
        Y = np.stack([it[1] for it in items])
        xb = torch.from_numpy(X).to(device)
        yb = torch.from_numpy(Y).to(device)
        # IMPORTANT: keep frozen backbone's BatchNorm in eval mode so its
        # running mean/var don't drift on the replay batch. Only the head
        # (which has no BN, but does have Dropout) goes into train mode.
        if args.no_freeze:
            model.train()
        else:
            model.head.train()
            model.feat.eval()
        logits = model(xb)
        loss = bce(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        state["train_steps"] += 1
        model.eval()

    def run_phase(phase: str, duration: float, pattern: str,
                  record_to_replay: bool, idx: int, rep: int,
                  wait_enter: bool = False) -> bool:
        t_end = time.time() + duration
        phase_start = time.time()
        trim_start_s = args.trim_start_ms / 1000.0
        trim_end_s = args.trim_end_ms / 1000.0
        cue_label = (np.asarray([int(c) for c in pattern], dtype=np.float32)
                     if record_to_replay else None)

        while True:
            now = time.time()
            if wait_enter:
                t_left = now - phase_start  # show elapsed
            else:
                t_left = t_end - now
                if t_left <= 0:
                    return True

            drain()

            # Inference at ~20 Hz (50 ms hop)
            if now - state["last_inf_t"] >= 0.05:
                w = latest_window()
                if w is not None:
                    raw = infer(w)  # raw sigmoid probs (unsmoothed)
                    # ── display-only smoothing pipeline (mirrors live_binary)
                    if args.median_k > 1:
                        state["probs_buf"].append(raw)
                        med = np.median(
                            np.stack(state["probs_buf"], axis=0), axis=0
                        ).astype(np.float32)
                    else:
                        med = raw
                    if state["probs_ema"] is None:
                        state["probs_ema"] = med.copy()
                    else:
                        a = args.ema_alpha
                        state["probs_ema"] = (
                            a * med + (1.0 - a) * state["probs_ema"]
                        )
                    sm = state["probs_ema"]
                    state["probs"] = sm
                    # hysteresis → bits
                    for i in range(5):
                        if sm[i] > args.on_thresh:
                            state["bits"][i] = 1
                        elif sm[i] < args.off_thresh:
                            state["bits"][i] = 0
                    # Record raw HOLD window into replay (NOT smoothed).
                    if record_to_replay and cue_label is not None:
                        elapsed = now - phase_start
                        if elapsed >= trim_start_s and (
                                wait_enter or t_left >= trim_end_s):
                            replay.append((w.copy(), cue_label.copy()))
                state["last_inf_t"] = now

            # Online training step
            if (not args.no_train
                    and now - state["last_train_t"] >= 1.0 / args.steps_per_sec):
                train_step()
                state["last_train_t"] = now

            # Save raw HOLD samples for the on-disk CSV
            if record_to_replay and cue_label is not None and not args.no_save_csv:
                # Emit the samples that landed in the ring since last frame.
                # Simpler: at end-of-phase we extract by timestamp — see below.
                pass

            # Render UI
            panel = render(1200, 640,
                           cue=pattern, probs=state["probs"],
                           bits=state["bits"],
                           phase=phase, t_left=t_left,
                           idx=idx, n_pat=n_pat,
                           rep=rep, total_rep=args.repeats,
                           train_steps=state["train_steps"],
                           replay_n=len(replay),
                           train_on=not args.no_train,
                           smooth_str=smooth_str,
                           wait_enter=wait_enter)
            cv2.imshow(win_name, panel)
            k = cv2.waitKey(10) & 0xFF
            if k == 27:                       # ESC
                return False
            if k == ord("s"):                 # skip current phase
                return True
            if wait_enter and k in (10, 13):  # Enter / Return
                return True

    # ───────── Run protocol ─────────────────────────────
    # Note: we collect CSV rows by snapshotting the ring at end of HOLD
    # by timestamp window — easier than threading.
    aborted = False
    try:
        for rep in range(1, args.repeats + 1):
            for idx, pat in enumerate(patterns):
                if not run_phase("READY", args.ready, pat, False, idx, rep):
                    aborted = True
                    raise KeyboardInterrupt
                hold_t0 = time.time()
                if not run_phase("HOLD", args.hold, pat, True, idx, rep,
                                 wait_enter=True):
                    aborted = True
                    raise KeyboardInterrupt
                hold_t1 = time.time()
                if not args.no_save_csv:
                    # Snapshot ring buffer for samples in [hold_t0, hold_t1].
                    ts = list(ring_t)
                    xs = list(ring_x)
                    bits = list(pat)
                    for t, ex in zip(ts, xs):
                        if hold_t0 <= t <= hold_t1:
                            csv_rows.append(
                                [f"{t:.6f}"]
                                + [f"{v:.4f}" for v in ex]
                                + bits
                            )
                if idx < n_pat - 1 or rep < args.repeats:
                    if not run_phase("REST", args.rest, pat, False, idx, rep):
                        aborted = True
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\nAborted." if aborted else "\nInterrupted.")

    stream.stop()
    cv2.destroyAllWindows()

    print(f"\nReplay buffer: {len(replay)} windows  |  "
          f"Train steps: {state['train_steps']}")

    # ───────── Save raw cued recording (CSV) ──────────────
    if csv_rows and not args.no_save_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_dir = Path(root) / "recordings" / f"coadapt_{stamp}"
        rec_dir.mkdir(parents=True, exist_ok=True)
        csv_path = rec_dir / "emg.csv"
        header = (["timestamp"] + [f"ch{i}" for i in range(8)]
                  + [f"b{i}" for i in range(5)])
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(csv_rows)
        with open(rec_dir / "meta.txt", "w") as f:
            f.write(f"created: {stamp}\n")
            f.write(f"base_model: {model_path}\n")
            f.write(f"hold_s: {args.hold}\n")
            f.write(f"rest_s: {args.rest}\n")
            f.write(f"ready_s: {args.ready}\n")
            f.write(f"repeats: {args.repeats}\n")
            f.write(f"patterns: {patterns}\n")
            f.write(f"train_steps: {state['train_steps']}\n")
            f.write(f"replay_size: {len(replay)}\n")
            f.write(f"synthetic: {args.synthetic}\n")
        print(f"Saved cued recording: {csv_path} ({len(csv_rows)} samples)")

    # ───────── Save adapted model ──────────────────────────
    if args.save and not args.no_train:
        save_path = abspath(args.save)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        ckpt = dict(base_ckpt)  # shallow copy
        ckpt["state_dict"] = model.state_dict()
        ckpt["coadapt"] = {
            "base_model": os.path.basename(model_path),
            "train_steps": state["train_steps"],
            "replay_size": len(replay),
            "lr": args.lr,
            "froze_backbone": not args.no_freeze,
            "trim_start_ms": args.trim_start_ms,
            "trim_end_ms": args.trim_end_ms,
        }
        torch.save(ckpt, save_path)
        print(f"Saved adapted model: {save_path}")
    elif args.save and args.no_train:
        print("--save ignored because --no_train was set.")
    else:
        print("(no --save given; adapted weights discarded)")


if __name__ == "__main__":
    main()
