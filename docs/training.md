# Training the 5-bit finger classifier

The pipeline has three stages — **record**, **train**, **run live**. This doc
explains every flag of each stage so you don't have to read the source.

---

## 1. Record

```bash
python -m src.record_bits [flags]
```

Walks you through 5-bit pose patterns shown on screen. While you hold each
pose, raw EMG (and optionally IMU) is written, labeled with the shown bits.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--hold` | `4.0` | Seconds to hold each pose. The full hold window is recorded as one labeled segment. |
| `--rest` | `2.0` | Seconds of rest between poses. Not recorded — gives you time to relax. |
| `--ready` | `1.5` | Pre-hold countdown so you can get into position. Not recorded. |
| `--repeats` | `3` | Cycles through the full pattern list this many times. More repeats → more data → better generalization. |
| `--patterns` | 14 default patterns | Comma-separated 5-bit strings (`THUMB,INDEX,MIDDLE,RING,PINKY` order). Override only if you want a focused subset, e.g. `--patterns 00000,11111,01000`. |
| `--synthetic` | off | Use BrainFlow's synthetic board — for testing the recorder without hardware. |
| `--out_dir` | `recordings/bits_<timestamp>` | Override the output directory. |
| `--imu` | off | **Also persist accel + gyro columns.** Required if you plan to train with `--imu`. Without this flag, the CSV only has `timestamp, ch0..ch7, b0..b4` — same as legacy recordings. |

### Output

```
recordings/bits_YYYYMMDD_HHMMSS/
  emg.csv      # one row per EMG sample
  meta.txt     # the flag values used for this session
```

CSV columns:
- without `--imu`: `timestamp, ch0..ch7, b0..b4`
- with `--imu`:    `timestamp, ch0..ch7, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, b0..b4`

### Tips

- **Wipe your forearm with alcohol** before each session and let it dry. EMG is brutally sensitive to skin contact.
- **Watch the warmup printout** — it shows `per-channel std`. If any channel is near 0, that electrode isn't making contact; reposition the band before the patterns start.
- **Re-don the band between sessions.** Cross-session generalization is what matters for live use; recording one giant session under one band placement makes the model overfit to that exact placement.
- **Cover all 5 fingers as positives.** If the printed `pos_frac` (in training) shows finger 3 at 0.05, the model can't learn that finger. Fix it by recording more `xxxxx`-with-ring-bent patterns.

---

## 2. Train

```bash
python -m src.train_bits [flags]
```

Loads sessions from `recordings/`, splits each session into segments
(contiguous stretches with the same bit pattern), filters EMG, slides
windows over segments, and trains either `EMG5Bit` (EMG only) or
`EMG5BitMulti` (EMG + IMU two-stream).

### Flags

#### Data selection

| Flag | Default | Meaning |
|---|---|---|
| `--sessions` | `"recordings/bits_*/emg.csv"` | Glob (or comma-separated globs) of CSVs to train on. To train on one specific session: `--sessions "recordings/bits_20260427_233300/emg.csv"`. To train on a date range: `--sessions "recordings/bits_20260427_*/emg.csv"`. |
| `--gap_s` | `0.1` | If two consecutive samples differ by more than this many seconds, the segment is closed (a new pose-hold is starting). Don't change unless you know your record_bits gaps. |

#### Windowing

| Flag | Default | Meaning |
|---|---|---|
| `--window` | `100` | Window size in samples = 200 ms at 500 Hz. The model sees one such window per inference. |
| `--step` | `10` | Sliding-window stride in samples. Smaller → more training windows per second of recording (e.g. 10 → ~50 windows/s held). Smaller window also = more overlapping → mild regularization. |
| `--val_frac` | `0.2` | Fraction of *segments* (not windows) held out for validation. Splitting at segment level avoids window leakage. |

#### Optimization

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | `60` | Maximum number of full passes over the training set. |
| `--batch_size` | `128` | Windows per gradient step. |
| `--lr` | `1e-3` | AdamW learning rate. |
| `--weight_decay` | `1e-3` | L2 weight decay (inside AdamW). |
| `--dropout` | `0.3` | Dropout in the MLP head. Lower if you have lots of data and the model is underfitting; raise if it's overfitting. |
| `--patience` | `12` | Early-stop if validation loss doesn't improve for this many consecutive epochs. |
| `--no-early-stop` | off | Always run the full `--epochs` regardless of patience. |
| `--seed` | `0` | RNG seed for the train/val segment split and the consistency-pair sampler. |

#### Resume / fine-tune

| Flag | Default | Meaning |
|---|---|---|
| `--resume` | none | Path to a checkpoint to continue training from. Reuses the checkpoint's norm stats + window so the input distribution stays consistent. |
| `--resume-lr` | (auto) | Override LR when resuming. By default, resuming drops LR to `3e-4` for stability. |

#### Consistency regularizer

When set, the loader yields a *pair* of windows from the same segment at a
small time offset, and the loss penalizes the model for changing its
prediction between them. Helps suppress live jitter at the cost of training
time.

| Flag | Default | Meaning |
|---|---|---|
| `--consistency-lambda` | `0.0` | Weight of the MSE between paired sigmoids. `0` disables the regularizer. Try `0.05–0.2` if your live output is jittery. |
| `--consistency-deltas` | `"5,50"` | `min,max` sample offsets for the pair window. At 500 Hz, `5,50` = 10–100 ms apart. |

#### IMU two-stream

| Flag | Default | Meaning |
|---|---|---|
| `--imu` | off | Train `EMG5BitMulti` (EMG + IMU branches). **Auto-filters out CSVs without IMU columns** so the IMU branch isn't fed zeros. Requires sessions recorded with `record_bits --imu`. |

#### Output

| Flag | Default | Meaning |
|---|---|---|
| `--save-name` | `model_<timestamp>.pt` | Checkpoint filename under `models/`. The "best validation loss" checkpoint is saved each time it improves. |

### What gets saved

`models/<save-name>` is a torch dict with:
- `state_dict` — model weights
- `input_mean` / `input_std` — EMG per-channel z-score stats
- `window` — input window length
- `multistream` — `True` if `--imu` was on
- `imu_mean` / `imu_std` — present only when `multistream=True`

Live code reads `multistream` to decide which architecture to instantiate.

### Recommended command (current data)

You only have one session with IMU columns (`bits_20260427_233300`). To train
two-stream on it:

```bash
python -m src.train_bits --imu \
  --sessions "recordings/bits_20260427_233300/emg.csv" \
  --save-name imu_only.pt
```

To compare against EMG-only on the *same* data (apples-to-apples):

```bash
python -m src.train_bits \
  --sessions "recordings/bits_20260427_233300/emg.csv" \
  --save-name emg_only_same_data.pt
```

If `emg_only_same_data.pt` and `imu_only.pt` both land near 0.5 mean accuracy,
the recording itself is bad (re-record). If EMG-only beats `--imu` clearly,
the IMU branch is hurting and we adjust the model.

### Reading the training log

```
Per-bit positive fraction (sample-weighted): [0.41, 0.45, 0.51, 0.40, 0.45]
```
Diagnostic: how often each finger is bent in the data. If any value is
near 0 or 1, you have an unbalanced dataset and that finger will train poorly.

```
Ep   3: train 0.5912 | val 0.6011 | per-bit acc [0.82, 0.86, 0.79, 0.81, 0.84] | mean 0.824
```
- `train` / `val` — BCE loss; lower is better.
- `per-bit acc` — fraction of validation windows where each finger's bit was predicted correctly.
- `mean` — average across the 5 fingers. This is the headline number.

Trustworthy zone: `mean > 0.85`. If it stalls below 0.7, the model is
underfitting (data quality issue) — don't keep tweaking model flags.

---

## 3. Run live

```bash
python hand_control_emg_arduino.py
```

Top-of-file constants you'll want to know:

| Constant | Default | Meaning |
|---|---|---|
| `PORT` | `/dev/cu.usbmodemCC8DA22028202` | Arduino serial port. |
| `MODEL` | `models/<file>.pt` | Checkpoint to load. **Edit this to point at your latest train output.** |
| `HOP_MS` | `50.0` | Inference period. 50 ms = 20 Hz. |
| `ON_THRESH` | `0.6` | Hysteresis: latch a finger ON when its smoothed prob > this. |
| `OFF_THRESH` | `0.4` | Hysteresis: latch OFF when smoothed prob < this. |
| `EMA_ALPHA` | `0.35` | EMA smoothing on probabilities. Lower = smoother + more lag. |
| `MEDIAN_K` | `5` | Median-filter window over the last K raw probs. Kills 1–2 frame spikes. |
| `LOCK_S` | `2.0` | Mirrors the Arduino's per-servo lockout (don't change without changing the sketch). |

The script auto-detects whether the checkpoint is multistream or EMG-only and
configures `BinaryPredictor` accordingly. No flag needed.

### Keys (during live run)

- `c` — sends `S` to the Arduino (detach all servos) and exits.
- `ESC` / `q` — quits without sending stop (servos hold).
