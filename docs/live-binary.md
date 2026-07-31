# Live binary inference (`src.live_binary`)

Runs the trained 5-bit classifier in real time and displays a 5-LED panel:
green = finger bent, grey = extended. Updates at the configured hop rate
(default 20 Hz). No Arduino, no servo control — pure model + visualization.

For the Arduino-driven version see `hand_control_emg_arduino.py`. Both share
the same `BinaryPredictor` and the same checkpoint format.

---

## Run

```bash
python -m src.live_binary --model models/<your-checkpoint>.pt
```

If the checkpoint was trained with `--imu` (i.e. `multistream=True`), the
predictor automatically instantiates `EMG5BitMulti` and feeds the IMU
stream too. You don't pass any flag for that.

### Recommended command (latest IMU model)

```bash
python -m src.live_binary --model models/imu_latest.pt
```

### Quick test without hardware

```bash
python -m src.live_binary --model models/imu_latest.pt --synthetic
```

---

## Flags

### Model & data

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `models/emg5bit_selflabel_best.pt` | Path to a checkpoint produced by `src.train_bits`. Whether it's EMG-only or two-stream is read from the checkpoint itself — no flag for that. |
| `--dataset` | none | HDF5 dataset path. Only used as a fallback for norm stats if the checkpoint doesn't have them embedded. Self-labeled checkpoints (everything saved by `train_bits`) embed their own stats, so you usually omit this. |

### Inference loop

| Flag | Default | Meaning |
|---|---|---|
| `--hop_ms` | `50.0` | Inference period in ms. 50 ms = 20 Hz. Lower = more responsive but more CPU and more frame-to-frame jitter. Higher = smoother but laggy. |
| `--device` | auto | `cuda` / `mps` / `cpu`. Auto-picks the best available. |
| `--synthetic` | off | Use BrainFlow's synthetic board for testing without the MindRove. |

### Smoothing pipeline (raw prob → bit)

The model emits 5 sigmoid probabilities every frame. Three filters in series
turn that into a stable bit:

```
prob_raw  →  median over K frames  →  EMA(alpha)  →  hysteresis  →  bit
```

#### Median filter

| Flag | Default | Meaning |
|---|---|---|
| `--median-k` | `5` | Window of the per-bit median filter, in frames. At `--hop_ms 50`, K=5 covers the last 250 ms. Kills 1–2 frame spikes (e.g. `0.9 → 0.1 → 0.9` becomes `0.9`). Set to `1` to disable. |

Use a larger K when you see brief blips of the wrong prediction; use a smaller K when responsiveness matters more than spike rejection.

#### EMA

| Flag | Default | Meaning |
|---|---|---|
| `--ema-alpha` | `0.35` | Weight on the *new* sample in the exponential moving average: `p_ema ← alpha * p_new + (1 - alpha) * p_ema`. Lower alpha = smoother + more lag. `1.0` disables EMA. |

`0.35` keeps about 100 ms of effective smoothing at 20 Hz. Lower it (e.g. 0.2) if the LEDs flicker; raise it (e.g. 0.6) if the response is too sluggish.

#### Hysteresis

Two thresholds with a dead zone between them, so probabilities hovering near the boundary don't chatter the bit on/off.

| Flag | Default | Meaning |
|---|---|---|
| `--on-thresh` | `0.6` | Latch the bit ON when the smoothed prob crosses **above** this value. |
| `--off-thresh` | `0.4` | Latch the bit OFF when the smoothed prob crosses **below** this value. |

Probabilities in the dead zone (0.4–0.6) keep whatever state was latched last. The `0.4 < 0.6` invariant is enforced — the script will refuse to start otherwise. Tighter band (e.g. `0.55 / 0.45`) → more sensitive, more chatter. Wider band (e.g. `0.7 / 0.3`) → very stable, slower to flip.

---

## What you see

The OpenCV window has:

- **5 circles** labeled THUMB / INDEX / MIDDLE / RING / PINKY. Green when the latched bit is 1, grey when 0. Each shows the post-smoothing probability as a percentage.
- **Status line at the bottom**:
  ```
  bits: 01010
  hop 50ms  inf 1.2ms  med K=5  ema 0.35  hyst >0.6 <0.4
  ```
  - `bits` — current latched 5-bit string (THUMB first).
  - `hop` — the configured period.
  - `inf` — actual model forward time. If this approaches `hop`, lower the inference rate or move to a smaller model.
  - The rest echoes the active smoothing config.

### Keys

- `q` or `ESC` — quit. Stops the predictor + EMG stream cleanly.

---

## Quick troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `EMG stream is not delivering data` after warmup | MindRove off, not on Wi-Fi, wrong IP, or armband not turned on. | Check the box, ping `192.168.4.1`, restart the armband. |
| `EMG warm-up was all zeros` | Electrodes not in contact. | Re-seat the band, wipe forearm with alcohol. |
| `Loaded norm stats from checkpoint.` not printed | Old HDF5-trained checkpoint without embedded stats. | Pass `--dataset path/to/dataset.hdf5` so the predictor can pull stats from it. |
| `Multistream checkpoint missing imu_mean/imu_std` | Tried to run a stale `--imu` checkpoint that pre-dates the IMU stats fields. | Re-train with current `train_bits.py --imu`. |
| LEDs flicker rapidly | Smoothing too aggressive, or model is genuinely uncertain near threshold. | Lower `--ema-alpha` (e.g. `0.2`), raise `--median-k` (e.g. `7`), or widen the hysteresis band (`--on-thresh 0.7 --off-thresh 0.3`). |
| LEDs are sluggish to react | Smoothing too aggressive, or hop too high. | Raise `--ema-alpha` (e.g. `0.6`), lower `--median-k`, or lower `--hop_ms` (e.g. `30`). |
| Wrong fingers light up consistently | Model trained on different electrode placement, or band is rotated. | Mark band orientation on your arm, re-seat to the same orientation as training. If the issue persists, re-train. |
| Some fingers stuck off (or on) | Per-finger imbalance in training data. | Check the `Per-bit positive fraction` line printed by `train_bits` for the checkpoint that produced this model. Re-record balanced patterns and retrain. |

---

## Reading the panel for diagnosis

The percentage shown inside each LED is the **post-smoothing** probability,
not the raw model output. If you see something like `48% / 52% / 49% / 51% / 50%`
across all five fingers and the LEDs flip randomly, the model is essentially
predicting a constant — it's at chance. That's a training problem, not a
smoothing problem; revisit `docs/training.md` and the `mean` accuracy of the
checkpoint.

If raw confidences are healthy (e.g. `92% / 4% / 3% / 5% / 4%` when only
thumb is bent) but the LED still flickers, then the smoothing knobs above
will fix it.
