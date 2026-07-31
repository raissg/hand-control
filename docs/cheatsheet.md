# Architecture Cheatsheet

Every concrete number in the EMG → finger-bits pipeline, plus a one-line
reason for each. Scan-friendly. For depth, see `ai-architecture.md` and
`temporal-consistency.md`.

---

## Sensors

| Item | Value | Why |
|---|---|---|
| EMG channels | 8 | What the MindRove armband ships with. Enough for finger-level discrimination on the forearm. |
| IMU channels | 6 (3 accel + 3 gyro) | Same armband, free signal. Accel = posture, gyro = motion. |
| Sample rate | 500 Hz | Above Nyquist for the 20–250 Hz EMG band. Standard in the literature. |
| Wireless link | Wi-Fi (UDP) | MindRove default — ~5 ms typical latency, no USB tethering. |

## Pre-processing (live + training)

| Item | Value | Why |
|---|---|---|
| Bandpass | 20–240 Hz, 4th-order Butterworth | 20 Hz cuts motion drift; 240 Hz keeps full muscle band, stays clear of Nyquist. |
| Notch | 50 Hz | Mains hum (Qatar/EU; would be 60 Hz in US). |
| Filter style | `filtfilt` (zero-phase) | No phase distortion across windows. |
| Per-channel z-score | mean/std from train split | EMG amplitude varies session-to-session; normalize so the model sees a consistent distribution. |
| IMU normalization | separate per-channel z-score | Accel has a 9.8 DC offset, gyro is zero-mean — different scales need different stats. |

## Input window

| Item | Value | Why |
|---|---|---|
| Window length | 200 ms = 100 samples @ 500 Hz | Long enough to contain one full muscle activation envelope, short enough to feel responsive. |
| Hop / update rate | 50 ms = 20 Hz | 4× overlap with the window — catches transitions promptly. |
| EMG tensor shape | `(8, 100)` | 8 channels × 100 timesteps. |
| IMU tensor shape | `(6, 100)` | Same time span as EMG, aligned 1:1. |

## EMG branch (1-D CNN)

| Layer | Params | Why |
|---|---|---|
| `Conv1d(8 → 32, k=5, pad=2)` | k=5 covers ~10 ms — fine waveform shape. 32 = enough learned shapes without overfitting. |
| `BatchNorm1d(32) → ReLU` | BN soaks up session-to-session amplitude drift; ReLU lets shapes compose. |
| `MaxPool1d(2)` | Halves time; doubles next layer's receptive field for free. |
| `Conv1d(32 → 64, k=5, pad=2)` | More features for richer mid-level shapes. |
| `BN → ReLU → MaxPool(2)` | Same idea, deeper. |
| `Conv1d(64 → 128, k=5, pad=2)` | High-level features. 128 chosen to roughly match the data complexity. |
| `BN → ReLU → AdaptiveAvgPool1d(1)` | Collapses time → fixed-size 128-D embedding regardless of window length. |
| Output | `(128,)` | One vector summary of muscle activity in the window. |
| **Receptive field** | ~40 samples ≈ **80 ms** | Matches the duration of a single muscle activation. By design. |

## IMU branch (1-D CNN, smaller)

| Layer | Params | Why |
|---|---|---|
| `Conv1d(6 → 16, k=7, pad=3)` | Bigger kernel — IMU signal is slower than EMG. |
| `BN → ReLU → MaxPool(4)` | Aggressive downsampling — gravity changes over 100+ ms, no need for ms resolution. |
| `Conv1d(16 → 32, k=5, pad=2)` | Mid-level posture/motion features. |
| `BN → ReLU → AdaptiveAvgPool1d(1)` | Collapse to fixed size. |
| Output | `(32,)` | Smaller than EMG embedding — IMU is supplementary. |

## Fusion head

| Layer | Why |
|---|---|
| `concat([EMG-128, IMU-32])` → 160-D | Late fusion is simplest, most data-efficient for small datasets. |
| `Linear(160 → 64) → ReLU` | Learn a nonlinear mix of all features. |
| `Dropout(0.3)` | Regularize against overfitting on the small dataset. |
| `Linear(64 → 5)` | 5 logits, one per finger. |

**Total params: ~50k.** Sized to the data we have, fits a sub-1 ms forward pass on CPU.

## Output → bits

| Step | Why |
|---|---|
| `sigmoid(logits)` per finger | 5 independent probabilities in `[0, 1]`. |
| Threshold at 0.5 (raw) | Default decision boundary. Hysteresis replaces this in live use. |
| Bit order | `[thumb, index, middle, ring, pinky]` | Matches `F<5 bits>` Arduino protocol. |

## Training

| Item | Value | Why |
|---|---|---|
| Loss | `BCEWithLogitsLoss` per finger | Fingers move independently → 5 yes/no questions, not a 32-way pick. |
| `pos_weight` | `(1−p) / p` per finger | Class imbalance correction; `p` = empirical positive fraction. |
| Optimizer | AdamW | Decoupled weight decay; standard for small CNNs. |
| Learning rate | 1e-3 | Default Adam. Lowered to 3e-4 when `--resume`. |
| Weight decay | 1e-3 | Mild L2; the conv body is small so we don't need much. |
| Batch size | 128 | Big enough for stable BatchNorm stats, small enough to fit anywhere. |
| Epochs | up to 60 | With early stop at patience=12. |
| Window stride (training) | 10 samples | ~50 windows/sec held → tons of training samples per recording. |
| Train/val split | 0.8/0.2, **segment-level** | Split by held pose, not by window — prevents leakage. |
| Seed | 0 (default) | Reproducible splits. |

## Temporal-consistency regularizer (optional)

| Item | Value | Why |
|---|---|---|
| Formula | `BCE(t) + BCE(t+Δ) + λ·MSE(p_t, p_{t+Δ})` | Model is forced to give similar outputs for two time-shifted views of the same hold. |
| `λ` | 0.05–0.2 typical (0 = off) | Too small = no effect; too large = collapses to a constant prediction. |
| `Δ` range | 10–100 ms (samples 5–50 @ 500 Hz) | Short enough that the label is unchanged, long enough to test robustness. |
| Effect | Less inference smoothing needed | Shifts work from inference (latency) to training (free). |

## Label remap (gesture-set reduction)

| Recorded pattern | Remapped to | Why |
|---|---|---|
| `00000`, `11111`, `10000`, `01000`, `11000`, `01100`, `11100` | (kept) | The 7 reliably-classifiable canonical gestures. |
| `10001` | `10000` | Pinky-add unreliable; thumb-only is stable. |
| `11001` | `11000` | Pinky-add unreliable. |
| `01110`, `01111`, `10111`, `11011`, `11101` | `11111` | Almost-fist variants → canonical fist. |

After remap: **val mean per-bit accuracy 91.8% → 99.1%.**

## Smoothing pipeline (inference only)

| Stage | Default | Why |
|---|---|---|
| Filter pad | 200 samples | Pre-window padding so `filter_emg` has run-up; avoids edge artifacts. |
| Median (per bit) | K=9 frames (~450 ms) | Kills 1–2 frame spikes that EMA can't handle without huge lag. |
| EMA | α=0.18 (~6-frame TC, ~300 ms) | Softens residual jitter. Lower = smoother + more lag. |
| EMA space | logit (default) | Sigmoid is nonlinear; averaging in logit space is more stable near saturation. |
| Hysteresis | on=0.7, off=0.3 | Dead zone prevents chatter at borderline activations. |
| min_consecutive | 3 frames | Bit only flips after 3 consecutive frames agree. |
| min_dwell | 0.2 s | After a flip, that bit can't flip again for 200 ms. |

All five tunable from the CLI.

## Live system

| Item | Value | Why |
|---|---|---|
| Inference loop | 50 ms (20 Hz) | Matches training hop. |
| Forward pass | < 1 ms (CPU) | Tiny model. |
| End-to-end latency | ~50–100 ms typical | Hop + smoothing + serial RTT. |
| Output channel | USB serial @ 115200 | Reliable on macOS, Arduino-default. |
| Servo command | `F<5 bits>` per change | One byte command + 5 ASCII bits = 6 bytes. |
| Per-servo lockout | 2 s on Arduino side | Hardware-level safety: a finger can't flip more than once every 2 s, no matter what the host says. |

## Hardware downstream

| Item | Value | Why |
|---|---|---|
| Servos | PDI-6225MG, 5× | 300° digital servos, enough range for finger flexion. |
| Pulse range | 500–2500 µs | Spec for 0–300° on PDI-6225MG. Default `Servo.attach()` clamps to 180°. |
| Per-finger angles | Calibrated, e.g. thumb 95°↔37° | Measured with the slider UI. |
| MCU | Arduino UNO R4 WiFi | Hardware PWM on 6 timers, USB CDC at 115200. |
| Power | External 6 V (battery) | Servo current spikes would brown-out a USB-only setup. |

## What we deliberately *don't* do

| Skipped | Why |
|---|---|
| Cross-session calibration / domain adaptation | Out of scope for this project; per-session norm stats are already in the checkpoint. |
| Spatial filter as first layer (à la EEGNet) | Could squeeze a few more % but adds complexity; not the bottleneck. |
| Structured output / CRF over fingers | BCE-per-finger handles unseen combinations gracefully; structured output needs more data. |
| Continuous joint-angle regression | Different model (`neuropose_*`); the binary classifier is the live demo. |
| Bigger model (transformer, RNN) | <30 min of recordings — anything larger memorizes. |

---

## Cheat-line summary

> 200 ms × 14 channels in → two-stream CNN (EMG: 3-conv k=5, IMU: 2-conv k=7/5) → concat 160-D → 2-layer MLP → 5 sigmoids → median + EMA + hysteresis + dwell → 5 bits at 20 Hz over USB serial → 5 servos.
