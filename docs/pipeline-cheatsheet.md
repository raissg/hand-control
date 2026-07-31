# Pipeline Cheatsheet — Full Walkthrough

End-to-end description of the EMG → finger-bits → servos pipeline. Every
concrete number, every kernel, every hyperparameter, plus a one-line
reason for each. Read top to bottom; each section assumes the previous.

---

## 0. The whole pipeline in one line

```
MindRove armband (8 EMG + 6 IMU @ 500 Hz)
  → EMGStream (filter + buffer)
  → 200 ms rolling window
  → Two-stream CNN (EMG branch + IMU branch)
  → Fusion MLP head → 5 logits
  → sigmoid → 5 probabilities
  → median + EMA + hysteresis + dwell
  → 5 latched bits
  → USB serial @ 115200
  → Arduino → 5 servos on the InMoov hand
```

Update rate: **20 Hz** (50 ms per inference). Total latency: ~50–100 ms.

---

## 1. Sensors

The MindRove armband sits on the forearm and streams over Wi-Fi (UDP).

- **8 EMG channels** at 500 Hz. Raw voltage from each electrode. Forearm
  muscles fire at 20–250 Hz, so 500 Hz is well above Nyquist.
- **6 IMU channels** at 500 Hz, aligned 1:1 with EMG: 3 from the
  accelerometer (X/Y/Z gravity + linear acceleration) and 3 from the
  gyroscope (X/Y/Z rotation rate).
- **Why both modalities**: EMG gives muscle activity (which finger is
  trying to bend). IMU gives arm posture and motion. Same gesture
  performed in different postures looks like different EMG, so IMU
  anchors the model against posture-induced drift.

---

## 2. Acquisition layer (`src/acquire.py`)

`EMGStream` is a thread that pulls samples from MindRove's BoardShim,
applies real-time filtering, and pushes onto a thread-safe queue.

- **Filtering on each window** (`filter_emg` in `convert_to_hdf5.py`):
  - Bandpass: **20–240 Hz**, 4th-order Butterworth.
    - 20 Hz cuts motion drift and DC offsets.
    - 240 Hz keeps the full muscle band, stays well clear of Nyquist.
  - Notch: **50 Hz** to kill mains hum (Qatar/EU electricity).
  - Style: `filtfilt` (zero-phase) — applied forward and backward so
    there's no phase distortion across windows.
- **Why filter twice** (live + offline at training time): the live
  filter eliminates motion drift before windowing; the offline filter
  during dataset construction guarantees the model trains on identical
  preprocessed data.

The IMU stream is *not* bandpass-filtered. Gravity is essentially DC and
posture changes are slow — bandpassing would destroy them.

---

## 3. Window

Every 50 ms, the consumer thread snapshots the last 200 ms from the ring
buffer:

- **EMG window**: shape `(8, 100)` — 8 channels × 100 samples.
- **IMU window**: shape `(6, 100)` — same time span.

Why these numbers:
- **200 ms** is the standard EMG window length. A muscle activation
  envelope lasts ~50–100 ms; 200 ms covers a full one with margin.
- **50 ms hop** = 4× overlap with the window, so transitions hit a
  prediction frame quickly instead of waiting for the next non-overlapping
  one.
- **20 Hz update** is fast enough that the user can't perceive lag, slow
  enough that the model has compute headroom for filtering and smoothing.

---

## 4. Per-channel z-score normalization

Before the model sees a window:

- EMG window is normalized by per-channel mean/std stored in the
  checkpoint (computed from the training split).
- IMU window is normalized by *separate* per-channel mean/std (also in
  the checkpoint).

Why separate stats:
- EMG amplitude varies wildly session-to-session (electrode contact, skin
  hydration, fatigue). z-scoring smooths that out.
- Accelerometer has a ~9.8 m/s² DC offset (gravity); gyro is zero-mean.
  Mixing them in one mean/std would compress accel and amplify gyro
  noise.

---

## 5. Two-stream CNN (`src/binary_model.py`)

The core model. Two parallel 1-D convnets, late-fused.

### 5a. EMG branch

```
Input:  (B, 8, 100)

  Conv1d(8 → 32, kernel=5, padding=2)
  BatchNorm1d(32) → ReLU
  MaxPool1d(2)                 → (B, 32, 50)

  Conv1d(32 → 64, kernel=5, padding=2)
  BatchNorm1d(64) → ReLU
  MaxPool1d(2)                 → (B, 64, 25)

  Conv1d(64 → 128, kernel=5, padding=2)
  BatchNorm1d(128) → ReLU
  AdaptiveAvgPool1d(1)         → (B, 128, 1)

  Flatten                      → (B, 128)
```

Why these specific choices:

- **3 stages** so the receptive field grows enough (~80 ms) to span a
  full muscle activation. With 1–2 stages the network would only see
  fragments.
- **Kernel size 5** in every conv: small enough to learn fine waveform
  detail, big enough that 3 stacked layers cover ~80 ms by the end.
- **`padding=2`** keeps each conv output the same length as its input —
  pooling does the downsampling explicitly, not the conv.
- **Channel growth 8→32→64→128**: the standard "more abstract features
  need more capacity" pattern. 128 dimensions is enough to carry rich
  muscle-pattern features but not so many that a small dataset overfits.
- **BatchNorm after every conv**: EMG amplitude varies a lot per
  recording session, so internal activations vary too. BN stabilizes
  training and acts as a mild regularizer.
- **MaxPool(2)** twice: each pool halves the time dimension and doubles
  the next layer's effective receptive field for free. Cheaper than
  using larger kernels.
- **AdaptiveAvgPool1d(1)** at the end: collapses the time dimension to a
  single vector. Two benefits: (a) the network is robust to small
  variations in window length, (b) the output is fixed-size 128 D
  regardless of input timing.

### 5b. IMU branch

```
Input:  (B, 6, 100)

  Conv1d(6 → 16, kernel=7, padding=3)
  BatchNorm1d(16) → ReLU
  MaxPool1d(4)                 → (B, 16, 25)

  Conv1d(16 → 32, kernel=5, padding=2)
  BatchNorm1d(32) → ReLU
  AdaptiveAvgPool1d(1)         → (B, 32, 1)

  Flatten                      → (B, 32)
```

Why it's different from EMG:

- **Bigger first kernel (7 vs 5)**: IMU signal is slow; a wider window
  catches meaningful posture/motion features.
- **MaxPool(4) up front**: aggressively downsamples — gravity changes
  over 100+ ms, no point keeping ms resolution.
- **Only 2 stages**: simpler signal, less depth needed.
- **Smaller channel widths (16, 32)**: 6 inputs vs 8 EMG; coarser
  features sufficient.
- **Smaller output (32 D vs 128 D)**: deliberate. The head should weight
  EMG primarily; IMU is supplementary information for posture
  conditioning and motion-artifact rejection.

### 5c. Fusion head

```
EMG embedding (128) ─┐
                     ├─→ concat (160) ─→ Linear(160 → 64) → ReLU
IMU embedding ( 32) ─┘                    Dropout(0.3)
                                          Linear(64 → 5)
                                                   │
                                                   ▼
                                             5 logits
```

Why:
- **Concatenation** is the simplest and most data-efficient multimodal
  fusion. With ~30 min of training data, anything fancier (cross-
  attention, gating) would add parameters without adding signal.
- **One hidden layer (64 D)**: enough to learn nonlinear cross-modal
  combinations, small enough to avoid overfitting.
- **Dropout(0.3)**: only on the head, where overfitting risk is highest.
  Conv body has very few parameters and doesn't need it.
- **Final `Linear(64 → 5)`**: 5 outputs, one per finger. Raw logits, not
  probabilities yet.

### Total parameters

~50,000. Small by modern standards, exactly because the dataset is small.
Forward pass on CPU: < 1 ms.

---

## 6. From logits to bits

Per finger:

```
prob = sigmoid(logit)            # (0, 1)
bit  = 1 if prob > 0.5 else 0    # { 0, 1 }
```

Two crucial design choices:

- **5 independent sigmoids** instead of 32-class softmax. Fingers move
  independently — bending index doesn't preclude bending middle. With
  sigmoids, the model shares features across fingers and generalizes to
  any of `2^5 = 32` combinations. With softmax it would have to learn
  each combination from scratch.
- **Bit ordering**: `[thumb, index, middle, ring, pinky]`. Same as the
  Arduino's `F<5 bits>` protocol.

---

## 7. Loss + training (`src/train_bits.py`)

### Per-finger BCE

```python
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

- **Binary Cross-Entropy** because each finger is a yes/no question.
- **`pos_weight = (1 - p) / p` per finger** where `p` is the empirical
  fraction of "1" labels for that finger. Counters class imbalance:
  fingers that are bent rarely get larger gradient when they *are* bent.
- **`BCEWithLogits`** rather than naïve sigmoid + BCE: numerically
  stable when logits are large (avoids underflow in `log(σ(z))`).

### Optimizer

| Hyperparam | Value | Why |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay; standard for small CNNs. |
| Learning rate | 1e-3 (3e-4 if `--resume`) | Default Adam; lower LR when fine-tuning to avoid clobbering existing weights. |
| Weight decay | 1e-3 | Mild L2 — conv body is small, doesn't need much. |

### Batching

| Hyperparam | Value | Why |
|---|---|---|
| Batch size | 128 | Stable BatchNorm stats, fits anywhere. |
| Window stride (training) | 10 samples | ~50 windows/sec held → enormous training set per recording. |
| Train/val split | 0.8 / 0.2, **segment-level** | Split by held pose (segment), not by window. Prevents leakage where train and val windows come from the same hold. |
| Seed | 0 | Reproducible splits. |
| Epochs | up to 60 | With early-stop patience 12. |

### Temporal consistency (optional)

The `--consistency-lambda` flag adds:

```
L_total = ½ [ BCE(f(x_t), y) + BCE(f(x_{t+Δ}), y) ] + λ · MSE(p_t, p_{t+Δ})
```

- Two windows from the same gesture-hold, time-shifted by `Δ ∈ [10, 100] ms`.
- The MSE term forces the model to give similar outputs for the two
  views — small time shifts are treated as the same control intent.
- λ ≈ 0.05–0.2 in practice. Larger collapses to constant prediction;
  smaller has no effect.
- **Effect**: stable model output frame-to-frame → less inference-time
  smoothing needed → lower live latency.
- **Cost**: trains a tiny bit slower per epoch (one extra forward + a
  small additional loss term). Inference cost: zero.

---

## 8. Label remap

A training-time label rewrite. Defined in `src/dataset_bits.py`:

```python
LABEL_REMAP = {
    "10001": "10000",
    "01110": "11111",
    "11001": "11000",
    "01111": "11111",
    "10111": "11111",
    "11011": "11111",
    "11101": "11111",
}
```

- The recorded EMG is unchanged; only the saved bit string is rewritten
  before the segment is added to the dataset.
- Reduces 14 recorded patterns → 7 canonical ones.
- Why: per-pattern val accuracy showed half the patterns were unreliable
  (< 90% exact match). Most were "near-miss" variants of a canonical
  gesture. Collapsing them gives ~3× more samples per kept gesture and
  graceful degradation when the user makes off-target poses live.
- **Result**: val mean per-bit accuracy 91.8% → **99.1%**.

Toggle with `--remap` flag in `train_bits` (off by default).

---

## 9. Smoothing pipeline at inference (`BinaryPredictor` in `src/live_binary.py`)

The model emits noisy frame-by-frame probabilities; the smoothing chain
turns them into a stable bit stream.

```
prob_raw  ─→ median(K=9)  ─→ EMA(α=0.18)  ─→ hysteresis(0.7/0.3)
                                                       │
                                                       ▼
                                          min_consecutive=3 frames
                                                       │
                                                       ▼
                                            min_dwell=0.2 s
                                                       │
                                                       ▼
                                                 latched bit
```

Stage by stage:

| Stage | Default | Why |
|---|---|---|
| `filter_pad` | 200 samples | Pre-window run-up so the offline filter doesn't have edge artifacts at the start of a buffer. |
| Median (per bit) | K=9 frames (~450 ms) | Kills 1–2 frame outliers that EMA can't catch without huge lag. |
| EMA | α=0.18 (~6-frame TC, ~300 ms) | Softens residual jitter cheaply. Lower α = smoother + more lag. |
| EMA space | logit space (default) | Sigmoid is nonlinear; averaging in logit space is more stable near 0/1 saturation. |
| Hysteresis on | 0.7 | Bit flips ON only when smoothed prob exceeds this. |
| Hysteresis off | 0.3 | Bit flips OFF only when below this. The 0.3–0.7 dead zone prevents chatter at borderline activations. |
| `min_consecutive` | 3 frames | Bit only flips after 3 consecutive frames agree. Catches anything earlier stages missed. |
| `min_dwell_s` | 0.2 s | After a flip, that finger can't flip again for 200 ms. Hard floor on flip rate. |

Every stage is independently tunable from the CLI. To run *minimal*
smoothing for diagnostics:

```
--filter-pad 0 --median-k 1 --ema-alpha 1.0
--min-consecutive 1 --min-dwell-s 0
--on-thresh 0.501 --off-thresh 0.499
```

---

## 10. Live system

```
predictor.predict()  every 50 ms
   └─ snapshot ring buffer
   └─ filter EMG
   └─ z-score EMG and IMU
   └─ model forward (< 1 ms)
   └─ smoothing pipeline
   └─ 5 latched bits
        │
        ▼
   if changed and (now - last_change[i] >= LOCK_S):
       commanded[i] = new_bit
       send "F<5 bits>" over USB serial
```

- **Forward pass**: < 1 ms on CPU.
- **End-to-end latency**: ~50–100 ms typical (50 ms hop + smoothing +
  USB RTT).
- **Per-finger lockout (Python side)**: 2 s mirror of the Arduino's
  hardware lockout, prevents serial spam.
- **Output protocol**: `F` byte + 5 ASCII bits = 6 bytes per change.

---

## 11. Hardware downstream

| Item | Value | Why |
|---|---|---|
| Servos | 5× PDI-6225MG | 300° digital servos with full finger-flexion range. |
| Pulse range | 500–2500 µs (300° span) | Default `Servo.attach()` is 544–2400 → only ~180°. We override. |
| Per-finger angles | Calibrated, e.g. thumb 95° (open) ↔ 37° (closed) | Measured with `slider_ui.py`. |
| Lockout | 2 s per finger on Arduino | A finger can't flip more than once every 2 s, no matter what the host says. Mechanical safety. |
| Pin 11 (extra) | Parked at 100° on boot, ignored by `F` protocol | Extra servo (e.g. wrist), under separate manual control. |
| MCU | Arduino UNO R4 WiFi | Hardware PWM on 6 timers, USB CDC at 115200. |
| Power | External 6 V battery, NOT USB | Servos draw 1–2 A peaks; USB-only browns out. |

---

## 12. What we deliberately do NOT do

| Skipped | Why |
|---|---|
| Cross-session calibration / domain adaptation | Out of scope; per-session norm stats are baked into the checkpoint. |
| Spatial-filter first layer (à la EEGNet) | Could squeeze a few % but adds complexity. EMG is not the bottleneck. |
| Structured output / CRF over fingers | BCE-per-finger handles unseen combinations gracefully; structured output needs more data. |
| Continuous joint-angle regression | Different model (`neuropose_*`); the binary classifier is the live demo. |
| Bigger model (transformer, RNN) | < 30 min of recordings — anything larger memorizes. |
| Cross-channel cross-talk filter | Would need many electrode placements to model; the band always sits the same way (one wear position). |

---

## 13. Cheat-line summary

```
200 ms × 14 channels in
   → two-stream CNN
       EMG: 3 conv k=5  (8→32→64→128)  pool 2,2  AvgPool→128
       IMU: 2 conv k=7,5 (6→16→32)      pool 4    AvgPool→32
   → concat 160 → Linear(160→64)→Dropout→Linear(64→5)
   → 5 sigmoids (BCE per finger, pos_weight)
   → median K=9 → EMA α=0.18 (logit) → hysteresis 0.7/0.3
                → min_consecutive=3 → min_dwell=0.2s
   → 5 bits at 20 Hz
   → USB serial @ 115200 → Arduino → 5 servos (PDI-6225MG, 0–300°)
```
