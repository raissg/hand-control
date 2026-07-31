# AI Architecture

A simple, specific description of how EMG (and IMU) becomes 5 finger bits in
real time, plus the reasoning behind each design choice.

The diagram below shows `EMG5BitMulti`, the two-stream EMG+IMU model. The
EMG-only `EMG5Bit` shares the same EMG branch but has its own three-layer head
(`128 → 32 → 16 → 5`) instead of the fusion head.

For the ground-up version of everything here — every layer defined from
scratch, the full training loop, and the complete inference path — see
`docs/neural-network-deep-dive.md`.

---

## 1. The problem we're solving

Take 8 raw EMG channels (and 6 IMU channels) streaming at 500 Hz from a
MindRove armband, and decide every 50 ms which of the user's 5 fingers are
currently bent. Output: a 5-bit vector `[thumb, index, middle, ring, pinky]`,
each independently 0 (extended) or 1 (bent).

This drives an Arduino that moves 5 servos on a 3D-printed hand.

---

## 2. The architecture in one picture

```
MindRove armband
   │  500 Hz, 8 EMG ch + 6 IMU ch
   ▼
[EMGStream]  thread-safe queue, raw samples (enable_filter=False live)
   │
   ▼
[Ring buffer 300 EMG samples]  bandpass + notch (filtfilt) applied to the
   │                           whole buffer, then the last 100 sliced out.
   │                           The 200-sample pad kills filtfilt edge
   │                           transients so live input matches training.
   ▼
[Rolling 200 ms window]    100 samples × 8 EMG    +   100 samples × 6 IMU
   │                             │
   │                             ▼
   │                    ┌────────────────┐
   │                    │   IMU branch    │
   │                    │ Conv1d(6→16,k7) │
   │                    │ BN ReLU MaxPool4│
   │                    │ Conv1d(16→32,k5)│
   │                    │ BN ReLU AvgPool │
   │                    │  → 32-D vector  │
   │                    └────────┬───────┘
   ▼                              │
┌────────────────────┐             │
│   EMG branch        │             │
│ Conv1d(8→32, k=5)   │             │
│ BN ReLU MaxPool2    │             │
│ Conv1d(32→64, k=5)  │             │
│ BN ReLU MaxPool2    │             │
│ Conv1d(64→128, k=5) │             │
│ BN ReLU AvgPool     │             │
│  → 128-D vector     │             │
└────────┬────────────┘             │
         │                          │
         └──────────┬───────────────┘
                    │  concat → 160-D
                    ▼
            [MLP head]
       Linear(160→64) ReLU Dropout
            Linear(64→5)
                    │
                    ▼  5 logits
            [Smoothing pipeline]   (operates in logit space)
       median K=9 → EMA(α=0.18) → sigmoid → snap-to-pattern OR hysteresis
                  → min-consecutive → min-dwell
                    │
                    ▼
            5 latched bits → Arduino over USB serial
```

---

## 3. Components, simply

### a. EMG branch (1-D CNN)

Three Conv1d → BatchNorm → ReLU stages with a small kernel (k=5); the first
two are followed by 2× max-pooling (100 → 50 → 25 timesteps) and the third by
global average pooling. Compresses 100 EMG samples × 8 channels into a single
128-D feature vector per window. Each position in the final feature map sees
32 input samples (64 ms) before that global average.

### b. IMU branch (also 1-D CNN, smaller)

Two Conv1d stages with bigger kernels (k=7, k=5) and stronger pooling (4×).
Compresses 100 IMU samples × 6 channels into a 32-D feature vector.

### c. Fusion head (MLP)

Concatenates `[EMG-128, IMU-32]` → 160-D → `Linear(160→64) → ReLU →
Dropout → Linear(64→5)`. Output: 5 logits, one per finger.

### d. Loss (training)

Binary cross-entropy with logits, *per finger*. Each of the 5 sigmoids is
its own decision, with `pos_weight` to handle the fact that some fingers
are bent in the data more often than others.

### e. Smoothing pipeline (inference only)

```
raw logits → median(K=9) → EMA(α=0.18) → sigmoid → probs
                ↓
        snap to nearest canonical pattern   (with --snap)
          OR per-bit hysteresis(0.7/0.3)    (without --snap)
                ↓
        min_consecutive=3 frames same → commit
                ↓
        min_dwell_s=0.2 held per finger → publish to servos
```

Median and EMA both run on **logits**, not probabilities (`logit_ema=True` in
`LIVE_INFERENCE_DEFAULTS`); the sigmoid is applied once afterwards. The median
is unaffected by this choice since the sigmoid preserves order, but the EMA is:
averaging in logit space avoids the sigmoid's compression at the extremes, so a
run of confident frames carries the weight it should. Note that `on_thresh` /
`off_thresh` are unused when `--snap` is on.

### f. Label remap (training time)

Before training, the 15 patterns of the original recording protocol
(`dataset_bits.RECORDED_PATTERNS`) are collapsed onto 8:

| Recorded | Remapped to |
|---|---|
| 00000, 11111, 10000, 01000, 00100, 11000, 01100, 11100 | (kept as-is) |
| 10001 | 10000 |
| 11001 | 11000 |
| 01110, 01111, 10111, 11011, 11101 | 11111 |

The model then only ever predicts one of the 8 canonical patterns, which is
also the vocabulary `--snap` selects from.

**Caveat.** `record_bits.DEFAULT_PATTERNS` has since been trimmed to 7
patterns (`00000 11111 10000 01000 11000 01100 11100`) — the 15-pattern list
survives only as a comment. `get_supported_patterns()` still derives its
vocabulary from `RECORDED_PATTERNS`, so `--snap` can snap the output to
`00100` (middle finger alone), a pattern the current recorder never captures
and the model therefore has no data for.

---

## 4. Why each choice

### Why a 1-D CNN (and not RNN / Transformer / MLP)

EMG is a stack of time series with strong **local temporal structure** —
muscle bursts, contraction onsets that span tens of milliseconds. CNNs are
the natural shape-detector for that: weight-shared, translation-invariant
in time, tiny parameter count vs. an MLP that flattens the window. RNNs
work but are slower per inference and overkill at 200 ms windows;
transformers need much more data than a single subject's recordings
provide.

### Why 200 ms / 100 samples at 500 Hz

EMG energy lives in 20–250 Hz, so 500 Hz is enough by Nyquist. A 200 ms
window is the standard sweet spot — long enough to capture a full
muscle-burst envelope, short enough to keep latency low and let the system
update at 20 Hz with overlapping windows.

### Why a separate IMU branch (not just stack channels)

EMG and IMU live on different time scales and statistics. EMG bursts are
fast (tens of ms); IMU motion (gravity vector, slow rotation) is slow
(hundreds of ms). One Conv1d trying to model both is sub-optimal. The
two-stream design lets each branch use its own kernel sizes, pooling, and
batch-norm statistics. The head sees a clean `[emg_emb || imu_emb]` and
combines them with a couple of linear layers.

We added IMU because:
1. **Posture conditioning** — accel reads gravity, so the model knows arm
   orientation. Same gesture in different postures has different EMG, but
   IMU anchors that.
2. **Motion-artifact rejection** — gyro spikes correlate with arm motion,
   so the model can learn to discount EMG bursts that are arm movement,
   not finger flexion.

### Why 5 sigmoids + BCE (not 32-class softmax)

Fingers move independently. A 32-class softmax would treat `01000` and
`01100` as completely unrelated labels and force the model to learn each
combination from scratch. BCE-per-finger says "each finger has its own
decision," shares features across fingers, and handles unseen multi-finger
combinations more gracefully.

### Why label remap (collapse 15 patterns to 8)

Per-pattern val accuracy on the 15-pattern model showed half the patterns
were unreliable (under 90% exact-match). Most of those unreliable patterns
were near-misses of a more stable canonical pattern (e.g. `01110` vs.
`11111`). Remapping at load time:
1. Increases effective training samples per kept pattern by ~3×.
2. Makes the model robust to "almost-X" finger poses live — they get
   bucketed onto the nearest canonical gesture instead of producing
   undefined output.
3. Keeps the recordings untouched, so the remap can be turned on/off via
   a single flag.

After remap, val mean per-bit accuracy went from 91.8% to **99.1%**. Part of
that gain is the problem getting easier rather than the model getting better —
distinguishing `01110` from `11111` is no longer asked of it, so it can no
longer be wrong about it. That is a defensible product call for poses this
hardware could not separate reliably, but it is not a pure modeling win.

### Why the smoothing pipeline (not just `prob > 0.5`)

The model is trained per-frame, so its raw output flickers around
threshold-hovering activations. Without smoothing the servos would judder.
The pipeline trades latency for stability:

- **Median (K=9)** — kills 1–2 frame spikes (e.g. `0.9, 0.1, 0.9` becomes
  `0.9`). Robust to outliers: 4 of the 9 frames can be arbitrarily wrong
  without moving it.
- **EMA (α=0.18, logit space)** — softens the remaining jitter; ~250 ms time
  constant at the 50 ms hop.
- **Hysteresis (0.7 / 0.3)** — wide dead zone prevents on/off chatter for
  borderline activations. Replaced by snap-to-pattern when `--snap` is on.
- **min_consecutive=3** — the whole 5-bit vector must be identical for 3
  frames in a row before it is committed.
- **min_dwell_s=0.2** — per finger, a *new* value must hold for 200 ms before
  it is published. If the candidate flips back first, the timer restarts and
  the published bit never moves.

Each layer is independently tunable from the live CLI.

The cost is latency. Rough worst case for one clean flip: ~100 ms (window)
+ ~250 ms (median) + ~250 ms (EMA) + ≤150 ms (min_consecutive) + 200 ms
(min_dwell) ≈ **0.8–1.0 s**. `src/compare_smoothing.py` evaluates this trade
on recorded validation segments instead of by feel.

### Why per-session norm stats embedded in the checkpoint

EMG amplitude varies wildly session-to-session (electrode contact, sweat,
fatigue). The training pipeline computes per-channel mean/std from the
*train* split and stores it in the checkpoint. At inference, the same
stats are used to z-score live EMG so the model sees the input
distribution it was trained on.

### Why such a small model (58k / 67k params)

`EMG5Bit` has **57,893** parameters (53,152 in the conv stack, 4,741 in the
head). `EMG5BitMulti` has **67,157** (53,152 EMG branch + 3,376 IMU branch +
10,629 fusion head). Two reasons to keep it there:
1. **Latency** — full forward pass is <1 ms on a laptop CPU. We can run
   at 20 Hz with headroom for filtering, smoothing, and serial I/O.
2. **Data** — we have <30 minutes of recordings. A bigger model would
   memorize and not generalize. The architecture is sized to the data we
   actually have.

---

## 5. What this architecture is *not* doing

Worth being explicit about, so reviewers don't ask:

- **No cross-session adaptation at run time.** The model is trained offline;
  we don't adapt to electrode drift or muscle fatigue during a live run.
  Offline, `record_bits --calibrate` plus `train_bits --resume
  --freeze-backbone` re-fits the head and norm stats to a new band session.
- **No spatial filter on EMG.** The first Conv1d treats the 8 channels as
  input channels with one fully-connected mix per kernel position. A
  proper "spatial filter" stage (à la EEGNet) might squeeze out a few more
  percent, but isn't currently in the pipeline.
- **No structured output.** Each finger is predicted independently. There
  is no joint distribution / CRF / mechanical-coupling prior.
- **No temporal model in the network itself.** The CNN sees one 200 ms
  window per inference. All temporal smoothing happens *after* the
  network, not inside it. (Lower latency, simpler training.)
- **No hand-pose reconstruction.** This model classifies 5 binary bits
  per finger, not continuous joint angles. Continuous-angle models exist
  in `src/` (`neuropose_*`, `eval_angles*`) but they aren't part of the
  current live binary pipeline.

---

## 6. Files where this architecture lives

| Concept | File |
|---|---|
| EMG-only model class | `src/binary_model.py` (`EMG5Bit`) |
| EMG+IMU two-stream model class | `src/binary_model.py` (`EMG5BitMulti`) |
| Training loop, BCE + pos_weight | `src/train_bits.py` |
| Dataset, segmenting, label remap | `src/dataset_bits.py` |
| Recording protocol, pattern list | `src/record_bits.py` |
| EMG bandpass + notch filter | `src/convert_to_hdf5.py` (`filter_emg`) |
| Live inference + smoothing pipeline | `src/live_binary.py` (`BinaryPredictor`, `LIVE_INFERENCE_DEFAULTS`) |
| Live Arduino loop (this is the demo) | `go.py` |
| Offline smoothing / accuracy comparison | `src/compare_smoothing.py` |
| Output-noise characterization | `src/analyze_model_noise.py` |
| Full ground-up walkthrough | `docs/neural-network-deep-dive.md` |
