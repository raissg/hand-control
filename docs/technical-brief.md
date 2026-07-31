# Technical Brief — EMG Finger Decoder

The version to give an engineer in five minutes. Every design choice below is
stated with the reason it was made and the cost it carries.

Deeper: `docs/ai-architecture.md` (design rationale) →
`docs/neural-network-deep-dive.md` (everything, from zero).

---

## What it does

An 8-channel EMG armband reads forearm muscle activity at 500 Hz. A small 1-D
CNN maps each 200 ms window to five independent bent/extended bits — one per
finger — and an Arduino drives five servos on a 3D-printed hand. Updates at
20 Hz.

## Spec

| | |
|---|---|
| Input | 8ch EMG + 6ch IMU (3 accel, 3 gyro) @ 500 Hz, MindRove armband |
| Window | 100 samples = 200 ms, hop 50 ms → 20 Hz inference |
| Output | 5 independent bits `[thumb, index, middle, ring, pinky]` |
| Model | 3-layer 1-D CNN + 2-layer MLP head; 57.9k params (67.2k with IMU) |
| Loss | `BCEWithLogitsLoss`, per finger, `pos_weight` for imbalance |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-3, dropout 0.3, batch 128 |
| Forward pass | <1 ms, laptop CPU |
| End-to-end latency | ~0.8–1.0 s (dominated by smoothing, not compute) |
| Validation | 99.1% mean per-bit accuracy, within-session — see caveats |
| Training data | <30 min of recordings, single subject |

## Pipeline

```
armband → ring buffer (300 samples)
        → bandpass 20–240 Hz + 50 Hz notch (filtfilt), slice last 100
        → z-score with stats stored in the checkpoint
        → CNN → 5 logits
        → median(9) → EMA(α=0.18) → sigmoid       [all in logit space]
        → snap to canonical pattern OR hysteresis(0.7/0.3)
        → 3-frame debounce → 200 ms per-finger dwell
        → 5 bits → serial → Arduino
```

---

## The five decisions that matter

**1-D CNN, not an RNN or Transformer.** EMG is a burst signal with strong local
temporal structure over tens of milliseconds. Convolution is the natural
shape-detector for that, it is weight-shared so it needs few parameters, and it
runs in under a millisecond. An RNN carries state but costs more per inference;
a Transformer needs far more data than one subject's recordings provide.

**200 ms window.** EMG energy lives in 20–250 Hz, so 500 Hz sampling is
sufficient by Nyquist. 200 ms is long enough to contain a full muscle-burst
envelope and short enough to update at 20 Hz with overlapping windows.

**Five sigmoids, not a 32-class softmax.** Fingers move independently. A
softmax over 32 combinations treats `01000` and `01100` as unrelated labels and
must learn the index finger separately for each combination containing it. Five
independent binary decisions share features across fingers, so every bent-index
window teaches the index bit regardless of the other four. Cost: nothing
enforces a plausible joint output, which is what the optional snap-to-pattern
step patches at inference.

**Two-stream fusion for IMU, not 14 stacked channels.** EMG bursts last tens of
milliseconds; IMU orientation drifts over hundreds. One kernel size cannot suit
both, so each stream gets its own kernels, pooling, and batch-norm statistics
(EMG → 128-D, IMU → 32-D), concatenated into the head. IMU earns its place two
ways: the accelerometer reads gravity, so the network knows arm posture, and
gyro spikes let it discount EMG bursts that are arm motion rather than finger
flexion.

**Smoothing outside the network, not inside it.** Per-frame output flickers, and
flicker makes servos judder. Five stages sit between the model and the hardware
— median filter, EMA, hysteresis or pattern snap, frame debounce, per-finger
dwell timer — each targeting a different failure mode. Keeping them out of the
network means every one is tunable live without retraining, at the cost of the
~1 s latency below.

---

## What the numbers mean

**99.1% is within-session, per-window, per-bit.** It is measured on held-out
*segments* from the same recording sessions, so it says the model generalizes to
unseen windows, not to a new day or a re-donned armband. Cross-session
performance is untested and is the honest open question.

**The split is at segment level, not window level.** Training windows overlap by
90%, so a window-level split would put near-duplicates on both sides and produce
a meaningless score. Splitting whole pose-holds avoids that leakage.

**The 91.8% → 99.1% jump came partly from making the task easier.** Label
remapping collapses 15 recorded patterns onto 8, folding near-misses onto stable
neighbors (`01110` → `11111`). The model is no longer asked to make the
distinctions it was failing, so it can no longer be wrong about them. That is
defensible for poses this hardware could not separate reliably — but it is not
purely a modeling win.

**Latency is a deliberate purchase, not a limitation of the model.** The forward
pass is under a millisecond. The ~1 s comes from ~100 ms of window, ~250 ms of
median, ~250 ms of EMA, ≤150 ms of frame debounce, and 200 ms of dwell. Every
one of those is a CLI flag. The system is tuned for stability over speed;
`src/compare_smoothing.py` evaluates the trade on recorded data.

---

## Known limitations

- **No cross-session adaptation at run time.** Electrode drift and fatigue are
  not compensated live. The offline answer is a ~50 s calibration recording plus
  a frozen-backbone head fine-tune.
- **Labels are asserted, not measured.** The recorder displays a pose and tags
  whatever arrives; a mismatch between prompt and hand is silent label noise.
  This is the largest error source in the pipeline.
- **Single subject, <30 min of data.** The model is sized to that, deliberately.
- **Binary bits, not joint angles.** Continuous-angle work exists in `src/` but
  is not this pipeline.
- **Open bug:** the `--snap` vocabulary is derived from the old 15-pattern list
  and includes `00100`, which the current 7-pattern recorder never captures.

---

## Questions engineers ask

**How is this not overfitting on 30 minutes of data?** The model is 58k
parameters with dropout, weight decay, global average pooling instead of a
flatten, and early stopping on validation loss. The checkpoint saved is the best
validation epoch, not the last. That said, the honest answer is that
within-session validation cannot prove cross-session generalization.

**Why filter live windows on a 300-sample buffer instead of 100?** `filtfilt` is
zero-phase but produces large edge transients on short inputs. Training filtered
whole multi-second segments, so window interiors were clean. Filtering 300
samples and slicing the last 100 reproduces that distribution. All 300 samples
are in the past, so nothing acausal happens.

**Why is the EMA in logit space?** The sigmoid is steep at 0.5 and flat at the
extremes, so averaging probabilities over-weights exactly the uncertain frames
near the decision boundary. Logits are roughly linear in evidence, so averaging
them averages confidence rather than a squashed version of it.

**Why does the checkpoint carry normalization statistics?** Because they are as
much part of the trained model as the weights. EMG amplitude shifts
session-to-session with contact, sweat, and fatigue. Feeding correctly-shaped
but differently-scaled input into the network produces confident nonsense with
no visible failure.

**Why `pos_weight`?** If a finger is bent in 20% of windows, always predicting
"extended" scores 80% while learning nothing. Weighting positives by
`(1−p)/p` equalizes the two classes' contribution to the loss.

**What breaks first in the field?** Armband placement. Rotating the band a
centimeter changes which muscles sit under which electrode, and the learned
features assume the training placement. That is what the calibration path exists
for.
