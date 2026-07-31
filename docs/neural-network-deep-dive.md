# The Neural Network, From Zero to Hero

This document explains the *entire* neural-network half of the project: what
the network is, why it has the shape it has, exactly what happens during
training, and exactly what happens during live inference. It assumes no prior
machine-learning knowledge and defines every term the first time it appears.

`docs/ai-architecture.md` is the one-page summary. This is the long version,
written against the actual source code rather than the summary. Where the two
disagree, section 13 lists the discrepancies.

Code discussed here:

| Concern | File |
|---|---|
| Model definitions | `src/binary_model.py` |
| Data loading, segmenting, windowing | `src/dataset_bits.py` |
| Signal filter | `src/convert_to_hdf5.py` (`filter_emg`) |
| Training loop | `src/train_bits.py` |
| Live inference + smoothing | `src/live_binary.py` |
| Recording protocol | `src/record_bits.py` |
| Hardware stream | `src/acquire.py` |
| Arduino bridge | `go.py` |

---

## Part 0 — The mental model in one paragraph

A muscle contracting produces a tiny electrical signal on the skin. An armband
with eight electrodes measures that signal 500 times per second. A neural
network is a function with adjustable numbers inside it ("parameters" or
"weights"). This project adjusts those numbers so the function maps *200
milliseconds of eight-channel electrical signal* to *five yes/no answers*: is
the thumb bent, is the index bent, and so on. "Training" is the process of
choosing the numbers by showing the function labeled examples. "Inference" is
running the finished function on live data. Everything below is detail on those
two processes.

---

## Part 1 — The task, stated precisely

### 1.1 Input

The MindRove armband streams two things at 500 Hz (500 samples per second):

- **EMG** — electromyography, 8 channels. Each channel is one electrode pair
  measuring the voltage difference caused by nearby muscle fibers firing.
  Forearm muscles move the fingers, so finger intent is visible here.
- **IMU** — inertial measurement unit, 6 channels: 3 accelerometer axes
  (`accel_x/y/z`) and 3 gyroscope axes (`gyro_x/y/z`). This measures how the
  arm itself is oriented and moving, not what the fingers are doing.

One "sample" is one instant in time: 8 EMG numbers plus 6 IMU numbers. See
`EMGSample` in `src/acquire.py`.

### 1.2 Output

Five numbers, each independently 0 or 1, in the fixed order
`[THUMB, INDEX, MIDDLE, RING, PINKY]`. `1` means bent (flexed), `0` means
extended. Written as a 5-character string: `01100` means index and middle bent,
everything else straight.

### 1.3 What kind of machine-learning problem this is

This is **multi-label binary classification**. Three terms:

- *Classification* — the output is a discrete choice, not a continuous number.
  (Predicting joint angles in degrees would be *regression*; that exists
  elsewhere in this repo but is not this model.)
- *Binary* — each individual decision has two possible answers.
- *Multi-label* — there are five such decisions, and they are made
  simultaneously and independently. This is different from *multi-class*, where
  the model picks exactly one option out of a list.

The multi-label framing matters, and section 6.3 explains why.

---

## Part 2 — From armband to training tensor

Nothing about the neural network makes sense without knowing exactly what
numbers reach it. This is the full path.

### 2.1 Recording (`src/record_bits.py`)

The recorder displays a target hand pose on screen and runs a three-phase cycle
per pattern:

```
READY (1.5 s, not recorded)  →  HOLD (4.0 s, RECORDED)  →  REST (2.0 s, not recorded)
```

During HOLD, every incoming sample is written to
`recordings/bits_<timestamp>/emg.csv` and tagged with the bit string currently
displayed. This is **self-labeling**: the label is not measured from the hand,
it is *asserted* by the prompt. If the pose displayed and the pose actually held
disagree, the label is wrong and the model learns the wrong thing. This is the
single largest source of label noise in the project.

The current `DEFAULT_PATTERNS` set is seven patterns:

```
00000  11111  10000  01000  11000  01100  11100
```

With `--repeats 3` and a 4-second hold, that is 7 × 3 × 4 = 84 seconds of
recorded signal per session, or roughly 42,000 samples.

Because REST is not recorded, the CSV contains **time gaps**. The timestamp
column jumps forward by several seconds between holds. This matters in the next
step.

### 2.2 Segmentation (`load_sessions` in `src/dataset_bits.py`)

A **segment** is one contiguous stretch of samples that (a) all carry the same
5-bit label and (b) contain no time gap larger than `gap_s` (default 0.1 s). A
new segment is started wherever the label changes or a gap appears:

```python
label_changed = np.any(np.diff(bits, axis=0, prepend=bits[0:1]) != 0, axis=1)
gap = dt > gap_s
boundaries = np.where(label_changed | gap)[0]
```

Segments shorter than `min_seg_samples` (100) are discarded, since they are too
short to filter or to yield even one window.

The segment is the atomic unit of this pipeline. Everything downstream —
filtering, windowing, the train/validation split — operates per segment.

### 2.3 Filtering (`filter_emg` in `src/convert_to_hdf5.py`)

Raw EMG contains three things the model should not see:

1. **Slow drift** below ~20 Hz — electrode polarization, skin movement, sweat.
   Not muscle activity.
2. **Mains hum** at 50 Hz — the building's electrical wiring radiating into the
   electrodes. Measured on this rig with `src/check_spectrum.py`; the code notes
   that a 60 Hz notch left a 12× spike untouched, so 50 Hz is correct here.
3. **Content above 240 Hz** — beyond the useful EMG band, and above 250 Hz it is
   aliasing anyway (Nyquist limit for 500 Hz sampling is 250 Hz).

So each segment is passed through a 4th-order Butterworth **bandpass** (20–240
Hz) and then a 50 Hz **notch** filter (a narrow band-stop that removes one
frequency).

Two terms:

- *Butterworth* — a filter design with maximally flat response inside the
  passband, i.e. it does not distort the frequencies it keeps.
- *`filtfilt`* — applies the filter forward through the signal, then backward
  through the result. Running it twice cancels the phase shift a filter normally
  introduces, so peaks stay at their true times. The cost is that it is
  **non-causal**: the output at time *t* depends on samples after *t*. That is
  fine offline, and section 9.2 explains how live inference handles it.

The whole segment is filtered as one block, deliberately. Filtering across a
time gap would smear the end of one pose into the start of the next, which is
why segmentation happens first.

**IMU is not filtered.** Its signal *is* the low-frequency content — gravity
shows up as a DC offset in the accelerometer, and slow wrist rotation shows up
in the gyroscope. A 20 Hz highpass would delete exactly the information the IMU
branch exists to provide.

### 2.4 Train/validation split (`split_segments`)

The dataset is split into two disjoint sets:

- **Training set** — the examples the model learns from.
- **Validation set** — examples held back, used only to measure whether the
  model generalizes to data it did not learn from.

The split is done at the **segment** level, not the window level, with
`val_frac=0.2` (20% of segments held out). This is critical. Windows overlap
heavily (see 2.6), so two windows from the same segment share most of their
samples. Splitting at the window level would put near-duplicates on both sides
and produce a validation score that looks excellent while meaning nothing. That
failure is called **leakage**, and splitting by segment avoids it.

### 2.5 Normalization (`compute_norm_stats`, `compute_imu_norm_stats`)

Neural networks train poorly when different input channels have wildly
different magnitudes. The fix is **z-scoring**, also called standardization:

```
x_normalized = (x - mean) / std
```

after which each channel has mean 0 and standard deviation 1.

The mean and standard deviation are computed **per channel**, over the
concatenated **training** segments only — never the validation segments, since
using validation data to compute anything that affects training is a form of
leakage.

EMG and IMU get **separate** statistics. Their natural scales are unrelated:
accelerometer readings sit near ±9.8 m/s² with a large constant gravity
component, gyroscope readings are radians per second centered near zero, and EMG
is microvolts. Pooling them into one set of statistics would let the larger
scale dominate. Z-scoring the accelerometer per channel also conveniently
subtracts the gravity offset while keeping its variation.

These statistics are saved inside the checkpoint (section 8) because live
inference must reproduce the exact same transformation.

### 2.6 Windowing (`BitsWindowDataset`)

The model does not consume a whole segment. It consumes a fixed-length
**window**: `window=100` samples = 200 ms at 500 Hz.

Windows are extracted by sliding along each segment with stride `step`:

```python
for start in range(0, T - window + 1, step):
    self.index.append((si, start))
```

- **Training** uses `step=10` (20 ms). Consecutive windows overlap by 90%. One
  4-second hold therefore yields roughly 190 training windows instead of 20.
  This is a cheap form of **data augmentation** — more examples from the same
  recording, each a slightly different view.
- **Validation** uses `step=window` (no overlap), so the validation score counts
  each piece of signal once rather than inflating it with near-duplicates.

Every window inherits its segment's label. This encodes an assumption worth
naming: **the pose is constant for the whole hold**, including the transition at
the very start where the hand is still closing. Some early windows are therefore
mislabeled by construction.

### 2.7 The final tensor shapes

PyTorch convolutions expect channels before time, so the window is transposed
from `(time, channels)` to `(channels, time)`:

```python
return torch.from_numpy(w).float().transpose(0, 1)   # (8, 100)
```

A **tensor** is just a multi-dimensional array. `B` below is the **batch size**:
how many independent examples are processed at once.

| Tensor | Shape | Meaning |
|---|---|---|
| `emg` | `(B, 8, 100)` | B windows, 8 EMG channels, 100 timesteps |
| `imu` | `(B, 6, 100)` | same time span, 6 IMU channels |
| `label` | `(B, 5)` | five 0.0/1.0 targets per window |
| `logits` | `(B, 5)` | raw model output, see section 6.1 |

---

## Part 3 — Neural-network building blocks, defined

Every layer used in this project, explained from scratch. Skip to Part 4 if
these are already familiar.

### 3.1 `Conv1d` — 1-D convolution

A convolution slides a small set of weights (a **kernel** or **filter**) along
the time axis and computes a weighted sum at each position. With
`Conv1d(in_channels=8, out_channels=32, kernel_size=5)`, the kernel spans 5
timesteps and all 8 input channels at once, and there are 32 different kernels,
each producing its own output channel:

```
output[b, o, t] = bias[o] + Σ_i Σ_j  W[o, i, j] · x[b, i, t + j - padding]
```

where `i` runs over the 8 input channels and `j` over the 5 kernel positions.

Two properties make this the right choice for EMG:

- **Weight sharing** — the same kernel is applied at every time position, so a
  muscle-burst shape is recognized regardless of *when* in the window it occurs.
  This is called translation invariance, and it also means far fewer parameters
  than a layer that treats each timestep separately.
- **Locality** — each output depends only on a short slice of input, which
  matches the physics: a muscle burst is a local event lasting tens of
  milliseconds.

`padding=2` with `kernel_size=5` adds two zeros at each end so the output has
the same length as the input, which keeps the shape arithmetic simple.

### 3.2 `BatchNorm1d` — batch normalization

For each channel, subtract the mean and divide by the standard deviation
computed across the batch **and** across time, then apply a learned scale and
shift (two parameters per channel).

Why: as training changes early layers, the distribution of values arriving at
later layers keeps shifting, which forces them to constantly re-adapt.
Normalizing at each stage stabilizes this and permits a higher learning rate.

Batch normalization behaves differently in two modes, and this difference causes
real bugs:

- **Training mode** (`model.train()`) — uses the statistics of the current
  batch, and updates a **running average** of them.
- **Evaluation mode** (`model.eval()`) — uses the stored running average, so a
  single input produces the same output regardless of what else is in the batch.

Live inference runs one window at a time, so eval mode is mandatory. A batch of
one has zero variance within itself, and forgetting `model.eval()` would produce
garbage.

### 3.3 `ReLU` — rectified linear unit

`ReLU(x) = max(0, x)`. Without a non-linear function between layers, stacking
linear layers collapses mathematically into a single linear layer and the
network can only fit straight lines. ReLU is the standard choice: cheap, and
its gradient is exactly 1 for positive inputs, which avoids the vanishing
gradients that plague saturating functions.

`inplace=True` overwrites the input array instead of allocating a new one — a
memory optimization with no effect on the math.

### 3.4 `MaxPool1d(2)` — max pooling

Split the time axis into non-overlapping pairs and keep the larger value,
halving the length. This throws away precise timing while keeping "a strong
activation happened around here", and it doubles the time span each subsequent
kernel covers. For EMG the exact millisecond of a burst does not matter, so this
is a good trade.

### 3.5 `AdaptiveAvgPool1d(1)` — global average pooling

Average over the entire remaining time axis, collapsing `(B, C, T)` to
`(B, C, 1)`. Two consequences:

- The output size no longer depends on the input length, so the same weights
  work for any window length. The IMU branch's docstring calls this out
  explicitly: "the AdaptiveAvgPool at the end makes `T_imu` free."
- The window is summarized by *average* activation rather than by a flattened
  copy of everything, which is far fewer parameters and much harder to overfit.

### 3.6 `Linear` — fully connected layer

`y = Wx + b`. Every output is a weighted sum of every input.
`Linear(128, 32)` holds a 128×32 weight matrix plus 32 biases.

### 3.7 `Dropout(0.3)` — dropout regularization

During training only, randomly zero 30% of the values in a layer's output on
every forward pass (and scale the rest up to compensate). This prevents the
network from relying on any single feature, since that feature might vanish. It
is a **regularizer**: something that makes training harder in order to make
generalization better. During evaluation it is disabled entirely — another
reason `model.eval()` matters.

---

## Part 4 — Architecture 1: `EMG5Bit` (EMG only)

```
Input  (B, 8, 100)          8 EMG channels, 200 ms

 feat:
   Conv1d(8 → 32,  k=5, pad=2)  → (B,  32, 100)
   BatchNorm1d(32) → ReLU
   MaxPool1d(2)                 → (B,  32,  50)

   Conv1d(32 → 64, k=5, pad=2)  → (B,  64,  50)
   BatchNorm1d(64) → ReLU
   MaxPool1d(2)                 → (B,  64,  25)

   Conv1d(64 → 128, k=5, pad=2) → (B, 128,  25)
   BatchNorm1d(128) → ReLU
   AdaptiveAvgPool1d(1)         → (B, 128,   1)

 head:
   Flatten                      → (B, 128)
   Linear(128 → 32) → ReLU
   Dropout(0.3)
   Linear(32 → 16)  → ReLU
   Linear(16 → 5)               → (B, 5)   logits
```

**Parameter count: 57,893** — 53,152 in the convolutional feature extractor,
4,741 in the head.

### 4.1 The shape of the design

Channels go up (8 → 32 → 64 → 128) while time goes down (100 → 50 → 25 → 1).
This is the standard convolutional funnel. Early layers hold a lot of temporal
detail about a few things; later layers hold many abstract features with little
temporal detail. The network progressively trades "when" for "what".

### 4.2 Receptive field

The **receptive field** is how many input samples influence one value at a given
depth. Tracking it through the stack:

| After | Receptive field | Stride | In milliseconds |
|---|---|---|---|
| Conv 1 | 5 | 1 | 10 ms |
| MaxPool 1 | 6 | 2 | 12 ms |
| Conv 2 | 14 | 2 | 28 ms |
| MaxPool 2 | 16 | 4 | 32 ms |
| Conv 3 | 32 | 4 | 64 ms |
| Global average pool | 100 (all) | — | 200 ms |

So each of the 25 positions in the final feature map summarizes a 64 ms slice,
adjacent positions are 8 ms apart, and global pooling averages all 25. 64 ms is
a reasonable span for a single muscle-activation burst, which is the shape the
network needs to detect.

---

## Part 5 — Architecture 2: `EMG5BitMulti` (EMG + IMU)

The two-stream model. Selected by `train_bits --imu`, and detected
automatically at inference from the checkpoint's `multistream` flag.

```
   emg (B, 8, 100)                      imu (B, 6, 100)
        │                                     │
   Conv1d(8→32, k=5) BN ReLU            Conv1d(6→16, k=7) BN ReLU
   MaxPool(2)                           MaxPool(4)          → (B,16,25)
   Conv1d(32→64, k=5) BN ReLU
   MaxPool(2)                           Conv1d(16→32, k=5) BN ReLU
   Conv1d(64→128, k=5) BN ReLU          AdaptiveAvgPool(1)
   AdaptiveAvgPool(1)                   Flatten             → (B, 32)
   Flatten          → (B, 128)                │
        │                                     │
        └──────────── concat ─────────────────┘
                        │  (B, 160)
              Linear(160 → 64) → ReLU
              Dropout(0.3)
              Linear(64 → 5)    → (B, 5) logits
```

**Parameter count: 67,157** — 53,152 EMG branch, 3,376 IMU branch, 10,629 head.

### 5.1 Why two separate branches instead of 14 stacked channels

The naive approach is to concatenate EMG and IMU into one 14-channel input.
That is worse for three reasons:

1. **Different timescales.** EMG bursts last tens of milliseconds; IMU
   orientation changes over hundreds. A single kernel size cannot be optimal for
   both. The separate branches use `k=5` with 2× pooling for EMG and `k=7` with
   4× pooling for IMU, so the IMU branch reaches a wide time span in fewer
   layers.
2. **Different statistics.** Batch normalization computes per-channel statistics
   from whatever is in the layer. Keeping the streams separate means each
   branch's normalization sees only its own kind of signal.
3. **Different information density.** EMG carries the actual finger signal and
   gets 128 dimensions; IMU is contextual and gets 32. Stacking channels would
   implicitly give them equal weight in every kernel.

### 5.2 Why IMU is included at all

- **Posture conditioning.** The accelerometer reads the gravity vector, which
  tells the model how the forearm is oriented. The same gesture produces
  different EMG in different arm postures; the IMU gives the network a way to
  account for that rather than treating it as noise.
- **Motion-artifact rejection.** Moving the arm both jostles the electrodes
  (producing EMG-looking artifacts) and spikes the gyroscope. Giving the network
  the gyroscope signal lets it learn to discount EMG bursts that coincide with
  arm motion.

### 5.3 Late fusion

Concatenating two per-stream embeddings and letting a small MLP combine them is
called **late fusion** — each stream is processed independently and merged only
at the end. The alternative, **early fusion**, merges raw channels at the input.
Late fusion is the right choice when the streams have different structure, which
they do here.

---

## Part 6 — The loss function

The **loss** is a single number measuring how wrong the model currently is.
Training means adjusting parameters to reduce it.

### 6.1 Logits and the sigmoid

The network's five outputs are **logits** — unbounded real numbers. A logit is
converted to a probability by the **sigmoid** function:

```
σ(z) = 1 / (1 + e^(-z))
```

which maps `(-∞, +∞)` to `(0, 1)`. `σ(0) = 0.5`, large positive logits approach
1, large negative approach 0. The inverse is the log-odds:
`logit(p) = log(p / (1 - p))`.

The network deliberately outputs logits rather than probabilities, for the
numerical-stability reason in 6.2.

### 6.2 Binary cross-entropy

For one bit with target `y ∈ {0, 1}` and predicted probability `p`:

```
L = -[ y · log(p) + (1 - y) · log(1 - p) ]
```

When `y = 1` this is `-log(p)`: zero if the model said 1.0, growing without
bound as the prediction approaches 0. Confident *and* wrong is punished
severely, which is the property that makes cross-entropy train well.

The code uses `nn.BCEWithLogitsLoss`, which takes logits directly rather than
probabilities. This is not a convenience — it is numerically necessary. Applying
sigmoid then log separately computes `log(σ(z))`, and for very negative `z` the
sigmoid underflows to exactly 0.0 and the log returns `-inf`. Fusing the two
operations lets PyTorch use the log-sum-exp identity and stay finite for any
input.

The five per-bit losses are averaged into one number.

### 6.3 Why five sigmoids rather than one 32-way softmax

Five bits have 2⁵ = 32 possible combinations, so a multi-class model with 32
mutually exclusive outputs is possible. It is a much worse fit:

- A softmax over 32 classes treats `01000` and `01100` as **unrelated labels**.
  It cannot know they share the index finger, so it must learn the index-finger
  EMG pattern separately for every combination containing it.
- The independent-sigmoid formulation shares all features across the five
  decisions, so every window containing a bent index finger teaches the index
  bit, regardless of what the other fingers were doing.
- Unseen combinations degrade gracefully. A softmax must assign an unseen
  combination near-zero probability; independent bits can compose an output they
  were never explicitly trained on.

The cost is that nothing prevents anatomically implausible outputs, since there
is no joint model over the five bits. Section 10.2's snap-to-pattern step is the
inference-time patch for that.

### 6.4 `pos_weight` — class imbalance

If a finger is bent in only 20% of training windows, a model that always
predicts "extended" for it is 80% accurate while learning nothing.
`pos_weight` counteracts this by scaling up the loss contribution of positive
examples:

```python
pos_frac  = (labels * weights[:, None]).sum(axis=0) / weights.sum()
pos_weight = [(1 - p) / max(p, 1e-3) for p in pos_frac]
```

A finger bent 20% of the time gets `0.8 / 0.2 = 4.0`, so each positive example
counts four times as much and the two classes contribute equally on average.
Note that `pos_frac` is weighted by segment length, so a long hold counts more
than a short one — the fraction reflects samples, not segments.

`pos_frac` is printed at the start of training and is the single most useful
diagnostic in the log. A value near 0 or 1 for any finger means the recording
protocol never varied that finger, and no amount of model tuning will fix it.

---

## Part 7 — The training loop

### 7.1 How learning actually works

Four steps, repeated for every batch:

1. **Forward pass** — run the batch through the network, get logits, compute the
   loss.
2. **Backward pass** (`loss.backward()`) — compute the **gradient** of the loss
   with respect to every parameter: for each of the 57,893 numbers, which
   direction and how strongly would changing it increase the loss. This is done
   by **backpropagation**, which is the chain rule from calculus applied
   backwards through the layers, reusing intermediate results so the whole thing
   costs about the same as one forward pass.
3. **Optimizer step** (`opt.step()`) — nudge every parameter a small distance in
   the direction that decreases the loss.
4. **Zero the gradients** (`opt.zero_grad()`) — PyTorch accumulates gradients by
   default, so they must be cleared before the next batch.

One pass over the whole training set is an **epoch**.

### 7.2 AdamW

```python
opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)
```

Plain gradient descent uses the same step size for every parameter. **Adam**
keeps a running estimate of each parameter's recent gradient mean and variance,
and scales that parameter's step accordingly — parameters with consistently
small gradients take larger steps. This makes it far less sensitive to the
initial learning rate.

**Weight decay** (`1e-3`) pulls every parameter slightly toward zero on every
step. This is a regularizer: it discourages any single weight from becoming
large, which tends to produce smoother functions that generalize better. The
"W" in AdamW refers to applying this decay directly to the weights rather than
folding it into the gradient, which is the mathematically correct version when
using Adam.

**Learning rate** (`lr=1e-3`) sets the step size. Too large and training
oscillates or diverges; too small and it takes forever or gets stuck. Resuming
from a checkpoint automatically drops it to `3e-4`, because a partly-trained
model needs finer adjustments than a random one.

**Batch size** (`128`) is how many windows are averaged into each gradient
estimate. Larger batches give less noisy gradients but fewer updates per epoch.

### 7.3 The training half of an epoch

```python
model.train()
for batch in train_loader:
    x, y = batch                       # or x, x_imu, y with --imu
    loss = loss_fn(forward_model(x, x_imu), y)
    opt.zero_grad(); loss.backward(); opt.step()
```

`model.train()` puts batch normalization into batch-statistics mode and enables
dropout. `DataLoader(..., shuffle=True, drop_last=True)` reshuffles every epoch
so the model never sees the same batch composition twice, and discards the final
partial batch (a very small last batch would give an unusually noisy gradient,
and would make batch-norm statistics unreliable).

### 7.4 The validation half of an epoch

```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        logits = forward_model(x, x_imu)
        vl += loss_fn(logits, y).item()
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == y).sum(dim=0).cpu().numpy()
```

- `model.eval()` switches batch norm to running statistics and disables dropout.
- `torch.no_grad()` skips building the graph needed for backpropagation, since
  nothing is being trained here. Faster, and uses less memory.
- Validation accuracy uses a plain `> 0.5` threshold with **no smoothing**. The
  reported number is therefore per-window, per-bit accuracy in isolation. Live
  behavior is better than this number suggests, because the smoothing pipeline
  suppresses isolated errors — the two numbers are not directly comparable.

The log line reports per-bit accuracy for all five fingers plus the mean:

```
Ep  12: train 0.1841 | val 0.2013 | per-bit acc [0.97, 0.99, 0.98, 0.96, 0.98] | mean 0.976
```

### 7.5 Checkpointing and early stopping

```python
if vl < best_val:
    best_val = vl; stall = 0
    torch.save(ckpt_payload, save_path)
else:
    stall += 1
    if not args.no_early_stop and stall >= args.patience:
        break
```

The checkpoint is written **only when validation loss improves**, so the saved
file is always the best-generalizing version seen, not the final epoch. If
validation loss fails to improve for `patience=12` consecutive epochs, training
stops early.

This is the defense against **overfitting**: the point where the model starts
memorizing training examples rather than learning the underlying pattern.
Overfitting shows up as training loss continuing to fall while validation loss
turns around and rises.

Note that the selection criterion is validation *loss*, not accuracy. Loss is
sensitive to confidence, so a model that is right just as often but less
confident scores worse and will not be saved.

### 7.6 The temporal-consistency regularizer

Enabled with `--consistency-lambda`, off by default.

The problem it targets: the model is trained on independent windows and has no
notion of time beyond the 200 ms it sees. Two windows 40 ms apart during the
same steady hold can produce noticeably different outputs, which appears live as
jitter.

The fix is to explicitly penalize that. When `pair_delta_range` is set, the
dataset returns *two* windows from the same segment, offset by a random Δ drawn
from `--consistency-deltas` (default 5–50 samples = 10–100 ms):

```python
emg_cat = torch.cat([x, x2], dim=0)          # both views, one forward pass
logits  = forward_model(emg_cat, imu_cat)
l1, l2  = logits[:x.size(0)], logits[x.size(0):]
bce  = 0.5 * (loss_fn(l1, y) + loss_fn(l2, y))
cons = ((torch.sigmoid(l1) - torch.sigmoid(l2)) ** 2).mean()
loss = bce + lam * cons
```

The consistency term is the mean squared difference between the two
**probability** vectors. It is zero when the model gives identical answers to
both views and grows as they diverge, so minimizing it pushes the model toward
temporal stability without needing any new labels. `λ` (`--consistency-lambda`)
sets how much this matters relative to being correct; 0.05–0.2 is the suggested
range.

Two implementation details worth knowing:

- Both views go through **one** forward pass via concatenation. This is faster,
  but it also means batch normalization computes its statistics over both views
  together, which couples them slightly.
- If a segment is too short for the offset, Δ is clamped, possibly to 0. The
  pair is then identical and the consistency term is exactly zero — harmless,
  just a wasted contribution.

### 7.7 Resume and calibration fine-tuning

Three related flags for adapting an existing model instead of training fresh:

**`--resume <ckpt>`** loads the weights and, critically, **reuses the
checkpoint's normalization statistics and window length**. The weights were
learned against a specific input distribution; recomputing the statistics on new
data would silently shift that distribution out from under them.

**`--freeze-backbone`** freezes the convolutional feature extractors and trains
only the head:

```python
frozen_modules = [model.emg_feat, model.imu_feat]   # or [model.feat]
for m in frozen_modules:
    for p in m.parameters():
        p.requires_grad = False
```

This is a **linear probe**: keep the learned feature representation, relearn
only how to map it to decisions. It works with far less data than full training,
because there are only ~10k trainable parameters instead of ~67k, and it is the
right tool when the features are still valid but the mapping has shifted.

Two subtleties in that block:

- `m.eval()` is called on the frozen modules at the top of **every** epoch, even
  though the model as a whole is in `train()` mode. Freezing parameters does not
  freeze batch-norm *running statistics*, which update during any forward pass in
  training mode. Without this line, a small calibration set would drag those
  statistics away from their pretrained values and corrupt the frozen features.
- When `--freeze-backbone` is combined with `--resume`, normalization statistics
  **are** recomputed from the new session, overriding the inherited ones. This
  looks like it contradicts the paragraph above, but it is the entire point of
  calibration: a re-donned armband produces a different amplitude distribution,
  and rescaling the input to match what the frozen features expect is exactly
  what needs to happen.

The intended workflow is `record_bits --calibrate` (one repeat, ~50 seconds) then
`train_bits --resume <ckpt> --freeze-backbone`.

### 7.8 Label remapping

Enabled with `--remap`. Applied in `load_sessions`, so it rewrites labels at
load time and never touches the recordings on disk.

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

`dataset_bits.RECORDED_PATTERNS` lists the 15 patterns from the original
recording protocol; applying the remap collapses them to **8** distinct
patterns:

```
00000  00100  01000  01100  10000  11000  11100  11111
```

The rationale, from `docs/ai-architecture.md`: per-pattern validation showed
about half the 15 patterns were unreliable, and most of the unreliable ones were
near-misses of a more stable neighbor (`01110` differs from `11111` by two
bits). Collapsing them roughly triples the effective training data per remaining
pattern and makes "almost-X" poses resolve to X rather than to something
undefined. The reported effect was mean per-bit validation accuracy rising from
91.8% to 99.1%.

The honest caveat: this improves the metric partly by making the problem easier.
The model is no longer asked to distinguish `01110` from `11111`, so it can no
longer be wrong about that distinction. That is a legitimate product decision —
those poses were not reliably distinguishable with this hardware — but it is not
purely a modeling improvement.

---

## Part 8 — The checkpoint

```python
ckpt_payload = {
    "state_dict":  model.state_dict(),   # all learned parameters
    "input_mean":  mean,                 # (8,) EMG per-channel mean
    "input_std":   std,                  # (8,) EMG per-channel std
    "window":      args.window,          # 100
    "multistream": bool(args.imu),       # which architecture
}
if args.imu:
    ckpt_payload["imu_mean"] = imu_mean  # (6,)
    ckpt_payload["imu_std"]  = imu_std   # (6,)
```

The design principle here is that **a checkpoint is self-describing**. It
carries not only the weights but everything needed to reconstruct the exact
input pipeline they were trained against. Live inference reads `multistream` to
decide which class to instantiate, `window` to size its ring buffer, and the
normalization statistics to preprocess incoming data identically.

This matters because normalization statistics are as much a part of the trained
model as the weights are. Feeding correctly-shaped but differently-scaled input
into a trained network produces confident nonsense, and nothing about the
failure looks like a bug.

---

## Part 9 — Live inference: getting to a probability

Everything from here lives in `src/live_binary.py`, primarily
`BinaryPredictor.predict()`.

### 9.1 Loading

```python
self.multistream = bool(is_dict and ckpt.get("multistream", False))
if self.multistream:
    self.model = EMG5BitMulti(...)
else:
    self.model = EMG5Bit(...)
self.model.load_state_dict(state)
self.model.eval()
```

The architecture is chosen from the checkpoint rather than from a command-line
flag, so a mismatch is impossible. `model.eval()` is essential for the batch-norm
and dropout reasons in sections 3.2 and 3.7. Multistream checkpoints missing IMU
statistics raise immediately rather than silently running with wrong scaling.

### 9.2 Buffering and the filter pad

```python
emg_buf_len = self.window + self.filter_pad     # 100 + 200 = 300
self.buffer = collections.deque(maxlen=emg_buf_len)
```

A background thread continuously drains the armband queue into a **ring buffer**
(a fixed-size deque that discards the oldest sample when full), so the buffer
always holds the most recent 300 EMG samples. In multistream mode
`push_emg_imu` advances both buffers in lockstep, guaranteeing the EMG and IMU
windows cover the same time range.

The `filter_pad` of 200 extra samples exists because of `filtfilt`. Filtering a
100-sample window in isolation produces large **edge transients** — distortion at
the start and end where the filter has no history to work with. Training filtered
whole multi-second segments, so its window interiors were clean. Filtering the
longer 300-sample buffer and then slicing off the trailing 100 samples gives the
model input that matches the training distribution:

```python
arr_full = filter_emg(arr_full, fs=SAMPLE_RATE)   # filter 300
arr = arr_full[-self.window:]                     # keep the last 100
```

`filtfilt` is non-causal, but all 300 samples are already in the past, so no
future information is used — this is not cheating, just a 200-sample buffer of
history.

Note also that the live stream is opened with `EMGStream(enable_filter=False)`.
The hardware layer's own real-time filter is deliberately bypassed so that
exactly one filter — the same `filter_emg` used in training — is applied.

### 9.3 Normalize and forward

```python
arr = (arr - self.input_mean) / self.input_std
x = torch.from_numpy(arr).float().transpose(0, 1).unsqueeze(0)   # (1, 8, 100)
logits_raw = self.model(x)[0].cpu().numpy()                       # (5,)
```

`unsqueeze(0)` adds the batch dimension, since the model always expects a batch
even when there is one example. The whole `predict()` method is decorated with
`@torch.no_grad()`.

At roughly 57–67k parameters this forward pass costs well under a millisecond on
a laptop CPU, which is what makes the 20 Hz loop comfortable.

---

## Part 10 — Live inference: probability to committed bits

The raw model output is a per-frame opinion with no memory. Applying `> 0.5`
directly would make the servos judder every time a probability crossed the
threshold. Five stages sit between the model and the hardware, each addressing a
different failure mode.

Defaults come from `LIVE_INFERENCE_DEFAULTS`:

```python
{
  "on_thresh": 0.7, "off_thresh": 0.3,
  "ema_alpha": 0.18, "median_k": 9, "logit_ema": True,
  "min_consecutive": 3, "min_dwell_s": 0.2,
  "filter_pad": 200,
}
```

At the default 50 ms hop the loop runs at 20 frames per second, so "frames" and
"50 ms" are interchangeable below.

### 10.1 Stage 1 — median filter (`median_k=9`)

Keep the last 9 frames and take the element-wise median per bit.

The median is the middle value of a sorted list, and it is **robust to
outliers**: up to 4 of the 9 frames can be arbitrarily wrong without moving it
at all. A mean would be dragged by a single extreme spike. This kills isolated
one- and two-frame glitches at the source, before they reach anything with
memory.

Cost: for a genuine state change to pass through, 5 of the 9 frames must be new,
so a step is delayed by about 250 ms.

The median is applied in whichever space the EMA runs in. Because the sigmoid is
strictly monotonic, it preserves order, so the median of the logits maps exactly
to the median of the probabilities — the choice is arbitrary and only matters for
keeping the buffer consistent.

### 10.2 Stage 2 — exponential moving average (`ema_alpha=0.18`, `logit_ema=True`)

```python
self._logits_ema = a * smooth_med + (1.0 - a) * self._logits_ema
probs = 1.0 / (1.0 + np.exp(-self._logits_ema))
```

An **EMA** blends each new value with the running estimate, weighting the new
sample by `α`. Every past frame still contributes, with exponentially decaying
weight. `α = 0.18` gives a time constant of about `-50ms / ln(0.82) ≈ 250 ms`,
i.e. roughly a 5–6 frame effective average. Smaller α means smoother and slower.

The averaging happens in **logit space**, then the sigmoid is applied once at the
end. This is the more subtle choice in the pipeline. The sigmoid is steepest at
p = 0.5 and flat at the extremes, so averaging probabilities compresses
information exactly where the decision is being made: a run of `0.98`s barely
moves the average, while noise around 0.5 moves it a lot. Logits are unbounded
and roughly linear in evidence, so averaging them corresponds to averaging the
underlying confidence rather than a squashed version of it. The pipeline still
smooths near the boundary, but no longer over-weights it.

### 10.3 Stage 3a — hysteresis (used when snap is off)

```python
if p > self.on_thresh:     self._bits_hyst[i] = 1
elif p < self.off_thresh:  self._bits_hyst[i] = 0
# else: keep the previous value
```

**Hysteresis** means the threshold for turning on differs from the threshold for
turning off, creating a dead zone (0.3 to 0.7) where the bit simply holds its
previous state. With a single threshold, a probability hovering near it flips the
output on every frame. With hysteresis, crossing 0.7 latches on, and it takes a
fall all the way below 0.3 to release. This is the same principle as a
thermostat's deadband.

### 10.4 Stage 3b — snap to a supported pattern (`--snap`)

```python
diffs = self._sup_patterns - probs[None, :]
dists = np.sum(diffs * diffs, axis=1)
cand  = self._sup_patterns[int(np.argmin(dists))]
```

Instead of five independent decisions, choose the whole 5-bit vector from a fixed
vocabulary — the canonical patterns from `get_supported_patterns()` — by
Euclidean distance to the probability vector.

This reintroduces the joint structure that independent sigmoids threw away
(section 6.3). The model can no longer emit a combination it was never trained
on; single-bit chatter that would take the output out of vocabulary is snapped
back. When snap is enabled, `on_thresh` and `off_thresh` are unused.

A precision note on the code comment claiming this is "equivalent to picking the
most likely class under an independent-bit Bernoulli model". Expanding the
squared distance, minimizing `Σ(pᵢ - bᵢ)²` over binary vectors is equivalent to
maximizing `Σ bᵢ·(2pᵢ - 1)`, whereas the Bernoulli maximum-likelihood choice
maximizes `Σ bᵢ·logit(pᵢ)`. Both weights are increasing in `pᵢ` and share a sign
at `pᵢ = 0.5`, so the two rules agree in the overwhelming majority of cases —
including all comparisons between candidates with the same number of ones. They
can differ when comparing candidates of different weight, because the logit rule
weights near-certain bits far more heavily. For example with `p = [0.7, 0.7,
0.12]`, the L2 rule prefers `[1,1,1]` while the Bernoulli rule prefers `[0,0,0]`.
The distinction is minor in practice but the two are not identical.

### 10.5 Stage 4 — N-frame debounce (`min_consecutive=3`)

The candidate **vector** must be identical for 3 consecutive frames before it is
committed; otherwise the previously committed vector keeps being returned. This
is vector-level: any single bit differing restarts the count. Adds up to 150 ms.

### 10.6 Stage 5 — per-finger minimum dwell (`min_dwell_s=0.2`)

The final gate, and unlike stage 4 it is **per finger** and **time-based** rather
than frame-based:

```python
if target == int(self._dwell_published[i]):
    self._dwell_pending[i] = target; self._dwell_since[i] = now; continue
if target != int(self._dwell_pending[i]):
    self._dwell_pending[i] = target; self._dwell_since[i] = now; continue
if (now - self._dwell_since[i]) >= self.min_dwell_s:
    self._dwell_published[i] = target
```

Three cases per finger: the candidate matches what is published (reset the
timer), the candidate changed while waiting (restart the timer), or the candidate
has been stable and different for long enough (publish it). Because each finger
has its own timer, the thumb can update while the ring finger is still settling.

Being time-based rather than frame-based makes this robust to a variable loop
rate — if a frame is dropped or the loop stalls, a frame counter would be fooled
but a wall-clock timer would not. It uses `time.monotonic()`, a clock that only
moves forward and is immune to system time adjustments.

### 10.7 The latency budget

Every stage buys stability with delay. Approximate worst case for one clean
finger flip:

| Source | Delay |
|---|---|
| Window (model sees 200 ms of history) | ~100 ms average |
| Median filter, K = 9 | ~250 ms |
| EMA, α = 0.18 | ~250 ms |
| `min_consecutive = 3` | ≤ 150 ms |
| `min_dwell_s = 0.2` | 200 ms |
| **Total** | **roughly 0.8–1.0 s** |

This is the fundamental trade of the current design: it is *stable* rather than
*fast*. Reducing perceived lag means reducing these constants, and the resulting
jitter is the price. `src/compare_smoothing.py` exists to evaluate that trade on
recorded data rather than by feel.

### 10.8 The loop that drives it

`go.py` is a thin Arduino bridge over this pipeline. It calls
`build_predictor()` so it inherits `LIVE_INFERENCE_DEFAULTS` rather than
restating them, calls `predictor.predict()` once per hop, applies an additional
per-finger cooldown mirroring the Arduino's own servo lockout, and writes the
bits over USB serial. Before starting, both `go.py` and `live_binary.main()`
refuse to run unless a warm-up delivers at least 250 samples in one second and
those samples are not all zeros — a fast, explicit failure for a disconnected
board or non-contacting electrodes, instead of a model confidently predicting
from noise.

---

## Part 11 — What this architecture deliberately does not do

- **No temporal model inside the network.** The CNN sees one isolated 200 ms
  window. All memory lives in the post-processing pipeline. An LSTM or a
  Transformer would carry state internally, but would need more data than the
  project has and would complicate the real-time loop.
- **No cross-session adaptation at run time.** Electrode drift and muscle fatigue
  during a session are not compensated. `--freeze-backbone` calibration is an
  offline answer to the same problem.
- **No spatial filtering of EMG.** The first convolution mixes the 8 channels
  with a single learned combination per kernel position. Architectures such as
  EEGNet use a dedicated spatial-filter stage first.
- **No structured output.** The five bits are predicted independently; snap is a
  post-hoc constraint, not a learned joint distribution.
- **No continuous joint angles.** Five binary bits only. Continuous-angle work
  lives in `src/neuropose_angles.py` and `src/eval_angles*.py` and is not part of
  this pipeline.

---

## Part 12 — Reading failures

| Symptom | Most likely cause | Where to look |
|---|---|---|
| `pos_frac` near 0 or 1 for a finger | That finger is never varied in the recordings | Re-record with patterns exercising it |
| Validation mean accuracy stuck below 0.7 | Data quality, not model capacity | Electrode contact, label honesty during HOLD |
| Validation loss rises while training loss falls | Overfitting | Early stopping should catch it; more data or higher dropout |
| Good validation accuracy, poor live behavior | Distribution shift between session and live | Re-don the band and recalibrate; check normalization statistics |
| Live output jitters | Smoothing too weak | Raise `median_k` / lower `ema_alpha`, or train with `--consistency-lambda` |
| Live output feels sluggish | Smoothing too strong | Lower `median_k`, raise `ema_alpha`, cut `min_dwell_s` — see 10.7 |
| One channel's std near 0 at warm-up | That electrode is not making contact | Reposition the band before recording |

---

## Part 13 — Known inconsistencies

### 13.1 Live: the snap vocabulary contains an untrained pattern

The only item here with a run-time effect.
`record_bits.DEFAULT_PATTERNS` has been trimmed to **7** patterns
(`00000 11111 10000 01000 11000 01100 11100`); the original 15-pattern list
survives only as a comment block. But `get_supported_patterns(remap=True)`
still derives its vocabulary from `dataset_bits.RECORDED_PATTERNS`, which is
still the full 15, and remapping those yields **8** patterns.

The extra one is `00100` (middle finger alone). With `--snap` enabled, the
predictor can therefore snap its output to a pattern the current recorder never
captures and the model has no training data for. Either trim
`RECORDED_PATTERNS` to match the recorder, or restore `00100` to the recording
protocol.

### 13.2 Training: `pos_weight` sees the validation set

`pos_frac` is computed over all segments *before* `split_segments` runs, so
validation label statistics influence a training hyperparameter. The effect is
negligible — it is one scalar per finger — but it is a small leak.

### 13.3 Documentation

`docs/ai-architecture.md` was corrected against the source alongside this
document: pattern counts (15 → 8, not 14 → 7), the EMA running in logit space,
exact parameter counts, the `min_dwell_s` mechanism, the live filtering path,
and the file table (`hand_control_emg_arduino.py` and `eval_imu_per_pattern.py`
do not exist in this checkout).

`docs/training.md` is still stale in its "Run live" section: it lists
`ON_THRESH 0.6 / OFF_THRESH 0.4 / EMA_ALPHA 0.35 / MEDIAN_K 5` and points at
`hand_control_emg_arduino.py`. The canonical values now live in
`LIVE_INFERENCE_DEFAULTS` (0.7 / 0.3 / 0.18 / 9) and the entry point is
`go.py`.

---

## Part 14 — Glossary

| Term | Meaning |
|---|---|
| **Backpropagation** | Computing gradients by applying the chain rule backwards through the network |
| **Batch** | A group of examples processed together in one forward/backward pass |
| **Batch normalization** | Per-channel standardization inside the network; behaves differently in train vs eval mode |
| **BCE** | Binary cross-entropy, the loss for yes/no predictions |
| **Convolution** | Sliding a small learned kernel along an axis and computing weighted sums |
| **Dropout** | Randomly zeroing activations during training to prevent over-reliance on any one feature |
| **EMA** | Exponential moving average; blends new values into a running estimate |
| **Epoch** | One full pass over the training set |
| **Gradient** | The direction and magnitude in which changing a parameter increases the loss |
| **Hysteresis** | Using different on and off thresholds to create a dead zone |
| **Inference** | Running a trained model on new data |
| **Leakage** | Information from validation or test data influencing training, inflating scores |
| **Logit** | Unbounded pre-sigmoid model output; log-odds of a probability |
| **Loss** | Scalar measure of how wrong the model currently is |
| **Multi-label** | Several independent yes/no outputs, as opposed to one choice among many |
| **Overfitting** | Memorizing training data at the expense of generalization |
| **Receptive field** | How many input samples influence a single value at some depth |
| **Regularizer** | Anything that makes training harder to make generalization better |
| **Segment** | One contiguous, single-label stretch of recording |
| **Sigmoid** | `1/(1+e^-z)`, mapping a logit to a probability |
| **Tensor** | A multi-dimensional array |
| **Weight decay** | Pulling parameters toward zero each step to discourage large weights |
| **Window** | The fixed-length slice of signal the model consumes (100 samples = 200 ms) |
| **Z-scoring** | Subtracting the mean and dividing by the standard deviation |
