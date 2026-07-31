# Temporal Consistency Loss

The poster's "Temporal Consistency" block, fully unpacked: what the formula
means, why each piece is there, where it lives in the code, and what it
buys you in practice.

---

## The formula

```
L = ½ [ BCE(f(x_t), y) + BCE(f(x_{t+Δ}), y) ]  +  λ · MSE(p_t, p_{t+Δ})
```

Two terms added together: a classification term and a consistency term.

### Symbols

| Symbol | Meaning |
|---|---|
| `x_t` | One 200 ms EMG (+ IMU) window starting at sample `t`. |
| `x_{t+Δ}` | A second window from the **same gesture-hold**, taken `Δ` samples later. At 500 Hz, `Δ` is sampled randomly in 10–100 ms. |
| `y` | The 5-bit label of that hold. **Same** for both views — the hand didn't move between them. |
| `f(x)` | The model's raw 5 logits for window `x`. |
| `p_t = σ(f(x_t))` | Sigmoid probabilities for window `t`. |
| `p_{t+Δ} = σ(f(x_{t+Δ}))` | Sigmoid probabilities for the time-shifted window. |
| `λ` | Scalar weight (0.05–0.2 in practice) controlling how much consistency matters relative to classification. |

### What each term does

- **Classification term** — `½ [ BCE(f(x_t), y) + BCE(f(x_{t+Δ}), y) ]`.
  Standard per-finger Binary Cross-Entropy applied to *both* views,
  averaged. "Predict the right finger bits, on both windows."

- **Consistency term** — `λ · MSE(p_t, p_{t+Δ})`. Squared difference
  between the two probability vectors. Zero when the model gives
  identical outputs for the two views, grows as they diverge.

---

## Why this loss exists

Without the consistency term, the network is graded on each window in
isolation. Two adjacent 200 ms windows from the same gesture-hold are
*nearly identical inputs*, but nothing in plain BCE forces the model to
produce nearly identical outputs.

At inference this manifests as **flickering**: shift the window by 50 ms
(one frame at 20 Hz) and the prediction can swing from `[0.7, 0.3, …]` to
`[0.3, 0.7, …]`. The downstream median + EMA smoothing has to clean it up,
and the cost is latency.

Adding the MSE-on-probs term explicitly punishes that. The network learns
features that are **invariant to small time shifts within the same
gesture**. Put differently: small time shifts are treated as the same
control intent.

---

## Why each design choice

| Choice | Why |
|---|---|
| Two views (not one) | You need two examples of the same input class with a small perturbation to define "consistent." Time-shifting within a held pose is a natural perturbation that's free — you already recorded it. |
| MSE on **probabilities**, not logits | Probabilities are bounded in `[0, 1]` so the MSE term has a stable scale. MSE on raw logits would explode when the model is very confident, dominating the BCE term. |
| `Δ ∈ [10, 100] ms` | Smaller → near-identical samples, no real test of robustness. Larger → can cross into a gesture transition where labels actually *should* differ. |
| Averaged BCE on both views | Ensures the model is graded equally on both windows. Prevents it from "ignoring" the time-shifted view to satisfy the consistency term cheaply. |
| Small `λ` | Too large and the model trivially satisfies consistency by predicting a constant — perfect consistency, terrible accuracy. `λ ≈ 0.1` keeps consistency as a soft prior that nudges, doesn't dominate. |

---

## Where it lives in code

`src/train_bits.py` — flags:

```python
ap.add_argument("--consistency-lambda", type=float, default=0.0,
                help="Weight on temporal-consistency MSE. 0 = disabled.")
ap.add_argument("--consistency-deltas", default="5,50",
                help="Range 'min,max' of sample offsets for the pair "
                     "window. At 500Hz, '5,50' = 10-100ms.")
```

The training loop, with the formula transliterated to PyTorch:

```python
# x  : window at time t       — shape (B, 8, 100)
# x2 : window at time t + Δ   — shape (B, 8, 100)
# y  : the shared 5-bit label — shape (B, 5)

logits = model(torch.cat([x, x2], dim=0))     # one fused forward pass
l1, l2 = logits[:x.size(0)], logits[x.size(0):]

bce  = 0.5 * (loss_fn(l1, y) + loss_fn(l2, y))                 # term 1
cons = ((torch.sigmoid(l1) - torch.sigmoid(l2)) ** 2).mean()   # term 2
loss = bce + lam * cons                                        # total
```

`lam` is `λ`, `cons` is the MSE term. The `BitsWindowDataset` produces
the `(x, x2, y)` triplets when constructed with a `pair_delta_range`.

---

## What it buys you

Two claims you can defend:

1. **Less downstream smoothing needed.** Because the model itself produces
   stable probabilities frame-to-frame, you can use a smaller median
   window or higher EMA `α` and still avoid jitter — i.e. **lower
   latency for the same visual stability**.
2. **Same accuracy at the operating point.** The consistency term doesn't
   make the model less correct; it makes it *self-consistent*. Per-frame
   accuracy stays the same as long as `λ` is reasonable.

The architectural punchline: this loss term shifts work from
**inference** (where latency is felt) into **training** (where it's free).

| Stabilizer | When it runs | What it costs |
|---|---|---|
| Temporal-consistency loss | training time | nothing at inference |
| Median / EMA / hysteresis / dwell | every inference frame | latency |
