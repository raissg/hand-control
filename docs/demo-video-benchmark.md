# Demo Video Benchmark

This is the repeatable video protocol implemented by `demo_benchmark.py`.

The goal is a polished demonstration in which the operator knows the next pose
slightly before the robot is allowed to move, while preserving the fact that
the **live EMG model** — not a timer — decides whether the robot receives that
pose.

## What the ideal video looks like

Use one fixed, uninterrupted camera shot. Frame all three at once:

1. the operator's forearm and hand;
2. the full robot hand;
3. enough of the laptop display to read the phase banner and timing.

The sequence on screen is:

```text
BASELINE  ->  CUE  ->  LIVE  ->  MATCH  ->  BASELINE  -> ...
```

- **BASELINE:** the operator opens and relaxes the hand. The next target remains
  hidden so cue onset is unambiguous.
- **CUE:** the target appears in large text with one card per finger. The
  operator moves immediately. The robot command is visibly gated for the
  configured lead time.
- **LIVE:** the gate is open. If the model has not matched yet, the screen says
  it is waiting. The script does not move the robot early or substitute the
  scheduled target.
- **MATCH:** the current model output equals the target, so that exact pose is
  sent to the Arduino and latched for the shot. Detection time, command time,
  and hold stability stay visible.
- **MISSED:** if the model never matches, the robot does not move and the miss
  remains in the final CSV. There is no hidden success path.

At the end, hold the camera on the summary frame for several seconds.

## Why this looks immediate

The normal smoothing path intentionally adds roughly 0.8–1.0 seconds of
stability delay. The default demo cue lead is 0.9 seconds:

```text
target appears -> operator starts moving -> model smoothing settles
                                             |
                              robot gate opens
                                             |
                       current output matches target -> command sent
```

If detection finishes during the cue lead, the robot command is sent as soon as
the visible gate opens. That creates a clean, near-immediate reveal without
pre-programming the robot pose.

This is **pre-cued choreography**, not a reaction-latency measurement. Every
frame carries the cue-lead disclosure. Do not crop that label or describe the
result as zero-latency control.

## Establish the lead from data

Do not guess the final cue lead. First run a zero-lead benchmark with multiple
repeats:

```bash
python demo_benchmark.py \
  --model "/absolute/path/to/checkpoint.pt" \
  --cue-lead-s 0 \
  --repeats 3 \
  --record-screen
```

Read `recognition_ms` in the generated CSV. It measures from visual cue onset
until the published model bits first equal the target. It includes:

- the person's reaction time;
- the time needed to form the pose;
- model window and smoothing delay.

It is **not** the neural network's forward-pass time.

Set the polished video's `--cue-lead-s` near the 90th percentile of those
measurements. For example, if 90% of successful trials are recognized within
870 ms, use `--cue-lead-s 0.9`. That makes most robot reveals immediate while
keeping the number grounded in an observed baseline.

## Record the polished take

Once the lead is selected:

```bash
python demo_benchmark.py \
  --model "/absolute/path/to/checkpoint.pt" \
  --cue-lead-s 0.9 \
  --record-screen
```

The clone currently has no `models/` directory, so a trained checkpoint must be
copied locally and supplied with `--model`.

Useful options:

```text
--no-flash                  use firmware already on the Arduino
--patterns 10000,01000,...  choose and order the target poses
--repeats 2                 repeat the same deterministic sequence
--windowed                  do not take over the full display
--no-invert-bits            stop applying go.py's model-to-Arduino complement
--cue-lead-s 0              remove choreography for latency measurement
```

The default measured sequence is the six non-baseline poses in the current
`record_bits.DEFAULT_PATTERNS`. `00000` is reserved as the reset between every
trial.

## What the benchmark records

Each run writes `benchmark_results/demo_benchmark_<timestamp>.csv`, with one row
per target:

- outcome (`matched`, `missed`, or `aborted`);
- cue lead;
- cue-to-first-detection time;
- cue-to-Arduino-command time;
- model target score when commanded;
- fraction of hold frames that remained on target;
- model and smoothing settings;
- model bits and actual Arduino command bits.

`--record-screen` also writes the synchronized UI to MP4. Use the external camera
as the primary evidence because it captures physical servo movement; use the
screen recording to inspect exact cues and timings.

## Recommended claims

Accurate:

> In this pre-cued, model-driven demo, the operator receives a 0.9-second visual
> lead. The robot is commanded only when the live smoothed EMG output matches
> the shown pose. The unassisted zero-lead results are reported separately.

Not accurate:

> The neural network reacts in 0.9 seconds.

The network forward pass is under a millisecond. The observed response includes
human movement, a 200 ms signal window, and deliberate temporal smoothing.

## Video credibility checklist

- Keep the entire run in one take.
- Keep both human and robot hands visible continuously.
- Keep the phase banner and cue-lead disclosure readable.
- Show at least one one-finger, one two-finger, one three-finger, and the
  all-finger pose.
- Do not delete misses from the CSV.
- Publish the zero-lead CSV beside the polished pre-cued run.
- Do not call serial dispatch time physical servo-completion time; there is no
  position sensor measuring completion.
