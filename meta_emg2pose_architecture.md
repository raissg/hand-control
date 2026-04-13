# Meta emg2pose — Model Architecture

Source: `~/emg2pose/` repo (Facebook Research, NeurIPS 2024 Datasets & Benchmarks)
Paper: https://arxiv.org/abs/2412.02725

---

## The Task

Input: 16-channel surface EMG at 2 kHz from a wrist-worn band
Output: 20 joint angles (in radians) for one hand, updated in real-time

20 joints = 4 per finger (thumb has CMC_FE, CMC_AA, MCP_FE, IP_FE; each other finger has MCP_AA, MCP_FE, PIP_FE, DIP_FE).

---

## Dataset

- 193 participants, 370 hours, 25,253 sessions
- Each session is ~1 minute of time-aligned EMG + motion-capture joint angles
- 29 different movement "stages" (different prompted hand movements)
- Ground truth from optical motion capture (not camera — actual mocap system)

---

## Main Model: vEMG2Pose (TDS + LSTM)

This is Meta's best model. ~6M parameters. It predicts velocity (change in joint angles) at each time step, conditioned on the previous state.

### Stage 1: TDS Feature Extractor

TDS = Time-Depth Separable convolutions, from a speech recognition paper (Hannun et al., 2019).

```
Input: (batch, 16 channels, T samples at 2kHz)

Conv1dBlock(16 → 256, kernel=11, stride=5)     # subsample 5x, 2000→400 Hz effective
Conv1dBlock(256 → 256, kernel=5, stride=2)      # subsample 2x, 400→200 Hz effective

TDS Stage 1:
  Conv1dBlock(256 → 256, kernel=17, stride=4)   # subsample 4x, 200→50 Hz
  2x TDS blocks (2D conv kernel=9 + FC + residual + LayerNorm)

TDS Stage 2:
  Conv1dBlock(256 → 256, kernel=9, stride=2)    # subsample 2x, 50→25 Hz
  2x TDS blocks (2D conv kernel=5 + FC + residual + LayerNorm)
  Linear(256 → 64)                               # project to 64-dim features

Output: (batch, 64, T') where T' ≈ T/80
```

Total temporal downsampling: 5 × 2 × 4 × 2 = 80x. So 2000 Hz input → ~25 Hz features.
Left context: 1790 samples = 895ms of EMG needed before first prediction.

Each TDS block does:
1. Reshape (B, C, T) → (B, channels, width, T) as a 2D tensor
2. Conv2D across (width, time) — captures cross-channel spatial patterns
3. ReLU + residual connection
4. LayerNorm
5. FC block: Linear → ReLU → Linear + residual + LayerNorm

### Stage 2: LSTM Decoder (sequential, stateful)

The TDS features are resampled to 50 Hz (the "rollout frequency"), then processed one timestep at a time through an LSTM:

```
For each timestep t:
  input = concat(TDS_features[t], previous_predicted_pose)  # 64 + 20 = 84 dims
  LSTM(84 → 512, 2 layers)
  Linear(512 → 20) × 0.01 scale factor

  predicted_velocity = LSTM output
  predicted_pose[t] = predicted_pose[t-1] + predicted_velocity
```

Key design decisions:
- **State conditioning**: the model receives its own previous prediction as input — it knows where the hand currently is
- **Velocity prediction**: outputs delta angles, not absolute angles — accumulated via cumsum
- **Initial position**: provided from ground truth at the start of each sequence (the model needs to know where to start)
- **Scale factor 0.01**: the LSTM output is multiplied by 0.01 before being treated as velocity — keeps predictions smooth and prevents large jumps

### Why Velocity + State?

This is the answer to "does it need to know the starting angle":
- **Yes**, vEMG2Pose takes the initial pose as input
- It predicts **how much to move** (velocity), not **where to be** (absolute position)
- The previous predicted pose is fed back as state conditioning
- This gives temporal coherence — predictions are smooth because each frame builds on the last

---

## Alternative Model: NeuroPose (Encoder-Decoder-ResNet)

Also implemented in the repo. No state conditioning, predicts absolute positions.

```
Input: (batch, 1, 16 channels, T samples) — treated as 2D image (channels × time)

Encoder:
  EncoderBlock(1 → 32, conv [3,2], maxpool [10,2])      # downsample time 10x, channels 2x
  EncoderBlock(32 → 128, conv [3,2], maxpool [8,2])      # downsample time 8x, channels 2x
  EncoderBlock(128 → 256, conv [3,2], maxpool [4,4])     # downsample time 4x, channels 4x

Residual (bottleneck):
  5 × ResidualBlock(256, conv [3,2], 3 convs each, dropout 0.05)

Decoder:
  DecoderBlock(256 → 128, conv [3,2], upsample [10,4])   # upsample time 10x, spatial 4x
  DecoderBlock(128 → 32, conv [3,2], upsample [8,4])     # upsample time 8x, spatial 4x
  DecoderBlock(32 → 1, conv [3,2], upsample [4,2])       # upsample time 4x, spatial 2x

Linear(32 → 20)  # final projection to 20 joint angles

Output: (batch, 20, T) — 20 absolute joint angles per timestep
```

Key differences from vEMG2Pose:
- No LSTM, no state conditioning, no velocity integration
- Predicts absolute positions directly
- No initial pose needed (`provide_initial_pos: False`)
- U-Net style encoder-decoder with skip-less residual blocks
- Window length: 4000 samples = 2 seconds

---

## Data Preprocessing

Minimal:
- Raw EMG → directly to model (no handcrafted features, no explicit filtering)
- RotationAugmentation: randomly shifts EMG channels by [-1, 0, +1] positions (simulates armband rotation)
- ChannelDownsampling: optionally takes every Nth channel
- Windowed: training windows of 11,790 samples (5.9 sec) for vEMG2Pose, 4,000 samples (2 sec) for NeuroPose

No RMS, MAV, WL, ZC, no bandpass filter, no notch filter. The network learns its own features from raw signal.

---

## Training

- PyTorch Lightning
- AdamW optimizer, lr=0.001
- Gradient clipping at 1.0 (vEMG2Pose) / 0.0 (NeuroPose)
- Loss: MAE on joint angles (with IK failure masking)
- Full dataset: 158 users for training, held-out users for testing
- Pre-trained checkpoints available: `~/emg2pose_model_checkpoints/`

---

## Results from Your Experiments (EXPERIMENTS.md)

| Experiment | test MAE (rad) | Fingertip error |
|---|---|---|
| General model → unseen user | 0.251 (~14°) | 29.1 mm |
| Fine-tuned → unseen user | 0.252 | 29.9 mm |
| From scratch → unseen user | 0.504 (~29°) | 71.7 mm |
| General model → seen user | 0.181 (~10°) | — |
| Fine-tuned 8ch/500Hz → unseen user | 0.287 (~16°) | — |

Key findings:
1. Pre-training is critical — from-scratch is 2x worse
2. Unseen users are 91% worse than seen users — EMG is highly individual
3. Degrading to 8ch/500Hz (your MindRove specs) costs ~47% accuracy
4. Fine-tuning doesn't help much without diverse movement data

---

## Architecture Comparison

| | Meta vEMG2Pose | Meta NeuroPose | NaviFlame |
|---|---|---|---|
| Task | 20 joint angle regression | 20 joint angle regression | 7-class gesture classification |
| Input | 16ch × 2kHz raw EMG | 16ch × 2kHz raw EMG | 8ch × 500Hz filtered EMG |
| Features | Learned (TDS convolutions) | Learned (2D convolutions) | Learned (SCNet + SFKNet) |
| Temporal model | LSTM (stateful) | None | None |
| Output | 20 velocities → cumsum | 20 absolute angles | 7 class probabilities |
| State conditioning | Yes (previous pose fed back) | No | No |
| Parameters | ~6M | unknown | unknown |
| Training data | 370 hours, 158 users | Same | Unknown (proprietary) |
| Your hardware compatible? | Partially (47% degradation at 8ch/500Hz) | Partially | Designed for MindRove 8ch/500Hz |
