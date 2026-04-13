# EMG-to-Hand-Pose Papers — Detailed Comparison

## Paper-by-Paper Results

### 1. Geng2022 — CNN-Attention (IEEE RA-L)
- **Task**: Continuous estimation of 10 finger joint angles (MCP joints) from 6 grasp movements
- **Hardware**: 12ch Trigno Wireless EMG (Delsys) at 2 kHz + CyberGlove II for ground truth
- **Dataset**: NinaPro DB2, 40 subjects (but only 15 used in their experiment)
- **Architecture**: Multi-scale CNN (kernels 3/5/7) + Multi-head Attention + Linear
- **Results**: CC=0.87, RMSE=9.65°, R²=0.73
- **Baselines beaten**: LSTM (CC=0.79, R²=0.62), SPGP (CC=0.75, R²=0.54)
- **Training time**: 43 min (vs 73 min for LSTM)

### 2. Liu2021 — NeuroPose (ACM WWW)
- **Task**: 21 DoF continuous 3D finger joint angle tracking (free-form motion)
- **Hardware**: 8ch Myo armband at 200 Hz + Leap Motion depth camera for ground truth
- **Dataset**: 12 users, 900s training data per user
- **Architecture**: Encoder-Decoder-ResNet (Conv-BN-ReLU-MaxPool encoder, ResNet bottleneck, Upsampling decoder)
- **Results**: Median error=6.24°, 90th-percentile error=18.33°
- **Transfer learning**: 90s of new-user data matches 13.5 min of from-scratch training
- **Latency**: 0.101s on smartphone (Sony Xperia Z3)
- **Robustness**: Consistent across sensor repositioning, wrist positions, and across days

### 3. Putro2024 — Transformer (Biomed Signal Process Control)
- **Task**: Estimate 22 finger joint angles (flexion-extension)
- **Hardware**: 16ch (2x Myo armbands) at 200 Hz
- **Dataset**: NinaPro DB5 (10 subjects)
- **Architecture**: Feature extraction (MDF) + 4-block Transformer (multi-head attention, 256 head size, 4 heads) + MLP regression
- **Results**: R²=0.970, RMSE=21.10°
- **Best feature**: Median Frequency (MDF), window=100 samples (500ms)
- **Beats**: LSTM+Attention (R²=0.957), GRU (R²=0.921)

### 4. Jiang2025 — TF2AngleNet (Biomed Signal Process Control)
- **Task**: Estimate 6 finger joint angles (practical synergy-based movements)
- **Hardware**: 5ch EMG at 1600 Hz + Leap Motion for non-contact ground truth
- **Dataset**: 5 subjects, 5 days, 60 trials per subject
- **Architecture**: Dual-stream CNN (1D raw EMG + 2D time-frequency spectrogram) → grouped convolution → 1D conv decoder
- **Results (within-session)**: CC=0.947, R²=0.892, NRMSE=9.5%
- **Cross-day**: CC=0.881, R²=0.770, NRMSE=13.0%
- **Cross-subject**: CC=0.602, R²=0.246, NRMSE=19.2%
- **Inference**: 2040 μs per sample (490 samples/sec)
- **Parameters**: 104,942 (very lightweight)

### 5. Salter2024 — emg2pose / vEMG2Pose (NeurIPS)
- **Task**: Estimate 20 joint angles (radians), continuous full-hand pose
- **Hardware**: 16ch Meta wristband at 2 kHz + 26-camera motion capture
- **Dataset**: 193 users, 370 hours, 25,253 sessions
- **Architecture**: TDS conv encoder (80x downsampling) → LSTM decoder (stateful, velocity prediction with state conditioning)
- **Results (general model → unseen user)**: MAE=0.251 rad (~14°), fingertip=29.1mm
- **Results (general → seen user)**: MAE=0.181 rad (~10°)
- **Results (8ch/500Hz degraded)**: MAE=0.287 rad (~16°), +47% worse
- **From scratch**: MAE=0.504 rad (~29°) — 2x worse without pre-training
- **Scale**: ~6M parameters, trained on 158 users

### 6. Kaifosh2025 — Meta Neuromotor Interface (Nature)
- **Task**: 3 separate HCI tasks (NOT hand pose regression)
  - 1D wrist continuous control
  - Discrete gesture detection (7 gestures)
  - Handwriting transcription
- **Hardware**: 16ch (48 electrode pins) Meta wristband at 2 kHz
- **Dataset**: Thousands of participants (proprietary)
- **Architecture**: MPF+LSTM (wrist), Conv1D+LSTM (gestures), MPF+Conformer (handwriting)
- **Results**:
  - Wrist: 0.66 target acquisitions/sec, <13°/s velocity error
  - Gestures: 0.88 detections/sec, ~93% per-gesture accuracy
  - Handwriting: 20.9 WPM, ~6% character error rate
- **Key result**: Performance scales as power law with training data and model size

---

## Ranking by Raw Results (within-subject / best-case accuracy)

Papers ranked purely by how accurately they predict joint angles in their best evaluation scenario.
Kaifosh2025 excluded (different task — classification/HCI, not regression).

| Rank | Paper | Best-Case Metric | Joints | Notes |
|---|---|---|---|---|
| 1 | **Putro2024 (Transformer)** | R²=0.970 | 22 | Within-subject, NinaPro DB5. Highest R² reported. But RMSE=21° is high — R² is inflated by large range of motion in the dataset. |
| 2 | **Jiang2025 (TF2AngleNet)** | CC=0.947, R²=0.892 | 6 | Within-session only. Drops hard cross-subject (R²=0.246). Fewest angles predicted. |
| 3 | **Geng2022 (CNN-Attention)** | CC=0.87, R²=0.73, RMSE=9.65° | 10 | Lowest RMSE in absolute degrees. But only 10 angles, within-subject. |
| 4 | **Liu2021 (NeuroPose)** | Median error=6.24° | 21 | Free-form motion (hardest task). With transfer learning, only 90s of user data needed. Robust across days. |
| 5 | **Salter2024 (emg2pose)** | MAE=0.181 rad (~10°) seen user | 20 | On seen users. Unseen user = ~14°. But this is the ONLY paper tested on 193 users with true cross-user generalization. |

### Why raw ranking is misleading

The papers are **not directly comparable** because they differ in:

| Factor | Easy (inflates score) | Hard (deflates score) |
|---|---|---|
| Evaluation | Within-subject | Cross-user |
| Motion type | Predefined gestures | Free-form arbitrary |
| Joints predicted | 6-10 | 20-22 |
| Dataset size | 5-15 subjects | 193 subjects |
| Ground truth | Data glove | Motion capture |

Putro2024 has the highest R² but was never tested cross-user. Salter2024 has "lower" numbers but was tested on 193 unseen users doing arbitrary motions — a fundamentally harder problem.

---

## Ranking by Relevance to This Project

What matters for building a real-time EMG-to-3D-hand system with MindRove 8ch/500Hz:

| Rank | Paper | Why |
|---|---|---|
| **1** | **Salter2024 (emg2pose)** | Largest dataset (370 hrs, 193 users), best cross-user generalization, state-conditioned velocity prediction, open-source code + data + checkpoints. Gold standard. Already tested at 8ch/500Hz (47% degradation but still works). |
| **2** | **Liu2021 (NeuroPose)** | Most similar to your hardware (8ch Myo, low sample rate). Free-form 21 DoF tracking. Transfer learning with 90s of data. Runs on a phone. Meta re-implemented it as a baseline in emg2pose repo. |
| **3** | **Jiang2025 (TF2AngleNet)** | Uses Leap Motion (non-contact) for ground truth — same concept as your MediaPipe/WiLoR plan. Dual time+frequency features. Tiny model (105K params). But only 5 subjects, 6 angles, and cross-subject performance collapses. |
| **4** | **Putro2024 (Transformer)** | Highest R² (0.970) but within-subject only on NinaPro. 22 angles is ambitious. Shows Transformers work for this task. No real-time or cross-user evaluation. |
| **5** | **Geng2022 (CNN-Attention)** | Multi-scale CNN + attention is a clean architecture. Low RMSE (9.65°). But only 10 angles, no cross-user, and uses expensive CyberGlove for ground truth. |
| **6** | **Kaifosh2025 (Nature)** | Not joint-angle regression. But proves that generic cross-user EMG decoding works at scale, and performance follows power-law scaling with data. Motivates collecting more data. |

---

## Key Takeaway

**emg2pose (Salter2024)** is the gold standard — open-source, massive dataset, proven architecture, and you already have the code, data, and checkpoints locally. **NeuroPose (Liu2021)** is the most relevant for your 8ch hardware and is already implemented as a baseline in the emg2pose repo. Your path: adapt one of these architectures for MindRove 8ch/500Hz, using camera-based hand tracking (MediaPipe/WiLoR) as ground truth instead of mocap.
