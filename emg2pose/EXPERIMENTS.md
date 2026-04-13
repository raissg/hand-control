# emg2pose Experiment Log

## Background

**emg2pose** predicts 20 hand joint angles from surface electromyography (sEMG) signals recorded from the forearm. The model is a TDS (Time-Depth Separable) convolutional network with ~6M parameters. Input: 16 EMG channels at 2 kHz. Output: 20 joint angles in radians.

The pre-trained general model was trained on the full dataset: **370 hours** of data from **158 users** across **29 movement stages**.

All experiments below were conducted on a MacBook Pro (Apple Silicon, MPS GPU).

---

## Data

| Dataset | Location | Contents |
|---|---|---|
| Mini dataset | `~/emg2pose_dataset_mini/` | 30 sessions from user `d387095792` (held-out — never in training) |
| Training user extract | `~/emg2pose_train_user/` | 7 sessions from user `03d49b393e` (was in training set) |
| Pre-trained checkpoints | `~/emg2pose_model_checkpoints/` | 5 checkpoints from Meta (regression, tracking, neuropose) |

### User d387095792 — Data Split

All 30 sessions from the same person, same day, same device. Split defined in `config/data_split/user_d387095792.yaml`.

| Split | Recordings | Sessions | Purpose |
|---|---|---|---|
| Train | 1, 4–13 (left+right) | 22 | Model learns from these |
| Val | 14, 15 (left+right) | 4 | Checked after each epoch for early stopping |
| Test | 2, 3 (left+right) | 4 | Evaluated once at the end, never seen during training |

---

## Experiment 1: General Model on Unseen User

**Question:** How well does the pre-trained general model perform on a user it has never seen?

**Setup:** Load `regression_vemg2pose.ckpt` (trained on 158 users). Run inference on user `d387095792`'s test sessions (recordings 2 & 3). This user was fully held out from training.

| Metric | Value |
|---|---|
| test_mae | 0.251 rad |
| test_fingertip_distance | 29.1 mm |
| test_landmark_distance | 17.3 mm |

**Takeaway:** The model produces reasonable predictions on a completely unseen user, but with noticeable errors — roughly 29mm fingertip error and 0.25 rad (~14°) average joint angle error.

---

## Experiment 2: Fine-tuning for Specific User

**Question:** Can we improve performance on user `d387095792` by fine-tuning the general model on their data?

**Setup:** Load pre-trained checkpoint, continue training on 22 sessions from this user only, evaluate on 4 test sessions.

| Model | Epochs | test_mae | test_fingertip (mm) | test_landmark (mm) |
|---|---|---|---|---|
| General (no fine-tuning) | 0 | 0.251 | 29.1 | 17.3 |
| Fine-tuned | 5 | 0.252 | 29.1 | 17.3 |
| Fine-tuned | 20 (best at epoch 9) | 0.252 | 29.9 | 17.9 |

**Takeaway:** Fine-tuning did **not** improve test performance, despite val_mae improving significantly (0.299 → 0.219). The val sessions (recordings 14, 15) and test sessions (recordings 2, 3) contain different movement stages — the model overfits to val-stage patterns that don't transfer to test stages. With only ~22 minutes of data lacking stage diversity, fine-tuning has limited benefit.

---

## Experiment 3: Training from Scratch (Single User)

**Question:** What if we train a fresh model from random weights using only this user's data?

**Setup:** Same 22 train / 4 val / 4 test split for user `d387095792`. No pre-trained weights — random initialization. Trained for 50 epochs (best at epoch 23, early stopped at 50).

| Model | test_mae | test_fingertip (mm) |
|---|---|---|
| General (pre-trained on 158 users) | 0.251 | 29.1 |
| Fine-tuned (20 epochs) | 0.252 | 29.9 |
| **From scratch** | **0.504** | **71.7** |

**Takeaway:** The from-scratch model is **2x worse** than the general model. Fingertip error jumped from 29mm to 72mm. With only ~22 minutes of data and 6M parameters, the model cannot learn the EMG→pose mapping from zero. Pre-training on the large multi-user dataset is essential.

---

## Experiment 4: General Model on Seen vs Unseen User

**Question:** How much better does the general model perform on a user whose data was included in training?

**Setup:** Run inference with the same general model on:
- **Seen user** (`03d49b393e`): test sessions that were held out from training, but the user's other sessions were in the training set
- **Unseen user** (`d387095792`): entirely held out from training

| User | In training set? | test_mae | Degradation |
|---|---|---|---|
| `03d49b393e` (seen) | Yes (other sessions) | **0.181** | — |
| `d387095792` (unseen) | No | **0.345** | +91% worse |

**Takeaway:** The general model performs dramatically better on users it has seen during training (0.181 vs 0.345 MAE — 91% degradation for unseen users). Every person's muscles produce different EMG patterns for the same hand movements. When the model has seen your specific EMG-to-pose mapping, it performs far better.

---

## Experiment 5: Degraded Input Quality (8ch / 500 Hz)

**Question:** What happens when we halve the EMG channels (16→8) and reduce sampling rate (2kHz→500Hz)?

**Setup:** Fine-tune the pre-trained model on user `d387095792`'s 22 training sessions. Compare full-quality inputs against degraded inputs. Both conditions: 20 epochs, LR=5e-4, batch_size=32, early stopping patience=10.

**Degradation method:**
- **Channel reduction:** Take every other EMG channel (0, 2, 4, ..., 14 → 8 channels)
- **Temporal reduction:** Take every 4th sample (2000 Hz → 500 Hz), applied to EMG, joint angles, and IK failure mask

**Two approaches for handling 8 channels:**

| Approach | How | Architecture change? |
|---|---|---|
| Adapted conv | Reshape first Conv1d from 16→8 input channels. Initialize from pretrained weights `[:, ::2, :]` | Yes |
| Duplicated channels | Take 8 channels, `repeat_interleave` back to 16. Pretrained model untouched | No |

### Results

| Condition | Val MAE | Test MAE | vs Baseline |
|---|---|---|---|
| Baseline (16ch @ 2kHz) | 0.1672 | 0.1935 | — |
| 8ch adapted conv (8ch @ 500Hz) | 0.2057 | 0.2871 | +48.4% |
| 8ch duplicated (8ch @ 500Hz) | 0.2098 | 0.2824 | +46.0% |

**Takeaway:** Dropping to 8 channels and 500Hz causes a ~47% increase in test MAE. The two adaptation methods (reshaping the conv layer vs duplicating channels) produce nearly identical results — the performance hit comes from the **information loss** itself, not the adaptation strategy. Half the spatial resolution makes it harder to distinguish which muscles are activating, and 4x lower temporal resolution gives a coarser view of muscle dynamics.

---

## Summary Table

| # | Experiment | test_mae | vs General |
|---|---|---|---|
| 1 | General model → unseen user | 0.251 | baseline |
| 2 | Fine-tuned (20 ep) → unseen user | 0.252 | +0.4% |
| 3 | From scratch → unseen user | 0.504 | +101% |
| 4a | General model → seen user | 0.181 | -28% |
| 4b | General model → unseen user (different test) | 0.345 | +37% |
| 5a | Fine-tuned 16ch/2kHz → unseen user | 0.1935 | -23% |
| 5b | Fine-tuned 8ch/500Hz adapted → unseen user | 0.2871 | +14% |
| 5c | Fine-tuned 8ch/500Hz duplicated → unseen user | 0.2824 | +13% |

---

## Key Findings

1. **Pre-training is critical.** Training from scratch on limited user data produces a model 2x worse than the pre-trained general model.

2. **User-specificity matters enormously.** The general model is 91% worse on unseen users vs seen users. EMG patterns are highly individual.

3. **Fine-tuning has limited value with low-diversity data.** When val and test contain different movement stages, fine-tuning improves val but not test. More diverse training data per user would likely help.

4. **Signal degradation (8ch/500Hz) costs ~47%.** Halving channels and quartering sample rate substantially degrades performance. The adaptation method (conv reshape vs channel duplication) makes negligible difference — it's the information loss that matters.

---

## Reproducibility

All experiment code is in the repository:

| File | Purpose |
|---|---|
| `config/data_split/user_d387095792.yaml` | User-specific train/val/test split |
| `scripts/degraded_finetune.py` | Degraded-input experiment (Experiment 5). Run with `--full` for all conditions |
| `scripts/extract_train_user.py` | HTTP range-request extraction of specific files from the 431 GiB remote tar |
| `notebooks/compare_finetuned.ipynb` | Visual comparison of ground truth vs model predictions |
| `notebooks/seen_user_inference.ipynb` | Seen vs unseen user inference visualization |
| `emg2pose/train.py` | Standard training/fine-tuning via Hydra CLI |

**Fine-tuning command (Experiments 1–3):**
```bash
/opt/anaconda3/envs/emg2pose/bin/python -m emg2pose.train \
  train=True eval=True \
  experiment=regression_vemg2pose \
  trainer.max_epochs=20 \
  data_split=user_d387095792 \
  data_location="$HOME/emg2pose_dataset_mini" \
  checkpoint="$HOME/emg2pose_model_checkpoints/regression_vemg2pose.ckpt"
```

**Degraded experiment (Experiment 5):**
```bash
PYTHONUNBUFFERED=1 /opt/anaconda3/envs/emg2pose/bin/python scripts/degraded_finetune.py --full
```
