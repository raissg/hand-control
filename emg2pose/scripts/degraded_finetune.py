"""
Degraded-input fine-tuning experiment for emg2pose.

Compares two conditions on user d387095792 (30 sessions in mini dataset):
  A) Baseline: 16 channels @ 2000 Hz (full quality)
  B) Degraded:  8 channels @  500 Hz (every-other channel, 4x temporal downsample)

Both fine-tune the pre-trained general model on 22 training sessions
with early stopping on validation MAE.

Usage:
    PYTHONUNBUFFERED=1 /opt/anaconda3/envs/emg2pose/bin/python scripts/degraded_finetune.py
"""

import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from emg2pose.data import Emg2PoseSessionData
from emg2pose.lightning import Emg2PoseModule
from emg2pose.utils import get_ik_failures_mask
import emg2pose.constants
import emg2pose.pose_modules
import emg2pose.metrics

# ── Paths ────────────────────────────────────────────────────────────────────
PRETRAINED_CKPT = Path("~/emg2pose_model_checkpoints/regression_vemg2pose.ckpt").expanduser()
DATA_DIR = Path("~/emg2pose_dataset_mini").expanduser()

TRAIN_SESSIONS = [
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-4_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-4_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-5_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-5_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-6_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-6_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-7_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-7_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-8_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-8_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-9_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-9_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-10_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-10_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-11_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-11_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-12_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-12_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-13_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-13_right",
]

VAL_SESSIONS = [
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-14_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-14_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-15_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-15_right",
]

TEST_SESSIONS = [
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-2_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-2_right",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-3_left",
    "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-3_right",
]

WINDOW_LENGTH = 11790  # same as regression_vemg2pose config
MAX_EPOCHS = 20
PATIENCE = 10
BATCH_SIZE = 32
LR = 5e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Dataset with optional degradation ────────────────────────────────────────
class DegradedEmgDataset(Dataset):
    """HDF5 session dataset with optional channel/temporal downsampling."""

    def __init__(
        self,
        hdf5_path: Path,
        window_length: int = 11790,
        stride: int | None = None,
        ch_stride: int = 1,
        t_stride: int = 1,
        jitter: bool = False,
        duplicate_channels: bool = False,
    ):
        self.session = Emg2PoseSessionData(hdf5_path)
        self.wl = window_length
        self.stride = stride or window_length
        self.ch_stride = ch_stride
        self.t_stride = t_stride
        self.jitter = jitter
        self.duplicate_channels = duplicate_channels

        n = len(self.session)
        self.n_windows = max(1, (n - self.wl) // self.stride + 1)
        self._no_ik = self.session.no_ik_failure

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        if self.jitter:
            leftover = len(self.session) - start - self.wl
            if leftover > 0:
                start += np.random.randint(0, min(self.stride, leftover))
        end = start + self.wl

        window = self.session[start:end]

        emg = torch.as_tensor(window["emg"]).float()
        ja = torch.as_tensor(window["joint_angles"]).float()
        no_ik = torch.as_tensor(self._no_ik[start:end])

        emg = emg[:, :: self.ch_stride]
        if self.duplicate_channels and self.ch_stride > 1:
            emg = emg.repeat_interleave(self.ch_stride, dim=1)
        emg = emg[:: self.t_stride]
        ja = ja[:: self.t_stride]
        no_ik = no_ik[:: self.t_stride]

        return {
            "emg": emg.T,
            "joint_angles": ja.T,
            "no_ik_failure": no_ik,
            "window_start_idx": start,
            "window_end_idx": end,
        }


def make_loaders(ch_stride, t_stride, batch_size, duplicate_channels=False):
    def _paths(names):
        return [DATA_DIR / f"{s}.hdf5" for s in names]

    def _dataset(paths, jitter):
        return ConcatDataset(
            [
                DegradedEmgDataset(
                    p,
                    window_length=WINDOW_LENGTH,
                    ch_stride=ch_stride,
                    t_stride=t_stride,
                    jitter=jitter,
                    duplicate_channels=duplicate_channels,
                )
                for p in paths
            ]
        )

    train_ds = _dataset(_paths(TRAIN_SESSIONS), jitter=True)
    val_ds = _dataset(_paths(VAL_SESSIONS), jitter=False)
    test_ds = _dataset(_paths(TEST_SESSIONS), jitter=False)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


# ── Model helpers ────────────────────────────────────────────────────────────
def load_pretrained(n_input_channels: int = 16):
    """Load the pretrained model, adapting the first conv if needed."""
    module = Emg2PoseModule.load_from_checkpoint(
        str(PRETRAINED_CKPT), map_location="cpu"
    )

    if n_input_channels != 16:
        # Conv1dBlock stores layers in self.conv = Sequential(Conv1d, ReLU, Dropout)
        conv_block = module.model.network.layers[0]
        old_conv = conv_block.conv[0]  # first element of Sequential is the Conv1d
        ch_ratio = 16 // n_input_channels
        new_conv = nn.Conv1d(
            n_input_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size[0],
            stride=old_conv.stride[0],
            padding=0,
        )
        new_conv.weight.data = old_conv.weight.data[:, ::ch_ratio, :].clone()
        new_conv.bias.data = old_conv.bias.data.clone()
        conv_block.conv[0] = new_conv
        print(f"  Adapted first conv: 16ch → {n_input_channels}ch")

    return module


def set_sample_rate(rate: int):
    """Patch EMG_SAMPLE_RATE everywhere it's been imported by value."""
    emg2pose.constants.EMG_SAMPLE_RATE = rate
    emg2pose.pose_modules.EMG_SAMPLE_RATE = rate
    emg2pose.metrics.EMG_SAMPLE_RATE = rate


# ── Training loop ────────────────────────────────────────────────────────────
def compute_mae(module, batch):
    """Forward pass → masked MAE."""
    mask = batch["no_ik_failure"].clone()
    if module.provide_initial_pos:
        mask[~mask[:, module.model.left_context]] = False
    batch["no_ik_failure"] = mask

    preds, targets, no_ik = module.model.forward(batch, module.provide_initial_pos)
    mask3d = no_ik.unsqueeze(1).expand_as(preds)
    if mask3d.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    return nn.L1Loss()(preds[mask3d], targets[mask3d])


def to_device(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def run_epoch(module, loader, optimizer, device, train=True):
    module.train(train)
    maes = []
    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for batch in loader:
            batch = to_device(batch, device)
            mae = compute_mae(module, batch)
            if train:
                optimizer.zero_grad()
                mae.backward()
                optimizer.step()
            maes.append(mae.item())
    return float(np.mean(maes))


def train_and_evaluate(name, n_channels, ch_stride, t_stride, sample_rate,
                       duplicate_channels=False):
    real_ch = 16 // ch_stride
    mode = "duplicated→16" if duplicate_channels else f"{real_ch}ch"
    banner = f"  {name}:  {mode} @ {sample_rate} Hz"
    print(f"\n{'=' * 60}")
    print(banner)
    print(f"{'=' * 60}")

    set_sample_rate(sample_rate)

    module = load_pretrained(n_channels)
    module = module.to(DEVICE)

    train_loader, val_loader, test_loader = make_loaders(
        ch_stride, t_stride, BATCH_SIZE, duplicate_channels=duplicate_channels
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None
    wait = 0

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_mae = run_epoch(module, train_loader, optimizer, DEVICE, train=True)
        val_mae = run_epoch(module, val_loader, None, DEVICE, train=False)

        tag = ""
        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(module.state_dict())
            wait = 0
            tag = " ★"
        else:
            wait += 1
            if wait >= PATIENCE:
                tag = "  (early stop)"

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:2d}/{MAX_EPOCHS}  "
            f"train_mae={train_mae:.4f}  val_mae={val_mae:.4f}  "
            f"best={best_val:.4f}  [{elapsed:.0f}s]{tag}"
        )
        if wait >= PATIENCE:
            break

    if best_state is not None:
        module.load_state_dict(best_state)

    test_mae = run_epoch(module, test_loader, None, DEVICE, train=False)
    print(f"\n  ► Test MAE = {test_mae:.4f}  (best val MAE = {best_val:.4f})")

    set_sample_rate(2000)
    return {"test_mae": test_mae, "best_val_mae": best_val, "name": name}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Pretrained checkpoint: {PRETRAINED_CKPT}")
    print(f"Data: {DATA_DIR}")
    print(f"Sessions: {len(TRAIN_SESSIONS)} train / {len(VAL_SESSIONS)} val / {len(TEST_SESSIONS)} test")
    print(f"Window: {WINDOW_LENGTH} samples | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"Max epochs: {MAX_EPOCHS} | Patience: {PATIENCE}")

    # Cached results from previous runs
    baseline = {"test_mae": 0.1935, "best_val_mae": 0.1672, "name": "Baseline (16ch/2kHz)"}
    adapted = {"test_mae": 0.2871, "best_val_mae": 0.2057, "name": "8ch adapted conv (8ch/500Hz)"}

    run_all = "--full" in sys.argv
    if run_all:
        baseline = train_and_evaluate(
            "Baseline (16ch/2kHz)", n_channels=16, ch_stride=1, t_stride=1, sample_rate=2000
        )
        adapted = train_and_evaluate(
            "8ch adapted conv (8ch/500Hz)", n_channels=8, ch_stride=2, t_stride=4, sample_rate=500
        )

    duplicated = train_and_evaluate(
        "8ch duplicated (8ch/500Hz)", n_channels=16, ch_stride=2, t_stride=4,
        sample_rate=500, duplicate_channels=True
    )

    print(f"\n{'=' * 60}")
    print(f"  RESULTS COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Condition':<32} {'Val MAE':>10} {'Test MAE':>10} {'Δ Test':>10}")
    print(f"  {'-' * 62}")
    for r in [baseline, adapted, duplicated]:
        delta = ((r["test_mae"] - baseline["test_mae"]) / baseline["test_mae"]) * 100
        sign = "+" if delta >= 0 else ""
        print(
            f"  {r['name']:<32} {r['best_val_mae']:>10.4f} {r['test_mae']:>10.4f} {sign}{delta:>9.1f}%"
        )
    print()


if __name__ == "__main__":
    main()
