"""NeuroPose with a bounded, anatomy-aware angle head.

Wraps `emg2pose.networks.NeuroPose` and adds three things the paper requires
but the stock Meta code leaves out:

  1. **bReLU output bounds** (Liu 2021 §4.1, eq. 4): each of the 20 angles is
     constrained to its anatomical range via a sigmoid-scaled clamp. We use
     sigmoid rather than a hard clamp so gradients stay alive when a
     prediction starts outside the range at init.

  2. **Anatomical derivation** (eqs. 1, 2): when `derive_coupled=True`, the
     four DIP angles and the thumb IP angle are computed from the PIP / MCP_FE
     they should kinematically follow, matching Liu's claim that "the actual
     output of the network is only 16 dimensions."

  3. **Smoothness loss** helper (eq. 13): returns ‖∇θ_t − ∇θ_{t−1}‖² over the
     time axis. Call it from the training loop and add to the per-joint MSE.

Output is still (B, 20, T) so downstream code doesn't need to branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from src.landmarks_to_angles import ANGLE_NAMES, ANGLE_RANGES_RAD


# Indices in the 20-angle vector that are kinematically slaved to another
# angle. Derivation rules come from Liu 2021 eqs. 1 and 2.
#
# eq. 1:  θ_dip = (2/3) · θ_pip          (four fingers)
# eq. 2:  θ_ip  = (1/2) · θ_mcp_fe       (thumb)
COUPLED_DERIVATIONS = [
    #   out_idx,   driver_idx,  factor
    (3,  2, 0.5),   # THUMB_IP_FE  = 0.5 * THUMB_MCP_FE
    (7,  6, 2 / 3), # INDEX_DIP_FE = 2/3 * INDEX_PIP_FE
    (11, 10, 2 / 3),# MIDDLE_DIP_FE
    (15, 14, 2 / 3),# RING_DIP_FE
    (19, 18, 2 / 3),# PINKY_DIP_FE
]


class BoundedAngleHead(nn.Module):
    """Maps unbounded `(B, 20, T)` logits → radians in per-joint [lo, hi].

    Uses `lo + (hi - lo) * sigmoid(x)`. Liu's bReLU is a hard clamp; sigmoid is
    the smooth equivalent and trains more reliably. Behaviorally identical in
    the saturation limit, which is what the loss will push toward anyway.
    """

    def __init__(self, ranges_rad: np.ndarray):
        super().__init__()
        assert ranges_rad.shape == (20, 2)
        lo = torch.tensor(ranges_rad[:, 0], dtype=torch.float32).view(1, 20, 1)
        hi = torch.tensor(ranges_rad[:, 1], dtype=torch.float32).view(1, 20, 1)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.lo + (self.hi - self.lo) * torch.sigmoid(logits)


class NeuroPoseAngles(nn.Module):
    """NeuroPose backbone + bounded-angle head + coupled-angle derivation."""

    def __init__(
        self,
        backbone: nn.Module,                        # emg2pose.networks.NeuroPose
        ranges_rad: np.ndarray = ANGLE_RANGES_RAD,
        derive_coupled: bool = True,
    ):
        super().__init__()
        assert getattr(backbone, "linear", None) is not None, \
            "backbone must be a NeuroPose with a final Linear layer"
        # The backbone's linear must emit 20 channels.
        out_dim = backbone.linear.out_features
        assert out_dim == 20, (
            f"backbone Linear should emit 20 logits, got {out_dim}. "
            f"Set `out_channels: 20` in the network yaml."
        )

        self.backbone = backbone
        self.head = BoundedAngleHead(ranges_rad)
        self.derive_coupled = derive_coupled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)          # (B, 20, T), unbounded
        angles = self.head(logits)         # (B, 20, T), bounded per joint

        if self.derive_coupled:
            # Replace slaved joints with their driver * factor. Done in-place on
            # a clone to preserve autograd, so gradients flow through the driver.
            angles = angles.clone()
            for out_idx, drv_idx, factor in COUPLED_DERIVATIONS:
                angles[:, out_idx, :] = factor * angles[:, drv_idx, :]

        return angles


# ---------------------------------------------------------------------------
# Loss functions (Liu 2021 §4.1)
# ---------------------------------------------------------------------------
def angle_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-joint MSE, averaged. `pred`, `target`: (B, 20, T). Paper uses MSE."""
    se = (pred - target) ** 2
    if mask is not None:
        se = se * mask[:, None, :]              # (B, 1, T) broadcasts to (B,20,T)
        return se.sum() / (mask.sum() * 20 + 1e-9)
    return se.mean()


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Liu 2021 eq. 13:  ‖∇θ_t − ∇θ_{t-1}‖²  (second derivative in time).

    Encourages constant-velocity motion — penalizes jerk.
    `pred`: (B, 20, T).
    """
    # First difference: velocity
    v = pred[..., 1:] - pred[..., :-1]         # (B, 20, T-1)
    # Second difference: jerk
    a = v[..., 1:] - v[..., :-1]               # (B, 20, T-2)
    return (a ** 2).mean()


def neuropose_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth_weight: float = 0.1,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Loss matching Liu 2021 eq. 14 (minus the per-joint-group decomposition).

    Returns (scalar_loss, {'mse': ..., 'smooth': ...}) for logging.
    """
    mse = angle_mse_loss(pred, target, mask)
    smooth = smoothness_loss(pred)
    total = mse + smooth_weight * smooth
    return total, {"mse": mse.detach(), "smooth": smooth.detach()}


# ---------------------------------------------------------------------------
# Builder: construct NeuroPoseAngles from a yaml config, matching train.get_model
# ---------------------------------------------------------------------------
def get_angle_model(config_path: str | Path,
                    derive_coupled: bool = True) -> NeuroPoseAngles:
    """Build a NeuroPoseAngles from a network yaml.

    The yaml must set `out_channels: 20` — no linear re-projection happens here.
    """
    from src.train import get_model
    backbone = get_model(str(config_path))
    return NeuroPoseAngles(backbone, ranges_rad=ANGLE_RANGES_RAD,
                           derive_coupled=derive_coupled)


if __name__ == "__main__":
    # Smoke test: forward pass shape + bounds.
    model = get_angle_model("emg2pose/config/network/neuropose_angles.yaml")
    model.eval()
    x = torch.randn(2, 8, 1000)  # EMG-only, 2s @ 500Hz
    with torch.no_grad():
        y = model(x)
    print(f"out shape: {tuple(y.shape)}  (expect (2, 20, 1000))")
    print(f"per-joint min/max over the batch (rad):")
    for i, name in enumerate(ANGLE_NAMES):
        lo, hi = float(y[:, i].min()), float(y[:, i].max())
        r_lo, r_hi = ANGLE_RANGES_RAD[i]
        in_range = (r_lo <= lo) and (hi <= r_hi)
        mark = "✓" if in_range else "✗"
        print(f"  {mark} {name:15s} "
              f"pred [{lo:+.2f}, {hi:+.2f}]  range [{r_lo:+.2f}, {r_hi:+.2f}]")
