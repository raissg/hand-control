"""NaviFlame-inspired 5-bit finger-bend classifier.

Architecture:
    Input:  (B, 8, 100)   - 8ch EMG, 200ms @ 500Hz
    Feature extractor:
        Conv1d(8->32, k=5)  BN ReLU  MaxPool(2)   -> (B, 32, 50)
        Conv1d(32->64, k=5) BN ReLU  MaxPool(2)   -> (B, 64, 25)
        Conv1d(64->128,k=5) BN ReLU  AdaptiveAvg  -> (B, 128, 1)
    Head:
        Flatten -> Linear(128->32) -> ReLU -> Dropout
                -> Linear(32->16) -> ReLU
                -> Linear(16->5)           (logits)
    Loss: BCEWithLogitsLoss  (5 independent sigmoids)
    Output at inference: sigmoid(logits) > 0.5  -> 5-bit string
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMG5Bit(nn.Module):
    def __init__(self, in_channels: int = 8, n_out: int = 5, dropout: float = 0.3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 32), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 8, T). Returns (B, 5) logits."""
        return self.head(self.feat(x))


# === IMU ADDITION =========================================================
# Two-stream model: EMG branch (fast bursts, kernel 5, two pools) + IMU
# branch (slow gravity/rotation signal, larger kernels, stronger pooling).
# Each branch normalizes its own stats with its own BN. The head sees the
# concatenated [emg_emb || imu_emb] vector.
# ==========================================================================
class EMG5BitMulti(nn.Module):
    """EMG + IMU two-stream classifier.

    Inputs:
        emg: (B, 8, T_emg)   — same window as EMG5Bit (e.g. 100 samples).
        imu: (B, 6, T_imu)   — same temporal window, can be the same length
                                since MindRove streams IMU aligned at 500 Hz.
                                If IMU is upsampled/decimated elsewhere, the
                                AdaptiveAvgPool at the end makes T_imu free.
    Output: (B, n_out) logits.
    """

    def __init__(
        self,
        emg_channels: int = 8,
        imu_channels: int = 6,
        n_out: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        # EMG branch — identical to EMG5Bit's feat.
        self.emg_feat = nn.Sequential(
            nn.Conv1d(emg_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),                                      # -> (B, 128)
        )

        # === IMU ADDITION ==================================================
        # IMU branch: slower-changing signal (gravity, wrist rotation), so we
        # use bigger kernels (k=7, then k=5) and aggressive pooling (4 then 2)
        # to summarize a 200 ms window with very few feature maps.
        # ===================================================================
        self.imu_feat = nn.Sequential(
            nn.Conv1d(imu_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),                                      # -> (B, 32)
        )

        # Head consumes the concatenated 128 + 32 = 160-D embedding.
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_out),
        )

    def forward(self, emg: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        emg_emb = self.emg_feat(emg)                           # (B, 128)
        imu_emb = self.imu_feat(imu)                           # (B,  32)
        return self.head(torch.cat([emg_emb, imu_emb], dim=1))
