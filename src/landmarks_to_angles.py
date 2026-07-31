"""Convert MediaPipe 3D hand landmarks → 20 joint angles (UMETrack / emg2pose schema).

Uses the same 3D geometry as the live angle visualizer:

FE (flexion/extension): arccos(dot(unit(a), unit(b))) in full 3D.
AA (abduction/adduction): signed angle in the palm plane (⊥ palm normal,
    built from (index MCP − wrist) × (pinky MCP − wrist)).

Joint ordering (matches emg2pose.constants.JOINTS):
   0 THUMB_CMC_FE    10 MIDDLE_PIP_FE
   1 THUMB_CMC_AA    11 MIDDLE_DIP_FE
   2 THUMB_MCP_FE    12 RING_MCP_AA
   3 THUMB_IP_FE     13 RING_MCP_FE
   4 INDEX_MCP_AA    14 RING_PIP_FE
   5 INDEX_MCP_FE    15 RING_DIP_FE
   6 INDEX_PIP_FE    16 PINKY_MCP_AA
   7 INDEX_DIP_FE    17 PINKY_MCP_FE
   8 MIDDLE_MCP_AA   18 PINKY_PIP_FE
   9 MIDDLE_MCP_FE   19 PINKY_DIP_FE

All angles in radians.

MediaPipe landmark indices:
  0 WRIST
  1-4   THUMB:  CMC, MCP, IP, TIP
  5-8   INDEX:  MCP, PIP, DIP, TIP
  9-12  MIDDLE: MCP, PIP, DIP, TIP
  13-16 RING:   MCP, PIP, DIP, TIP
  17-20 PINKY:  MCP, PIP, DIP, TIP
"""

from __future__ import annotations
import numpy as np

WRIST  = 0
THUMB  = (1, 2, 3, 4)
INDEX  = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING   = (13, 14, 15, 16)
PINKY  = (17, 18, 19, 20)

NUM_ANGLES = 20

ANGLE_NAMES = [
    "THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
    "INDEX_MCP_AA", "INDEX_MCP_FE", "INDEX_PIP_FE", "INDEX_DIP_FE",
    "MIDDLE_MCP_AA", "MIDDLE_MCP_FE", "MIDDLE_PIP_FE", "MIDDLE_DIP_FE",
    "RING_MCP_AA", "RING_MCP_FE", "RING_PIP_FE", "RING_DIP_FE",
    "PINKY_MCP_AA", "PINKY_MCP_FE", "PINKY_PIP_FE", "PINKY_DIP_FE",
]

ANGLE_RANGES_RAD = np.array([
    (-1.0,  1.3),   # THUMB_CMC_FE
    (-1.0,  1.3),   # THUMB_CMC_AA
    ( 0.0,  1.6),   # THUMB_MCP_FE
    ( 0.0,  1.6),   # THUMB_IP_FE
    (-0.7,  0.7),   # INDEX_MCP_AA
    ( 0.0,  2.0),   # INDEX_MCP_FE
    ( 0.0,  2.0),   # INDEX_PIP_FE
    ( 0.0,  2.0),   # INDEX_DIP_FE
    (-0.7,  0.7),   # MIDDLE_MCP_AA
    ( 0.0,  2.0),   # MIDDLE_MCP_FE
    ( 0.0,  2.0),   # MIDDLE_PIP_FE
    ( 0.0,  2.0),   # MIDDLE_DIP_FE
    (-0.7,  0.7),   # RING_MCP_AA
    ( 0.0,  2.0),   # RING_MCP_FE
    ( 0.0,  2.0),   # RING_PIP_FE
    ( 0.0,  2.0),   # RING_DIP_FE
    (-0.7,  0.7),   # PINKY_MCP_AA
    ( 0.0,  2.0),   # PINKY_MCP_FE
    ( 0.0,  2.0),   # PINKY_PIP_FE
    ( 0.0,  2.0),   # PINKY_DIP_FE
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit3(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return np.asarray(v, dtype=np.float64) / (n + eps)


def _angle_fe_3d(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    """Unsigned angle in [0, π] between 3D bone vectors a and b."""
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + eps
    cos_th = float(np.dot(a, b)) / denom
    return float(np.arccos(np.clip(cos_th, -1.0 + eps, 1.0 - eps)))


def _palm_normal(lm: np.ndarray) -> np.ndarray:
    """Palm normal: (index MCP − wrist) × (pinky MCP − wrist), unit length."""
    vi = lm[INDEX[0]] - lm[WRIST]
    vp = lm[PINKY[0]] - lm[WRIST]
    n = np.cross(vi, vp)
    if np.linalg.norm(n) < 1e-9:
        n = np.cross(vi, lm[MIDDLE[0]] - lm[WRIST])
    return _unit3(n)


def _signed_angle_in_plane(
    v1: np.ndarray, v2: np.ndarray, plane_normal: np.ndarray, eps: float = 1e-9,
) -> float:
    """Signed angle from v1 to v2 after projecting both onto plane ⊥ plane_normal."""
    n = _unit3(plane_normal, eps)
    p1 = v1 - n * np.dot(v1, n)
    p2 = v2 - n * np.dot(v2, n)
    if np.linalg.norm(p1) < eps or np.linalg.norm(p2) < eps:
        return 0.0
    u1 = p1 / np.linalg.norm(p1)
    u2 = p2 / np.linalg.norm(p2)
    sin_th = float(np.dot(n, np.cross(u1, u2)))
    cos_th = float(np.clip(np.dot(u1, u2), -1.0 + eps, 1.0 - eps))
    return float(np.arctan2(sin_th, cos_th))


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def landmarks_to_angles(lm: np.ndarray) -> np.ndarray:
    """Convert one frame of MediaPipe world landmarks to 20 joint angles (radians).

    Args:
        lm: (21, 3) MediaPipe world landmarks (raw or wrist-centered — angles
            are translation- and scale-invariant so either works).

    Returns:
        (20,) float32 array in ANGLE_NAMES order.
    """
    assert lm.shape == (21, 3), f"expected (21,3), got {lm.shape}"
    lm = lm.astype(np.float64, copy=True)
    out = np.zeros(NUM_ANGLES, dtype=np.float64)

    palm_n = _palm_normal(lm)

    # THUMB
    thumb_mc = lm[THUMB[0]] - lm[WRIST]
    cmc_bone = lm[THUMB[1]] - lm[THUMB[0]]
    mcp_bone = lm[THUMB[2]] - lm[THUMB[1]]
    ip_bone  = lm[THUMB[3]] - lm[THUMB[2]]

    out[0] = _angle_fe_3d(thumb_mc, cmc_bone)
    out[1] = _signed_angle_in_plane(thumb_mc, cmc_bone, palm_n)
    out[2] = _angle_fe_3d(cmc_bone, mcp_bone)
    out[3] = _angle_fe_3d(mcp_bone, ip_bone)

    # FOUR FINGERS
    for F, i_aa, i_fe, i_pip, i_dip in (
        (INDEX,  4,  5,  6,  7),
        (MIDDLE, 8,  9, 10, 11),
        (RING,  12, 13, 14, 15),
        (PINKY, 16, 17, 18, 19),
    ):
        mcp, pip, dip, tip = F
        metacarpal  = lm[mcp] - lm[WRIST]
        proximal    = lm[pip] - lm[mcp]
        middle_bone = lm[dip] - lm[pip]
        distal_bone = lm[tip] - lm[dip]

        out[i_aa]  = _signed_angle_in_plane(metacarpal, proximal, palm_n)
        out[i_fe]  = _angle_fe_3d(metacarpal, proximal)
        out[i_pip] = _angle_fe_3d(proximal, middle_bone)
        out[i_dip] = _angle_fe_3d(middle_bone, distal_bone)

    return out.astype(np.float32)


def landmarks_seq_to_angles(lm_seq: np.ndarray) -> np.ndarray:
    """Vectorized over time. lm_seq: (T, 21, 3) → (T, 20)."""
    T = lm_seq.shape[0]
    out = np.empty((T, NUM_ANGLES), dtype=np.float32)
    for t in range(T):
        out[t] = landmarks_to_angles(lm_seq[t])
    return out
