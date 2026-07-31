"""More experiments on the nature of the damping:

4. What does the model predict for ZERO input? That's its "learned rest pose"
   — the prediction it falls back on when it has no signal. If it's close to
   the mean training pose, the model treats the mean as a default.

5. PCA comparison: how many independent directions of motion does the model
   actually use? If predictions live in a low-rank subspace vs. the full
   truth-space, the model can ONLY move in a few directions — any real motion
   orthogonal to its learned subspace is invisible to it.

6. Per-joint correlation: which joints are tracked well (r > 0.5 per coord
   across windows) and which ones are not tracked at all (|r| < 0.2).

7. "Best-case" windows: the single window where the model gets closest to
   truth. Is its damping ratio closer to 1.0 there? That would mean the model
   IS capable of full amplitude, just usually chooses not to.
"""

import argparse

import h5py
import numpy as np
import torch

from src.convert_to_hdf5 import filter_emg
from src.train import get_model

NUM_EMG = 8
WINDOW = 1000


def gather_center(h5f, split, mean, std, n, rng, stride=200):
    sessions = list(h5f[split].keys())
    X_list, Y_list, Y_full = [], [], []
    for s in sessions:
        emg = h5f[split][s]["emg"][:]
        lm = h5f[split][s]["landmarks"][:]
        T = len(emg)
        starts = list(range(0, T - WINDOW + 1, stride))
        rng.shuffle(starts)
        starts = starts[:n]
        for start in starts:
            e = emg[start:start + WINDOW].astype(np.float32)
            l = lm[start:start + WINDOW].astype(np.float32)
            e[:, :NUM_EMG] = filter_emg(e[:, :NUM_EMG], fs=500)
            e = (e - mean) / std
            X_list.append(e.T)
            Y_list.append(l[WINDOW // 2])     # center label
            Y_full.append(l.T)
    return np.stack(X_list), np.stack(Y_list), np.stack(Y_full)


def run_model(model, X, device, batch=32):
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i+batch]).float().to(device)
        with torch.no_grad():
            o = model(xb).cpu().numpy()
        outs.append(o)
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.hdf5")
    ap.add_argument("--model", default="models/neuropose_mindrove_best.pt")
    ap.add_argument("--config", default="emg2pose/config/network/neuropose_mindrove.yaml")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--emg-only", action="store_true",
                    help="Slice model input to first 8 channels (must match the checkpoint's training).")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    device = torch.device("cpu")
    model = get_model(args.config).to(device)
    state = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    with h5py.File(args.dataset, "r") as f:
        mean = np.asarray(f.attrs["input_mean"], dtype=np.float32)
        std = np.asarray(f.attrs["input_std"], dtype=np.float32)
        X, Y_center, Y_full = gather_center(f, "train", mean, std, args.n, rng)

    if args.emg_only:
        X = X[:, :NUM_EMG, :]
        print(f"--emg-only: sliced model input to {X.shape[1]} channels")

    print(f"Train windows gathered: {len(X)}")

    # ==============================================================
    # EXP 4 — "Rest pose": what does the model predict for zero input?
    # ==============================================================
    print("\n" + "=" * 60)
    print("EXP 4 — Learned 'rest pose' (model output on zero input)")
    print("=" * 60)
    with torch.no_grad():
        n_ch = X.shape[1]
        zero_in = torch.zeros(1, n_ch, WINDOW).float()
        rest = model(zero_in).cpu().numpy()[0, :, WINDOW // 2]  # (60,)
    mean_pose = Y_center.mean(axis=0)
    diff = np.linalg.norm(rest - mean_pose)
    pose_extent = np.linalg.norm(mean_pose)
    print(f"  ||rest_pose||            = {np.linalg.norm(rest):.4f}")
    print(f"  ||dataset_mean_pose||    = {pose_extent:.4f}")
    print(f"  ||rest - dataset_mean||  = {diff:.4f}")
    print(f"  relative distance        = {diff / (pose_extent + 1e-9):.4f}")
    print(f"  (small relative distance ⇒ 'rest pose' ≈ mean training pose)")

    # Also: on every train window, how close is the prediction to 'rest'?
    P = run_model(model, X, device)                         # (N, 60, T)
    P_center = P[:, :, WINDOW // 2]                          # (N, 60)
    dists_to_rest = np.linalg.norm(P_center - rest, axis=1)  # (N,)
    dists_to_mean = np.linalg.norm(P_center - mean_pose, axis=1)
    truth_dists_to_mean = np.linalg.norm(Y_center - mean_pose, axis=1)
    print(f"\n  Per-window ||pred - rest||: mean={dists_to_rest.mean():.4f} std={dists_to_rest.std():.4f}")
    print(f"  Per-window ||pred - dataset_mean||: mean={dists_to_mean.mean():.4f} std={dists_to_mean.std():.4f}")
    print(f"  Per-window ||truth - dataset_mean||: mean={truth_dists_to_mean.mean():.4f} std={truth_dists_to_mean.std():.4f}")
    print(f"  Pred stays {dists_to_mean.mean() / truth_dists_to_mean.mean():.3f}x as far from the mean as truth does.")

    # ==============================================================
    # EXP 5 — PCA of predictions vs truth
    # ==============================================================
    print("\n" + "=" * 60)
    print("EXP 5 — Effective rank of motion (PCA of center predictions vs truth)")
    print("=" * 60)
    def effective_rank(M):
        # Remove the mean, then explained variance ratios.
        Mc = M - M.mean(axis=0, keepdims=True)
        # SVD is overkill but fine for 500x60.
        s = np.linalg.svd(Mc, compute_uv=False)
        var = s ** 2
        var /= var.sum()
        cum = np.cumsum(var)
        rank80 = int(np.searchsorted(cum, 0.80)) + 1
        rank95 = int(np.searchsorted(cum, 0.95)) + 1
        rank99 = int(np.searchsorted(cum, 0.99)) + 1
        return var, rank80, rank95, rank99

    v_t, r80_t, r95_t, r99_t = effective_rank(Y_center)
    v_p, r80_p, r95_p, r99_p = effective_rank(P_center)
    print(f"  Truth: ranks for 80/95/99% var = {r80_t}/{r95_t}/{r99_t}   top-5 var = {np.round(v_t[:5], 3).tolist()}")
    print(f"  Pred : ranks for 80/95/99% var = {r80_p}/{r95_p}/{r99_p}   top-5 var = {np.round(v_p[:5], 3).tolist()}")

    # Alignment: how much of the prediction's variance is explained by the truth's top components?
    Yc = Y_center - Y_center.mean(axis=0, keepdims=True)
    Pc = P_center - P_center.mean(axis=0, keepdims=True)
    U_t, _, _ = np.linalg.svd(Yc, full_matrices=False)
    # Project predictions onto the subspace spanned by the top-k truth PCs
    for k in [2, 5, 10]:
        basis = U_t[:, :k]            # (N, k)
        # Project Pc onto that basis
        proj = basis @ (basis.T @ Pc)  # (N, 60)
        frac = (proj ** 2).sum() / ((Pc ** 2).sum() + 1e-9)
        print(f"  Fraction of prediction variance living in truth's top-{k:<2d} PCs: {frac:.3f}")

    # ==============================================================
    # EXP 6 — Per-coord correlation stratification
    # ==============================================================
    print("\n" + "=" * 60)
    print("EXP 6 — Per-coord correlation stratification")
    print("=" * 60)
    rs = np.zeros(60)
    for c in range(60):
        p = P_center[:, c]
        t = Y_center[:, c]
        if p.std() < 1e-8 or t.std() < 1e-8:
            rs[c] = 0.0
        else:
            rs[c] = np.corrcoef(p, t)[0, 1]
    # Map back to joints (0..19) after wrist removed, axis = x/y/z
    joint_axes = ["thumb_CMC", "thumb_MCP", "thumb_IP", "thumb_tip",
                  "idx_MCP", "idx_PIP", "idx_DIP", "idx_tip",
                  "mid_MCP", "mid_PIP", "mid_DIP", "mid_tip",
                  "ring_MCP", "ring_PIP", "ring_DIP", "ring_tip",
                  "pnk_MCP", "pnk_PIP", "pnk_DIP", "pnk_tip"]
    print(f"  Coords with r > 0.5: {int((rs > 0.5).sum())}/60")
    print(f"  Coords with r > 0.3: {int((rs > 0.3).sum())}/60")
    print(f"  Coords with |r| < 0.2: {int((np.abs(rs) < 0.2).sum())}/60")
    # Per-joint summary (mean of 3 axes)
    per_joint_r = rs.reshape(20, 3).mean(axis=1)
    order = np.argsort(-per_joint_r)
    print("  Best-tracked joints (mean r over xyz):")
    for i in order[:5]:
        print(f"    {joint_axes[i]:12s}  r̄={per_joint_r[i]:+.3f}  (x={rs[i*3]:+.2f}, y={rs[i*3+1]:+.2f}, z={rs[i*3+2]:+.2f})")
    print("  Worst-tracked joints:")
    for i in order[-5:]:
        print(f"    {joint_axes[i]:12s}  r̄={per_joint_r[i]:+.3f}  (x={rs[i*3]:+.2f}, y={rs[i*3+1]:+.2f}, z={rs[i*3+2]:+.2f})")

    # ==============================================================
    # EXP 7 — Best-case windows
    # ==============================================================
    print("\n" + "=" * 60)
    print("EXP 7 — Best-case windows: does the model EVER hit full amplitude?")
    print("=" * 60)
    # For each window: truth "activity" = std of truth landmarks across time within window,
    # pred activity = same for pred. Look at ratios sorted.
    truth_act = Y_full.std(axis=2).mean(axis=1)          # (N,)
    pred_act = P.std(axis=2).mean(axis=1)                # (N,)
    ratios = pred_act / (truth_act + 1e-9)
    print(f"  pred/truth amplitude ratio: min={ratios.min():.3f}  median={np.median(ratios):.3f}  max={ratios.max():.3f}")
    # Print top-5 and bottom-5 with their energies
    order = np.argsort(-ratios)
    print("  Top-5 windows (pred matches or exceeds truth):")
    for i in order[:5]:
        print(f"    ratio={ratios[i]:.3f}  pred_act={pred_act[i]:.4f}  truth_act={truth_act[i]:.4f}")
    print("  Bottom-5 windows (most damped):")
    for i in order[-5:]:
        print(f"    ratio={ratios[i]:.3f}  pred_act={pred_act[i]:.4f}  truth_act={truth_act[i]:.4f}")


if __name__ == "__main__":
    main()
