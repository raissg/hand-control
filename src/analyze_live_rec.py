"""Analyze a live_compare5 CSV: is the prediction monotonic with GT?

For each of the 5 joints:
  - Pearson r  (linear)
  - Spearman ρ (monotonic — directly tests "GT↑ ⇒ Pred↑")
  - Linear fit pred = a·gt + b, R²
  - Derivative check: Pearson r between d(gt)/dt and d(pred)/dt
    (does the prediction *move with* the hand, not just hover?)
  - Scatter plot + velocity scatter

Usage:
    python -m src.analyze_live_rec recordings/live_rec_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


LABELS = ["T-CMC", "I-MCP", "M-MCP", "R-MCP", "P-MCP"]


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x, y):
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _linear_fit(x, y):
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2)


def load_csv(path):
    ts = []; gt = []; pr = []
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        gt_cols = [i for i, h in enumerate(header) if h.startswith("gt_")]
        pr_cols = [i for i, h in enumerate(header) if h.startswith("pred_")]
        for row in r:
            ts.append(float(row[0]))
            gt.append([float(row[i]) for i in gt_cols])
            pr.append([float(row[i]) for i in pr_cols])
    return np.array(ts), np.array(gt), np.array(pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--plot_out", default=None)
    args = ap.parse_args()

    path = Path(args.csv_path)
    ts, gt, pr = load_csv(path)
    N, J = gt.shape
    print(f"Loaded {N} frames from {path.name} ({ts[-1]-ts[0]:.1f} s)")
    print()

    print("=" * 92)
    print(f"{'Joint':<8} {'N':>5} {'GTrng':>10} {'PRrng':>10} "
          f"{'Pearson':>9} {'Spearman':>9} {'slope_a':>9} {'b':>8} {'R2':>7} {'dρ':>7}")
    print("-" * 92)

    pearsons = []
    spearmans = []
    slopes = []
    r2s = []
    dcorrs = []

    for j in range(J):
        x = gt[:, j]; y = pr[:, j]
        p = _pearson(x, y)
        s = _spearman(x, y)
        a, b, r2 = _linear_fit(x, y)

        # Derivative / velocity: Pearson on diffs (ignores DC offset)
        dx = np.diff(x); dy = np.diff(y)
        dp = _pearson(dx, dy)

        pearsons.append(p); spearmans.append(s)
        slopes.append(a); r2s.append(r2); dcorrs.append(dp)

        print(f"{LABELS[j]:<8} {N:>5d} {x.ptp():>9.1f}° {y.ptp():>9.1f}° "
              f"{p:>9.3f} {s:>9.3f} {a:>9.3f} {b:>+8.1f} {r2:>7.3f} {dp:>7.3f}")

    print("-" * 92)
    print(f"{'MEAN':<8} {'':>5} {'':>10} {'':>10} "
          f"{np.mean(pearsons):>9.3f} {np.mean(spearmans):>9.3f} "
          f"{np.mean(slopes):>9.3f} {'':>8} {np.mean(r2s):>7.3f} "
          f"{np.mean(dcorrs):>7.3f}")
    print("=" * 92)
    print()
    print("Guide:")
    print("  Spearman ρ:  >0.7 strong monotonic, 0.4-0.7 moderate, <0.2 weak/none, <0 inverted")
    print("  slope_a:     >0 claim holds; ~1 = well-scaled; <0 inverted")
    print("  R²:          fraction of GT variance a linear pred explains")
    print("  dρ:          velocity correlation — does pred MOVE with hand (not just sit near mean)?")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = args.plot_out or str(path.with_suffix("")) + "_analysis.png"
        fig, axes = plt.subplots(2, J, figsize=(3.2 * J, 6.2))

        for j in range(J):
            ax = axes[0, j]
            x = gt[:, j]; y = pr[:, j]
            ax.scatter(x, y, s=4, alpha=0.35, color="tab:red")
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y=x")
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slopes[j] * xs + np.polyfit(x, y, 1)[1],
                    color="tab:blue", lw=1.2, label=f"a={slopes[j]:.2f}")
            ax.set_title(
                f"{LABELS[j]}\nρ={spearmans[j]:.2f}  r={pearsons[j]:.2f}  R²={r2s[j]:.2f}",
                fontsize=9,
            )
            ax.set_xlabel("GT (°)"); ax.set_ylabel("Pred (°)")
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)

            ax2 = axes[1, j]
            dx = np.diff(x); dy = np.diff(y)
            ax2.scatter(dx, dy, s=3, alpha=0.3, color="tab:green")
            ax2.axhline(0, color="k", lw=0.5); ax2.axvline(0, color="k", lw=0.5)
            ax2.set_title(f"velocity  dρ={dcorrs[j]:.2f}", fontsize=9)
            ax2.set_xlabel("d(GT)/Δt"); ax2.set_ylabel("d(Pred)/Δt")
            ax2.tick_params(labelsize=7)

        fig.suptitle(f"{path.name} — {N} frames", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out, dpi=110)
        print(f"Plot saved: {out}")
    except ImportError:
        print("matplotlib missing; skipping plots.")


if __name__ == "__main__":
    main()
