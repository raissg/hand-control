"""
End-to-end pipeline: recording session -> trained model.

Usage:
    # Process a specific recording and train
    python src/pipeline.py train 20260414_150000

    # Process multiple recordings and train
    python src/pipeline.py train 20260414_150000 20260414_160000

    # Process ALL recordings and train
    python src/pipeline.py train --all

    # Run individual steps:
    python src/pipeline.py postprocess 20260414_150000   # extract landmarks only
    python src/pipeline.py convert 20260414_150000       # build HDF5 only
    python src/pipeline.py train-only                    # train on existing HDF5

Steps when you run `train <session_id>`:
    1. Post-process: run MediaPipe on saved video -> landmarks.csv
    2. Convert: align EMG + landmarks -> dataset.hdf5
    3. Train: NeuroPose regression on the dataset
"""

import argparse
import sys
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

RECORDINGS_DIR = _PROJECT_ROOT / "recordings"
DATASET_PATH = _PROJECT_ROOT / "data" / "dataset.hdf5"


def find_sessions(session_ids=None, all_sessions=False):
    """Resolve session IDs to directory paths."""
    if all_sessions:
        sessions = sorted([d.name for d in RECORDINGS_DIR.iterdir() if d.is_dir()])
        if not sessions:
            print("No recordings found in", RECORDINGS_DIR)
            sys.exit(1)
        return sessions
    if not session_ids:
        print("Provide session IDs or --all")
        sys.exit(1)
    # Verify they exist
    for sid in session_ids:
        p = RECORDINGS_DIR / sid
        if not p.is_dir():
            print(f"Session not found: {p}")
            sys.exit(1)
    return session_ids


def step_postprocess(session_ids):
    """Run MediaPipe on recorded videos to extract landmarks."""
    from src.recorder import postprocess_landmarks

    for sid in session_ids:
        session_dir = RECORDINGS_DIR / sid
        landmarks_path = session_dir / "landmarks.csv"

        if landmarks_path.exists():
            print(f"\n[postprocess] {sid}: landmarks.csv already exists, skipping.")
            print(f"  (delete {landmarks_path} to re-extract)")
            continue

        print(f"\n[postprocess] {sid}: extracting landmarks from video...")
        postprocess_landmarks(session_dir)


def step_convert(session_ids, *, include_imu=True, output_name="dataset.hdf5"):
    """Convert EMG + landmarks CSVs to HDF5 dataset."""
    from src.convert_to_hdf5 import build_dataset

    print(f"\n[convert] Building dataset from {len(session_ids)} session(s)...")
    out_path = _PROJECT_ROOT / "data" / output_name
    build_dataset(
        config_path=str(_PROJECT_ROOT / "src" / "config.yaml"),
        session_ids=session_ids,
        include_imu=include_imu,
        output_name=output_name,
    )
    print(f"[convert] Dataset saved to {out_path}")


def step_train(epochs=50, batch_size=32, lr=0.001):
    """Train NeuroPose on the HDF5 dataset."""
    from src.train import main as train_main

    if not DATASET_PATH.exists():
        print(f"[train] Dataset not found: {DATASET_PATH}")
        print("  Run `python src/pipeline.py convert <session_ids>` first.")
        sys.exit(1)

    print(f"\n[train] Starting training on {DATASET_PATH}")
    # Simulate argparse for train.py's main()
    sys.argv = [
        "train.py",
        "--dataset", str(DATASET_PATH),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
    ]
    train_main()


def cmd_train(args):
    """Full pipeline: postprocess -> convert -> train."""
    session_ids = find_sessions(args.sessions, args.all)
    print(f"Pipeline: {len(session_ids)} session(s): {session_ids}")

    step_postprocess(session_ids)
    step_convert(session_ids)
    step_train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


def cmd_postprocess(args):
    session_ids = find_sessions(args.sessions, args.all)
    step_postprocess(session_ids)


def cmd_convert(args):
    session_ids = find_sessions(args.sessions, args.all)
    out = args.output
    if getattr(args, "no_imu", False) and out == "dataset.hdf5":
        out = "dataset_filtered_emg_no_imu.hdf5"
    step_convert(session_ids, include_imu=not args.no_imu, output_name=out)


def cmd_live(args):
    """Run real-time inference and visualization."""
    from src.visualizer import main as live_main
    import sys
    
    # Forward arguments if needed to the visualizer module
    # Or just let visualizer's argparse handle sys.argv.
    # Hack: override sys.argv so argparse in visualizer.main() can consume it
    orig_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    if args.synthetic:
        sys.argv.append("--synthetic")
    try:
        live_main()
    finally:
        sys.argv = orig_argv

def cmd_train_only(args):
    step_train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


def main():
    parser = argparse.ArgumentParser(
        description="EMG hand pose pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py train 20260414_150000
  python src/pipeline.py train --all --epochs 100
  python src/pipeline.py postprocess 20260414_150000
  python src/pipeline.py convert --all
  python src/pipeline.py train-only --epochs 100
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train (full pipeline) ---
    p_train = sub.add_parser("train", help="Full pipeline: postprocess -> convert -> train")
    p_train.add_argument("sessions", nargs="*", help="Session IDs (directory names in recordings/)")
    p_train.add_argument("--all", action="store_true", help="Process all recordings")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=0.001)
    p_train.set_defaults(func=cmd_train)

    # --- postprocess only ---
    p_post = sub.add_parser("postprocess", help="Extract landmarks from recorded video")
    p_post.add_argument("sessions", nargs="*", help="Session IDs")
    p_post.add_argument("--all", action="store_true")
    p_post.set_defaults(func=cmd_postprocess)

    # --- convert only ---
    p_conv = sub.add_parser("convert", help="Build HDF5 dataset from sessions")
    p_conv.add_argument("sessions", nargs="*", help="Session IDs")
    p_conv.add_argument("--all", action="store_true")
    p_conv.add_argument(
        "--no-imu",
        action="store_true",
        help="HDF5 inputs: filtered EMG only (see convert_to_hdf5 --no-imu).",
    )
    p_conv.add_argument(
        "--output",
        type=str,
        default="dataset.hdf5",
        help="HDF5 filename under data/ (default: dataset.hdf5).",
    )
    p_conv.set_defaults(func=cmd_convert)

    # --- live (inference + visualizer) ---
    p_live = sub.add_parser("live", help="Run real-time hand visualization from EMG inference")
    p_live.add_argument("--synthetic", action="store_true", help="Use synthetic hardware stream")
    p_live.set_defaults(func=cmd_live)

    # --- train-only (skip postprocess/convert) ---
    p_tonly = sub.add_parser("train-only", help="Train on existing HDF5 dataset")
    p_tonly.add_argument("--epochs", type=int, default=50)
    p_tonly.add_argument("--batch_size", type=int, default=32)
    p_tonly.add_argument("--lr", type=float, default=0.001)
    p_tonly.set_defaults(func=cmd_train_only)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
