import argparse
import logging
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from src.acquire import EMGStream
from src.inference import EMGPredictor
from src.ground_truth import HAND_CONNECTIONS, WRIST
from src.session_3d_viewer import _style_3d_hand_axes, redraw_skeleton_3d

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Live 3D hand visualizer from EMG inference.")
    parser.add_argument("--model", type=str, default="models/neuropose_mindrove_best.pt", help="Path to PyTorch model.")
    parser.add_argument("--config", type=str, default="emg2pose/config/network/neuropose_mindrove.yaml", help="Path to NeuroPose config.")
    parser.add_argument("--dataset", type=str, default="data/dataset.hdf5", help="Path to HDF5 to read norm stats.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic EMG stream (for testing without band).")
    parser.add_argument("--fps", type=int, default=30, help="Target visualization FPS.")
    parser.add_argument("--ema", type=float, default=0.5, help="EMA alpha (1.0 = no smoothing).")
    parser.add_argument("--no-mirror-fix", action="store_true",
                        help="Skip the X-axis flip. Recorder flipped frames before landmarking, "
                             "so ground truth is in mirrored space; we un-flip here by default.")
    parser.add_argument("--debug", action="store_true", help="Print per-frame prediction diagnostics.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip the offline bandpass+notch filter. Use for models trained "
                             "on raw EMG (dataset built with convert_to_hdf5.py --no-filter).")
    parser.add_argument("--emg-only", action="store_true",
                        help="Model expects 8-ch EMG-only input (no IMU). Slices off accel/gyro "
                             "from the stream. Must match the dataset used for norm stats.")
    args = parser.parse_args()

    # 1. Initialize hardware stream
    logger.info(f"Starting EMG Stream (synthetic={args.synthetic})...")
    # enable_filter=False: training applies bandpass+notch offline (filtfilt)
    # on the rolling window. EMGPredictor mirrors that, so the stream must
    # deliver raw samples.
    stream = EMGStream(synthetic=args.synthetic, enable_filter=False)
    try:
        stream.start()
    except Exception as e:
        logger.error(f"Failed to start EMG stream: {e}")
        sys.exit(1)

    # 2. Initialize Inference predictor
    predictor = EMGPredictor(
        model_path=args.model,
        config_path=args.config,
        dataset_path=args.dataset,
        ema_alpha=args.ema,
        filter_enable=not args.no_filter,
        emg_only=args.emg_only,
    )
    predictor.start_background_loop(stream, fps=args.fps + 10)

    # 3. Setup Matplotlib 3D plotting
    plt.style.use("dark_background")
    fig = plt.figure("Real-time EMG -> 3D Hand", figsize=(8, 8))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax_3d = fig.add_subplot(111, projection="3d")
    
    elev, azim = -22.0, -60.0
    redraw_skeleton_3d(ax_3d, None, elev, azim)

    logger.info("Entering visualization loop. Close the window to stop.")
    
    prev_pred = None
    frame_idx = 0
    try:
        while plt.fignum_exists(fig.number):
            elev = ax_3d.elev
            azim = ax_3d.azim

            xyz_3d = predictor.predict()  # (20, 3)

            if args.debug:
                # Per-frame: input buffer energy + prediction norm + change since last frame
                buf = np.array(predictor.buffer, dtype=np.float32)
                emg_rms = float(np.sqrt(np.mean(buf[:, :8] ** 2)))
                delta = 0.0 if prev_pred is None else float(np.linalg.norm(xyz_3d - prev_pred))
                pred_span = float(xyz_3d.max() - xyz_3d.min())
                if frame_idx % 10 == 0:
                    logger.info(
                        f"frame {frame_idx}: emg_rms={emg_rms:.2f} "
                        f"pred_span={pred_span:.3f} delta={delta:.4f}"
                    )
                prev_pred = xyz_3d.copy()
                frame_idx += 1

            # Undo the recorder's horizontal flip: predicted landmarks live in
            # flipped-frame coordinates, so negate X for display.
            if not args.no_mirror_fix:
                xyz_3d = xyz_3d.copy()
                xyz_3d[:, 0] = -xyz_3d[:, 0]

            # Prepend wrist at origin (removed during preprocessing) -> 21 joints for drawing.
            if xyz_3d.shape[0] == 20:
                xyz_3d = np.vstack([np.zeros((1, 3)), xyz_3d])

            redraw_skeleton_3d(ax_3d, xyz_3d, elev, azim)

            # Process UI events
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            # Rate limit
            plt.pause(1.0 / args.fps)
            
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Stopping...")
        predictor.stop()
        stream.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()