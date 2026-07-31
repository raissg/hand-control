"""
Minimal webcam -> mp4 test. Prints per-frame timing so we can see
whether frames are actually being captured/written at the expected rate.

Probes cameras 0..N-1 and records from each that opens.

Usage:
    python src/test_video.py                    # all cameras 0..3, 5s each
    python src/test_video.py --max-cameras 6 --seconds 3 --fourcc mp4v
    python src/test_video.py --camera 1         # only camera 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


BACKENDS = [
    ("AVFOUNDATION", cv2.CAP_AVFOUNDATION),
    ("ANY", cv2.CAP_ANY),
]


def try_open(cam_idx: int):
    """Try multiple backends and return (cap, backend_name) on success."""
    for name, flag in BACKENDS:
        cap = cv2.VideoCapture(cam_idx, flag)
        if cap.isOpened():
            # Grab once to force real negotiation (some virtual cams lie about
            # isOpened() until you actually read).
            ok, _ = cap.read()
            if ok:
                return cap, name
            print(f"[{name}] opened cam {cam_idx} but first read() failed")
            cap.release()
        else:
            print(f"[{name}] cv2.VideoCapture({cam_idx}) could not open")
    return None, None


def record_one(cam_idx: int, seconds: float, fourcc_str: str, fps: float,
               out_path: Path, width: int, height: int) -> bool:
    print(f"\n===== camera {cam_idx} -> {out_path.name} =====")
    cap, backend = try_open(cam_idx)
    if cap is None:
        print(f"[skip] camera {cam_idx} did not open on any backend")
        return False
    print(f"[open] backend={backend}")

    # Force resolution if user asked — virtual cams (Camo, OBS) often need this
    if width > 0 and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[cam] size={w}x{h} reported_fps={cam_fps:.1f}")
    if w == 0 or h == 0:
        print("[ERR] camera reports 0x0 — negotiation failed, aborting this cam")
        cap.release()
        return False

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[ERR] VideoWriter failed ({fourcc_str} {w}x{h} @ {fps})")
        cap.release()
        return False
    print(f"[out] fourcc={fourcc_str} container_fps={fps}")

    t0 = time.time()
    last = t0
    frames = 0
    dropped = 0

    while True:
        ok, frame = cap.read()
        t_cap = time.time()
        if not ok:
            dropped += 1
            print(f"[drop] read() failed at t={t_cap - t0:.3f}s")
            if dropped > 20:
                break
            continue

        writer.write(frame)
        frames += 1
        dt = t_cap - last
        last = t_cap
        if frames % 10 == 0 or frames <= 5:
            print(f"[f{frames:4d}] dt={dt*1000:6.1f}ms t={t_cap - t0:5.2f}s")

        if t_cap - t0 >= seconds:
            break

    elapsed = time.time() - t0
    cap.release()
    writer.release()

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
    print(f"captured={frames}  dropped={dropped}  "
          f"elapsed={elapsed:.2f}s  eff_fps={frames/elapsed:.2f}  "
          f"size={size_mb:.2f}MB")

    vr = cv2.VideoCapture(str(out_path))
    if not vr.isOpened():
        print("[ERR] cannot reopen written file")
        return False
    rb_frames = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    rb_fps = vr.get(cv2.CAP_PROP_FPS)
    rb_w = int(vr.get(cv2.CAP_PROP_FRAME_WIDTH))
    rb_h = int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vr.release()
    print(f"readback: {rb_frames} frames, {rb_w}x{rb_h} @ {rb_fps:.2f} fps")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None,
                    help="Test only this camera index (else probes 0..max-1)")
    ap.add_argument("--max-cameras", type=int, default=4,
                    help="Probe camera indices 0..N-1 (default 4)")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fourcc", default="avc1", help="avc1 | mp4v | XVID | MJPG")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=0, help="Force capture width (0 = don't set)")
    ap.add_argument("--height", type=int, default=0, help="Force capture height (0 = don't set)")
    ap.add_argument("--out-dir", default=".", help="Output directory for test files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [args.camera] if args.camera is not None else list(range(args.max_cameras))

    results = {}
    for idx in indices:
        path = out_dir / f"test_video_cam{idx}.mp4"
        results[idx] = record_one(idx, args.seconds, args.fourcc, args.fps, path,
                                  args.width, args.height)

    print("\n===== summary =====")
    for idx, ok in results.items():
        print(f"  camera {idx}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
