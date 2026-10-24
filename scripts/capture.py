"""Capture one aligned RGB-D frame from a RealSense D435/D435i.

    python scripts/capture.py --out data/frames/$(date +%Y%m%d_%H%M%S)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim.camera import capture_realsense  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--average", type=int, default=10, help="frames median-averaged for depth")
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    frame = capture_realsense(args.width, args.height, average_frames=args.average, serial=args.serial)
    out = frame.save(args.out)
    valid = (frame.depth > 0).mean()
    print(f"saved {out}  model={frame.intrinsics.model} serial={frame.intrinsics.serial} valid_depth={valid:.1%}")


if __name__ == "__main__":
    main()
