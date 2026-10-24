"""Run segmentation -> reconstruction -> 6-DoF pose on one RGB-D frame and export a scene directory.

    # from a saved frame
    python scripts/run_pipeline.py --frame data/frames/xxx --out data/scenes/xxx
    # capture live from the RealSense first
    python scripts/run_pipeline.py --capture --out data/scenes/xxx --prompts cup bottle box
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim.camera import RGBDFrame, capture_realsense  # noqa: E402
from real2sim.pipeline import PipelineConfig, Real2SimPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame", help="directory with rgb.png / depth.npy / intrinsics.json")
    src.add_argument("--capture", action="store_true", help="capture from the RealSense")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    ap.add_argument("--prompts", nargs="+", help="override SAM 3 text prompts")
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text()) or {}
    known = {f.name for f in fields(PipelineConfig)}
    cfg = PipelineConfig(**{k: v for k, v in raw.items() if k in known})
    if args.prompts:
        cfg.prompts = args.prompts

    frame = capture_realsense() if args.capture else RGBDFrame.load(args.frame)
    Real2SimPipeline(cfg).run(frame, args.out)


if __name__ == "__main__":
    main()
