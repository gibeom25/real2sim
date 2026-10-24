"""RealSense D435/D435i capture and RGB-D frame I/O.

A frame on disk is a directory with:
    rgb.png          color image (H, W, 3), uint8
    depth.npy        depth aligned to color, float32 meters, 0 = invalid
    intrinsics.json  {"fx", "fy", "cx", "cy", "width", "height", "serial", "model"}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    serial: str = ""
    model: str = ""

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class RGBDFrame:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 meters
    intrinsics: Intrinsics

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.rgb).save(out_dir / "rgb.png")
        np.save(out_dir / "depth.npy", self.depth.astype(np.float32))
        (out_dir / "intrinsics.json").write_text(json.dumps(asdict(self.intrinsics), indent=2))
        # 16-bit millimeter PNG for quick viewing in other tools
        Image.fromarray(np.clip(self.depth * 1000.0, 0, 65535).astype(np.uint16)).save(out_dir / "depth_mm.png")
        return out_dir

    @classmethod
    def load(cls, frame_dir: str | Path) -> "RGBDFrame":
        frame_dir = Path(frame_dir)
        rgb = np.array(Image.open(frame_dir / "rgb.png").convert("RGB"))
        depth = np.load(frame_dir / "depth.npy").astype(np.float32)
        intr = Intrinsics(**json.loads((frame_dir / "intrinsics.json").read_text()))
        assert depth.shape == rgb.shape[:2], f"depth {depth.shape} vs rgb {rgb.shape}"
        return cls(rgb, depth, intr)


def capture_realsense(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    warmup_frames: int = 30,
    average_frames: int = 10,
    serial: str | None = None,
) -> RGBDFrame:
    """Capture one RGB-D frame from a D435/D435i, depth aligned to the color camera.

    Depth is median-averaged over `average_frames` frames (ignoring zeros) to reduce noise.
    """
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
        depth_scale = depth_sensor.get_depth_scale()

        align = rs.align(rs.stream.color)
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter(1)

        for _ in range(warmup_frames):
            pipeline.wait_for_frames()

        depths, color = [], None
        for _ in range(average_frames):
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)
            depths.append(np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale)
            color = np.asanyarray(color_frame.get_data()).copy()

        stack = np.stack(depths, 0)
        stack[stack <= 0] = np.nan
        depth = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)

        color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intr = Intrinsics(
            fx=color_intr.fx,
            fy=color_intr.fy,
            cx=color_intr.ppx,
            cy=color_intr.ppy,
            width=color_intr.width,
            height=color_intr.height,
            serial=device.get_info(rs.camera_info.serial_number),
            model=device.get_info(rs.camera_info.name),
        )
    finally:
        pipeline.stop()
    return RGBDFrame(color, depth, intr)
