"""Render a ground-truth tabletop RGB-D frame in Isaac Sim (stand-in for the RealSense when it isn't connected).

YCB objects on a table, seen by a camera with D435-like color intrinsics. Writes a frame directory in the same
format as scripts/capture.py plus gt.json (camera pose) and gt_points/<name>.npy (object surface points in the
OpenCV camera frame) so tests/eval_against_gt.py can score the reconstructed scene.

    python tests/render_isaac_frame.py --out data/frames/isaac_ycb --headless --enable_cameras
"""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

W, H, FX, FY, CX, CY = 1280, 720, 910.0, 910.0, 640.0, 360.0  # typical D435 color @1280x720
YCB = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"
OBJECTS = {  # name: (usd, xy on table, yaw deg)
    "cracker_box": ("003_cracker_box.usd", (-0.14, 0.06), 20.0),
    "sugar_box": ("004_sugar_box.usd", (0.02, 0.14), -35.0),
    "soup_can": ("005_tomato_soup_can.usd", (0.10, -0.02), 0.0),
    "mustard_bottle": ("006_mustard_bottle.usd", (-0.02, -0.10), 60.0),
}

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60))
stage = sim.stage
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))
sim_utils.DistantLightCfg(intensity=2000.0, angle=0.5).func("/World/Sun", sim_utils.DistantLightCfg(intensity=2000.0))
table_cfg = sim_utils.CuboidCfg(
    size=(1.2, 0.9, 0.04),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.72, 0.68), roughness=0.8),
)
table_cfg.func("/World/Table", table_cfg, translation=(0.0, 0.0, -0.02))

for name, (usd, (x, y), yaw) in OBJECTS.items():
    cfg = sim_utils.UsdFileCfg(usd_path=f"{YCB}/{usd}")
    q = (np.cos(np.deg2rad(yaw) / 2), 0.0, 0.0, np.sin(np.deg2rad(yaw) / 2))
    cfg.func(f"/World/Objects/{name}", cfg, translation=(x, y, 0.0), orientation=q)


# lift each object so its lowest point touches the table (YCB origins aren't at the bottom)
def world_bbox(path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()


for name in OBJECTS:
    zmin = world_bbox(f"/World/Objects/{name}").GetMin()[2]
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/Objects/{name}"))
    op = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate][0]
    t = op.Get()
    op.Set(type(t)(t[0], t[1], t[2] - zmin + 0.001))

camera = Camera(CameraCfg(
    prim_path="/World/Camera",
    width=W,
    height=H,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        [FX, 0, CX, 0, FY, CY, 0, 0, 1], W, H, clipping_range=(0.05, 5.0)
    ),
))
sim.reset()
EYE, TARGET = np.array([0.05, -0.45, 0.42]), np.array([0.0, 0.0, 0.03])
camera.set_world_poses_from_view(torch.tensor([EYE], device=sim.device, dtype=torch.float32),
                                 torch.tensor([TARGET], device=sim.device, dtype=torch.float32))
for _ in range(30):  # render only (no physics step), so the USD poses stay the ground truth
    sim.render()
    camera.update(sim.get_physics_dt())

rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float32)
depth[~np.isfinite(depth)] = 0
depth[depth > 4.0] = 0
depth = depth + np.random.default_rng(0).normal(0, 0.002, depth.shape).astype(np.float32) * (depth > 0)  # ~D435 noise

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
Image.fromarray(rgb).save(out / "rgb.png")
np.save(out / "depth.npy", depth)
(out / "intrinsics.json").write_text(json.dumps(
    dict(fx=FX, fy=FY, cx=CX, cy=CY, width=W, height=H, serial="isaac", model="Isaac D435-like"), indent=2))

# camera pose (OpenCV axes) from the same look-at: z forward, x right, y down, world up = +z
z = (TARGET - EYE) / np.linalg.norm(TARGET - EYE)
x = np.cross(z, [0.0, 0.0, 1.0])
x /= np.linalg.norm(x)
T_world_cam = np.eye(4)
T_world_cam[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
T_world_cam[:3, 3] = EYE
T_cam_world = np.linalg.inv(T_world_cam)

# GT surface points per object (world -> camera). These come from the USD poses set above; physics only lets
# objects drop the 1 mm clearance during the 60 steps, which is below the accuracy being measured.

(out / "gt_points").mkdir(exist_ok=True)
gt = {"T_world_cam": T_world_cam.tolist(), "objects": {}}
xf_cache = UsdGeom.XformCache()
for name in OBJECTS:
    pts_all = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/World/Objects/{name}")):
        if prim.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(prim).ComputePurpose() != UsdGeom.Tokens.guide:
            pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            if len(pts) == 0:
                continue
            M = np.asarray(xf_cache.GetLocalToWorldTransform(prim), dtype=np.float64).T  # row-vector -> column
            pts_all.append(pts @ M[:3, :3].T + M[:3, 3])
    pts_w = np.concatenate(pts_all)
    pts_c = pts_w @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    np.save(out / "gt_points" / f"{name}.npy", pts_c.astype(np.float32))
    ext = pts_w.max(0) - pts_w.min(0)
    gt["objects"][name] = {"center_cam": pts_c.mean(0).tolist(), "extents_world_aabb": ext.tolist()}
# consistency check: GT points must land on the rendered depth where visible
errs = []
for name in OBJECTS:
    p = np.load(out / "gt_points" / f"{name}.npy")
    u = np.round(p[:, 0] * FX / p[:, 2] + CX).astype(int)
    v = np.round(p[:, 1] * FY / p[:, 2] + CY).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d = depth[v[ok], u[ok]]
    diff = np.abs(d - p[ok, 2])
    errs.append(np.median(diff[diff < np.percentile(diff, 30)]))  # front-facing points only
gt["reprojection_check_m"] = float(np.max(errs))
print(f"[render] GT reprojection check: {np.max(errs) * 1000:.1f} mm (should be ~noise level)")
(out / "gt.json").write_text(json.dumps(gt, indent=2))
print(f"[render] saved {out}  valid depth {(depth > 0).mean():.1%}")
sys.stdout.flush()
os._exit(0)
