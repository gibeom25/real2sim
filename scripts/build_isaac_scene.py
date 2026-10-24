"""Build an Isaac Sim scene from a real2sim scene directory and optionally verify/visualize it.

Each reconstructed object becomes its own rigid-body USD (objects/obj_XXX/obj.usd: visual mesh with vertex
colors + convex-decomposition collider + mass), and scene.usd references them at their estimated poses on a
static table whose top surface is the world z=0 plane.

    # Export USD only (works in `real2sim` env without Isaac Sim, e.g., on Jetson Thor):
    python scripts/build_isaac_scene.py --scene data/scenes/xxx --export-only

    # Verify physics settling or open GUI in `env_isaaclab`:
    python scripts/build_isaac_scene.py --scene data/scenes/xxx --headless --settle 240
    python scripts/build_isaac_scene.py --scene data/scenes/xxx            # open the GUI and keep simulating
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim.usd_export import export_usd_scene  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True, help="scene directory produced by run_pipeline.py")
ap.add_argument("--density", type=float, default=500.0, help="object density kg/m^3")
ap.add_argument("--friction", type=float, default=0.8)
ap.add_argument("--table_margin", type=float, default=0.3, help="table extends this far beyond the objects, m")
ap.add_argument("--settle", type=int, default=0, help="physics steps to run and report object drift (0 = skip)")
ap.add_argument("--drop_height", type=float, default=0.0, help="raise objects before --settle (to test the drift check)")
ap.add_argument("--export-only", action="store_true", help="write obj.usd and scene.usd without launching Isaac Sim")
ap.add_argument("--skip-export", action="store_true", help="use existing scene.usd without regenerating it")

try:
    from isaaclab.app import AppLauncher

    _HAS_ISAACLAB = True
    AppLauncher.add_app_launcher_args(ap)
except ImportError:
    _HAS_ISAACLAB = False
    ap.add_argument("--headless", action="store_true", help="run without GUI")

args = ap.parse_args()
scene_dir = Path(args.scene).resolve()
scene = json.loads((scene_dir / "scene.json").read_text())

if args.export_only or not _HAS_ISAACLAB:
    if not _HAS_ISAACLAB and not args.export_only:
        print("[real2sim] isaaclab not installed in current env; running --export-only mode")
    print(f"[real2sim] exporting USD scene in {scene_dir}")
    stage_path = export_usd_scene(
        scene_dir, density=args.density, friction=args.friction, table_margin=args.table_margin
    )
    print(f"[real2sim] wrote {stage_path}")
    sys.exit(0)

app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402


def settle(stage_path: Path, steps: int):
    """Run physics and report how far each object moves: large drift = bad pose or collision geometry."""
    from isaaclab.sim import SimulationCfg, SimulationContext

    omni.usd.get_context().open_stage(str(stage_path))
    stage = omni.usd.get_context().get_stage()
    if args.drop_height > 0:  # sanity check of the drift measurement itself
        for o in scene["objects"]:
            op = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/Objects/{o['id']}")).GetOrderedXformOps()[0]
            op.Set(op.Get() + Gf.Vec3d(0, 0, args.drop_height))
    sim = SimulationContext(SimulationCfg(dt=1 / 120))
    sim.reset()
    # GPU physics state is only visible through the tensor API (USD xforms / physx queries are stale)
    view = sim.physics_sim_view.create_rigid_body_view("/World/Objects/obj_*")
    ids = [p.rsplit("/", 1)[-1] for p in view.prim_paths]

    def positions():
        return dict(zip(ids, view.get_transforms()[:, :3].cpu().numpy()))

    start = positions()
    for _ in range(steps):
        sim.step(render=not args.headless)
        if not app.is_running():
            break
    end = positions()
    report = {k: float(np.linalg.norm(end[k] - start[k])) for k in start}
    for k, d in report.items():
        print(f"  {k}: drift {d * 1000:.1f} mm" + ("   <-- check pose / mesh" if d > 0.02 else ""))
    (scene_dir / "settle_report.json").write_text(json.dumps({"steps": steps, "drift_m": report}, indent=2))


stage_path = scene_dir / "scene.usd"
if not args.skip_export or not stage_path.exists():
    print(f"[real2sim] building Isaac scene from {scene_dir}")
    stage_path = export_usd_scene(
        scene_dir, density=args.density, friction=args.friction, table_margin=args.table_margin
    )
    print(f"[real2sim] wrote {stage_path}")
else:
    print(f"[real2sim] using existing {stage_path}")

if args.settle > 0:
    settle(stage_path, args.settle)
elif not args.headless:
    omni.usd.get_context().open_stage(str(stage_path))

if not args.headless:
    while app.is_running():
        app.update()
# SimulationApp.close() can block indefinitely on shutdown in headless mode; everything is saved by now
sys.stdout.flush()
os._exit(0)
