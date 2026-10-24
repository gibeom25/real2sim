"""Isaac Lab bridge: turn a real2sim scene directory into per-object RigidObjectCfg entries.

Run `scripts/build_isaac_scene.py` once first so objects/obj_XXX/obj.usd exist. Then, in an Isaac Lab
InteractiveSceneCfg / ManagerBasedRLEnvCfg:

    from real2sim.isaaclab_scene import load_real2sim_objects, real2sim_table_cfg

    @configclass
    class PickPlaceSceneCfg(InteractiveSceneCfg):
        table = real2sim_table_cfg("data/scenes/xxx")
        ...

    scene_cfg = PickPlaceSceneCfg(num_envs=1024, env_spacing=2.0)
    for name, cfg in load_real2sim_objects("data/scenes/xxx").items():
        setattr(scene_cfg, name, cfg)

Poses are in the real2sim world frame (z up, table top at z=0); offset them with `origin` to place the
table in front of your robot.
"""

from __future__ import annotations

import json
from pathlib import Path


def _scene(scene_dir: str | Path) -> tuple[Path, dict]:
    scene_dir = Path(scene_dir).resolve()
    return scene_dir, json.loads((scene_dir / "scene.json").read_text())


def load_real2sim_objects(
    scene_dir: str | Path,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass: float | None = None,
    prim_root: str = "{ENV_REGEX_NS}",
) -> dict:
    """Returns {object_id: RigidObjectCfg}, each spawned from its own USD at the estimated pose."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    scene_dir, scene = _scene(scene_dir)
    cfgs = {}
    for obj in scene["objects"]:
        usd = scene_dir / "objects" / obj["id"] / "obj.usd"
        if not usd.exists():
            raise FileNotFoundError(f"{usd} missing - run scripts/build_isaac_scene.py first")
        pos = tuple(p + o for p, o in zip(obj["pose_world"]["position"], origin))
        cfgs[obj["id"]] = RigidObjectCfg(
            prim_path=f"{prim_root}/{obj['id']}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass) if mass is not None else None,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=tuple(obj["pose_world"]["quat_wxyz"])),
        )
    return cfgs


def real2sim_table_cfg(
    scene_dir: str | Path,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    margin: float = 0.3,
    thickness: float = 0.05,
    prim_root: str = "{ENV_REGEX_NS}",
):
    """Static box whose top face is the reconstructed support plane (z=0 + origin)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    _, scene = _scene(scene_dir)
    xs = [o["pose_world"]["position"][0] for o in scene["objects"]] or [0.0]
    ys = [o["pose_world"]["position"][1] for o in scene["objects"]] or [0.0]
    r = max((max(o["extents"]) for o in scene["objects"]), default=0.2)
    size = (max(xs) - min(xs) + 2 * (r + margin), max(ys) - min(ys) + 2 * (r + margin), thickness)
    center = ((max(xs) + min(xs)) / 2 + origin[0], (max(ys) + min(ys)) / 2 + origin[1], origin[2] - thickness / 2)
    return AssetBaseCfg(
        prim_path=f"{prim_root}/Table",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.45)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=center),
    )
