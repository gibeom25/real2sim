"""RGB-D frame -> segmented, reconstructed, posed objects -> scene directory for Isaac Sim.

Output layout (scene_dir):
    frame/                    the input RGB-D frame
    segmentation.png          mask overlay
    masks/obj_XXX.png         per-object masks
    objects/obj_XXX/mesh.ply  mesh in its object frame (meters, vertex colors)
    scene_cam.glb             all meshes + observed cloud (red) in camera frame, for checking alignment
    scene.json                camera, support plane, per-object 6-DoF poses
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from . import geometry as geo
from .camera import RGBDFrame
from .reconstruction import Sam3DReconstructor, clean_mesh
from .segmentation import Detection, Sam3Segmenter, overlay


@dataclass
class PipelineConfig:
    prompts: list[str] = field(default_factory=lambda: ["cup", "mug", "bottle", "can", "box", "bowl", "fruit", "toy", "block", "tool"])
    confidence_threshold: float = 0.5
    min_area_frac: float = 0.0005
    max_area_frac: float = 0.25
    max_depth: float = 2.0  # meters; farther depth is treated as invalid
    min_object_points: int = 200
    plane_distance: float = 0.008
    on_plane_reject_frac: float = 0.8  # drop "objects" whose points mostly lie on the table plane
    icp: bool = True
    icp_min_fitness: float = 0.3  # keep SAM 3D pose if ICP fits worse than this
    snap_to_plane: bool = True
    snap_max_gap: float = 0.03
    max_faces: int = 20000
    sam3d_layout_postprocess: bool = True
    seed: int = 42
    export_usd: bool = True
    density: float = 500.0
    friction: float = 0.8
    table_margin: float = 0.3


def _pose_dict(T: np.ndarray) -> dict:
    return {
        "position": [float(x) for x in T[:3, 3]],
        "quat_wxyz": geo.rotation_to_quat_wxyz(T[:3, :3]),
        "rpy_deg": geo.rotation_to_rpy_deg(T[:3, :3]),
        "matrix": T.tolist(),
    }


class Real2SimPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.segmenter = Sam3Segmenter(confidence_threshold=cfg.confidence_threshold)
        self.reconstructor = Sam3DReconstructor(layout_postprocess=cfg.sam3d_layout_postprocess)

    def segment(self, frame: RGBDFrame) -> list[Detection]:
        cfg = self.cfg
        dets = self.segmenter(frame.rgb, cfg.prompts, cfg.min_area_frac, cfg.max_area_frac)
        print(f"[segment] {len(dets)} candidate instances")
        return dets

    def run(self, frame: RGBDFrame, scene_dir: str | Path) -> dict:
        cfg = self.cfg
        scene_dir = Path(scene_dir)
        (scene_dir / "masks").mkdir(parents=True, exist_ok=True)
        frame.save(scene_dir / "frame")

        depth = frame.depth.copy()
        depth[depth > cfg.max_depth] = 0
        pointmap = geo.backproject(depth, frame.intrinsics)

        # --- 1. support plane & world frame -------------------------------------------------
        valid = pointmap[np.isfinite(pointmap).all(-1)]
        sample = valid[np.random.default_rng(0).choice(len(valid), min(len(valid), 200000), replace=False)]
        plane, _ = geo.fit_support_plane(sample, cfg.plane_distance)
        print(f"[plane] normal={plane[:3].round(3)} d={plane[3]:.3f}")

        # --- 2. segmentation ----------------------------------------------------------------
        dets = self.segment(frame)
        Image.fromarray(overlay(frame.rgb, dets)).save(scene_dir / "segmentation.png")

        objects, cam_meshes, obs_clouds = [], [], []
        for det in dets:
            obs = geo.masked_points(pointmap, det.mask)
            if len(obs) < cfg.min_object_points:
                print(f"[skip] {det.label}: only {len(obs)} depth points")
                continue
            dist = np.abs(obs @ plane[:3] + plane[3])
            if (dist < 1.5 * cfg.plane_distance).mean() > cfg.on_plane_reject_frac:
                print(f"[skip] {det.label}: lies on the support plane (flat / table region)")
                continue
            obs = geo.clean_points(obs)
            oid = f"obj_{len(objects):03d}"
            Image.fromarray(det.mask.astype(np.uint8) * 255).save(scene_dir / "masks" / f"{oid}.png")

            # --- 3. shape + initial layout (SAM 3D) -------------------------------------------
            print(f"[sam3d] {oid} ({det.label}, score {det.score:.2f})")
            rec = self.reconstructor(frame.rgb, det.mask, pointmap, seed=cfg.seed)

            # --- 4. 6-DoF refinement against the observed depth (ICP) -------------------------
            T_cam_local = rec.T_cam_local
            fitness = rmse = None
            if cfg.icp:
                src = geo.visible_surface(rec.vertices_cam, rec.faces, frame.intrinsics)
                if len(src) > 50:
                    T_icp, fitness, rmse = geo.icp_refine(src, obs)
                    if fitness >= cfg.icp_min_fitness:
                        T_cam_local = T_icp @ T_cam_local
                    print(f"        icp fitness={fitness:.2f} rmse={rmse * 1000:.1f}mm"
                          f"{'' if fitness >= cfg.icp_min_fitness else ' (rejected)'}")

            # object frame = canonical SAM 3D axes, origin at the mesh bounding-box center
            mesh = trimesh.Trimesh(rec.vertices_local, rec.faces, vertex_colors=rec.vertex_colors, process=False)
            mesh = clean_mesh(mesh, cfg.max_faces)
            center = mesh.bounds.mean(0)
            mesh.apply_translation(-center)
            T_local_obj = np.eye(4)
            T_local_obj[:3, 3] = center
            T_cam_obj = T_cam_local @ T_local_obj

            odir = scene_dir / "objects" / oid
            odir.mkdir(parents=True, exist_ok=True)
            mesh.export(odir / "mesh.ply")
            mesh.export(odir / "mesh.obj")

            cam_mesh = mesh.copy()
            cam_mesh.apply_transform(T_cam_obj)
            cam_meshes.append(cam_mesh)
            obs_clouds.append(obs)

            objects.append({
                "id": oid,
                "label": det.label,
                "score": det.score,
                "mesh": f"objects/{oid}/mesh.ply",
                "mesh_obj": f"objects/{oid}/mesh.obj",
                "mask": f"masks/{oid}.png",
                "scale": rec.scale,
                "extents": [float(x) for x in mesh.extents],
                "icp_fitness": fitness,
                "icp_rmse": rmse,
                "pose_cam": _pose_dict(T_cam_obj),
                "_T_cam_obj": T_cam_obj,
                "_mesh": mesh,
            })

        # --- 5. world frame: z = table normal, origin under the objects ----------------------
        anchor = np.mean([o["_T_cam_obj"][:3, 3] for o in objects], 0) if objects else valid.mean(0)
        T_cam_world = geo.world_from_plane(plane, anchor)
        T_world_cam = np.linalg.inv(T_cam_world)
        for o in objects:
            T_world_obj = T_world_cam @ o.pop("_T_cam_obj")
            mesh = o.pop("_mesh")
            if cfg.snap_to_plane:
                zmin = geo.transform_points(T_world_obj, np.asarray(mesh.vertices))[:, 2].min()
                if zmin < cfg.snap_max_gap:  # penetrating, or hovering just above
                    T_world_obj[2, 3] -= zmin
            o["pose_world"] = _pose_dict(T_world_obj)

        scene = {
            "frame": "frame",
            "conventions": {
                "camera": "OpenCV (x right, y down, z forward)",
                "world": "z up = support-plane normal, origin on the plane under the objects, meters",
                "quat": "wxyz",
                "object_frame": "SAM 3D canonical axes, origin at mesh bbox center",
            },
            "camera": {
                "intrinsics": frame.intrinsics.__dict__,
                "T_world_cam": T_world_cam.tolist(),
            },
            "support_plane": {"plane_cam": plane.tolist(), "plane_world": [0.0, 0.0, 1.0, 0.0]},
            "objects": objects,
        }
        (scene_dir / "scene.json").write_text(json.dumps(scene, indent=2))

        if cam_meshes:
            obs_all = np.concatenate(obs_clouds)
            debug = trimesh.util.concatenate(cam_meshes)
            trimesh.Scene([debug, trimesh.PointCloud(obs_all, colors=[255, 0, 0, 255])]).export(scene_dir / "scene_cam.glb")
        print(f"[done] {len(objects)} objects -> {scene_dir / 'scene.json'}")
        if cfg.export_usd:
            try:
                from .usd_export import export_usd_scene

                stage_path = export_usd_scene(
                    scene_dir, density=cfg.density, friction=cfg.friction, table_margin=cfg.table_margin
                )
                print(f"[usd]  wrote {stage_path}")
            except ImportError:
                print("[usd]  usd-core (pxr) not installed; skipping USD export")
        return scene
