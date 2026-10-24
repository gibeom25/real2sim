"""Per-object 3D shape + layout with SAM 3D Objects, conditioned on the metric RealSense pointmap."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh

SAM3D_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "sam-3d-objects"

# glb export rotates SAM 3D's z-up object space to y-up; this undoes it (see layout_post_optimization_utils.get_mesh)
_GLB_TO_LOCAL = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


@dataclass
class ObjectReconstruction:
    vertices_cam: np.ndarray  # (V, 3) mesh in OpenCV camera frame, meters
    faces: np.ndarray  # (F, 3)
    vertex_colors: np.ndarray | None  # (V, 4) uint8
    T_cam_local: np.ndarray  # 4x4 rigid: SAM 3D canonical (z-up) object frame -> camera
    scale: float  # metric scale applied to the canonical mesh
    vertices_local: np.ndarray  # (V, 3) canonical frame, metric scale


class Sam3DReconstructor:
    def __init__(self, tag: str = "hf", compile: bool = False, layout_postprocess: bool = True):
        sys.path.insert(0, str(SAM3D_ROOT / "notebook"))
        from inference import Inference

        config = SAM3D_ROOT / "checkpoints" / tag / "pipeline.yaml"
        if not config.exists():
            raise FileNotFoundError(f"{config} missing - run scripts/download_checkpoints.sh first")
        self.pipeline = Inference(str(config), compile=compile)._pipeline
        self.layout_postprocess = layout_postprocess

        from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
        from pytorch3d.transforms import Transform3d

        rot = camera_to_pytorch3d_camera(device="cuda").rotation
        self._cv_to_p3d = Transform3d().rotate(rot).to("cuda")

    def _pointmap_p3d(self, pointmap_cv: np.ndarray) -> torch.Tensor:
        pts = torch.from_numpy(pointmap_cv.astype(np.float32)).cuda()
        h, w, _ = pts.shape
        return self._cv_to_p3d.transform_points(pts.reshape(1, -1, 3)).reshape(h, w, 3)

    def __call__(self, rgb: np.ndarray, mask: np.ndarray, pointmap_cv: np.ndarray, seed: int = 42) -> ObjectReconstruction:
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

        rgba = np.concatenate([rgb[..., :3], (mask.astype(np.uint8) * 255)[..., None]], -1)
        out = self.pipeline.run(
            rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=False,
            with_layout_postprocess=self.layout_postprocess,
            use_vertex_color=True,
            pointmap=self._pointmap_p3d(pointmap_cv),
        )
        glb: trimesh.Trimesh = out["glb"]
        if isinstance(glb, trimesh.Scene):
            glb = glb.dump(concatenate=True)

        scale = out["scale"].reshape(1, 3).float()
        R = quaternion_to_matrix(out["rotation"].reshape(1, 4).float())
        t = out["translation"].reshape(1, 3).float()
        local_to_p3d = compose_transform(scale, R, t)
        local_to_cv = local_to_p3d.compose(self._cv_to_p3d.inverse())

        v_local = np.asarray(glb.vertices, dtype=np.float64) @ _GLB_TO_LOCAL.T
        v_cam = local_to_cv.transform_points(torch.from_numpy(v_local).float().cuda()[None])[0].cpu().numpy()

        # decompose the similarity transform local->cam into rigid T and scalar scale
        basis = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device="cuda")
        b = local_to_cv.transform_points(basis[None])[0].cpu().numpy().astype(np.float64)
        A = (b[1:] - b[0]).T  # columns = images of unit axes
        s = float(np.cbrt(abs(np.linalg.det(A))))
        U, _, Vt = np.linalg.svd(A / s)
        Rc = U @ Vt
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rc, b[0]

        colors = None
        if glb.visual is not None and getattr(glb.visual, "kind", None) == "vertex":
            colors = np.asarray(glb.visual.vertex_colors)
        return ObjectReconstruction(v_cam, np.asarray(glb.faces), colors, T, s, v_local * s)


def clean_mesh(mesh: trimesh.Trimesh, max_faces: int = 20000) -> trimesh.Trimesh:
    """Largest connected component + decimation so physics cooking stays fast."""
    mesh.merge_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.area)
    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        except Exception as e:  # fast_simplification missing - keep full mesh
            print(f"[clean_mesh] decimation skipped: {e}")
    mesh.remove_unreferenced_vertices()
    return mesh
