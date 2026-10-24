"""Synthetic end-to-end check of everything except the neural networks.

Renders a depth image of known meshes on a tilted table, feeds ground-truth masks and a deliberately
perturbed "SAM 3D" result (wrong pose by a few cm / degrees) into Real2SimPipeline, and checks that the
plane fit + ICP + world-frame export recover the true object poses.

    python tests/test_synthetic.py --out data/scenes/synthetic
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim import geometry as geo  # noqa: E402
from real2sim.camera import Intrinsics, RGBDFrame  # noqa: E402
from real2sim.pipeline import PipelineConfig, Real2SimPipeline  # noqa: E402
from real2sim.reconstruction import ObjectReconstruction  # noqa: E402
from real2sim.segmentation import Detection  # noqa: E402

W, H = 640, 480
INTR = Intrinsics(fx=605.0, fy=605.0, cx=322.0, cy=241.0, width=W, height=H, model="synthetic")


def make_scene():
    """Objects in a table frame (z up, table top at z=0), and the camera looking down at ~45 deg."""
    rng = np.random.default_rng(1)
    objs = [
        ("box", trimesh.creation.box([0.10, 0.06, 0.08])),
        ("can", trimesh.creation.cylinder(radius=0.033, height=0.12, sections=48)),
        ("block", trimesh.creation.box([0.05, 0.12, 0.04])),
    ]
    T_world_objs = []
    for (name, m), xy in zip(objs, [(-0.12, 0.02), (0.05, 0.08), (0.10, -0.08)]):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("z", rng.uniform(-90, 90), degrees=True).as_matrix()
        T[:3, 3] = [xy[0], xy[1], m.extents[2] / 2]
        T_world_objs.append(T)
    T_world_cam = np.eye(4)  # camera 0.6 m away, pitched down 50 deg, OpenCV axes
    R = Rotation.from_euler("x", -(90 + 50), degrees=True).as_matrix()
    T_world_cam[:3, :3] = R
    T_world_cam[:3, 3] = -R @ np.array([0, 0, 0.6]) + np.array([0.0, 0.0, 0.0])
    return objs, T_world_objs, T_world_cam


def render(objs, T_world_objs, T_world_cam):
    T_cam_world = np.linalg.inv(T_world_cam)
    table = trimesh.creation.box([1.5, 1.5, 0.02])
    table.apply_translation([0, 0, -0.01])
    meshes = [table] + [m.copy().apply_transform(T) for (_, m), T in zip(objs, T_world_objs)]
    for m in meshes:
        m.apply_transform(T_cam_world)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([(u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy, np.ones_like(u, float)], -1).reshape(-1, 3)
    depth = np.full(H * W, np.inf)
    label = np.full(H * W, -1)
    for i, m in enumerate(meshes):
        locs, ray_idx, _ = m.ray.intersects_location(np.zeros_like(dirs), dirs, multiple_hits=False)
        z = locs[:, 2]
        closer = z < depth[ray_idx]
        depth[ray_idx[closer]] = z[closer]
        label[ray_idx[closer]] = i - 1
    depth[~np.isfinite(depth)] = 0
    depth = depth + np.random.default_rng(0).normal(0, 0.001, depth.shape) * (depth > 0)  # 1 mm noise
    rgb = np.full((H, W, 3), 128, np.uint8)
    return depth.reshape(H, W).astype(np.float32), label.reshape(H, W), rgb


class FakeSegmenter:
    def __init__(self, labels, names):
        self.labels, self.names = labels, names

    def __call__(self, rgb, prompts, *a):
        dets = []
        for i, n in enumerate(self.names):
            m = self.labels == i
            ys, xs = np.nonzero(m)
            dets.append(Detection(m, np.array([xs.min(), ys.min(), xs.max(), ys.max()], float), 0.9, n))
        return dets


class FakeReconstructor:
    """Returns the true mesh with a perturbed pose, mimicking SAM 3D layout error."""

    def __init__(self, objs, T_cam_objs):
        self.items, self.calls, self.init_poses = list(zip(objs, T_cam_objs)), 0, []

    def __call__(self, rgb, mask, pointmap, seed=0):
        (_, mesh), T_true = self.items[self.calls]
        self.calls += 1
        rng = np.random.default_rng(self.calls)
        dT = np.eye(4)
        dT[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) / np.sqrt(3) * np.deg2rad(6)).as_matrix()
        dT[:3, 3] = rng.normal(size=3) / np.sqrt(3) * 0.01
        T = T_true @ dT
        self.init_poses.append(T)
        v_local = np.asarray(mesh.vertices)
        return ObjectReconstruction(geo.transform_points(T, v_local), np.asarray(mesh.faces), None, T, 1.0, v_local)


def surface_error(T_est, T_true, mesh, n=2000):
    """Mean surface distance between the mesh under both poses (handles symmetric shapes)."""
    pts = trimesh.sample.sample_surface(mesh, n, seed=0)[0]
    a, b = geo.transform_points(T_est, pts), geo.transform_points(T_true, pts)
    from scipy.spatial import cKDTree

    return cKDTree(b).query(a)[0].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data/scenes/synthetic"))
    args = ap.parse_args()

    objs, T_world_objs, T_world_cam = make_scene()
    depth, labels, rgb = render(objs, T_world_objs, T_world_cam)
    T_cam_objs = [np.linalg.inv(T_world_cam) @ T for T in T_world_objs]

    pipe = Real2SimPipeline.__new__(Real2SimPipeline)
    pipe.cfg = PipelineConfig(snap_to_plane=True)
    pipe.segmenter = FakeSegmenter(labels, [n for n, _ in objs])
    pipe.reconstructor = FakeReconstructor(objs, T_cam_objs)
    scene = pipe.run(RGBDFrame(rgb, depth, INTR), args.out)

    # compare in the camera frame (the world frame's origin/yaw is a free choice of the pipeline)
    ok = True
    print("\nobject   | ICP fitness | init surface err (mm) | final surface err (mm) | bottom z (mm)")
    for (name, mesh), o, T_true, T_init in zip(objs, scene["objects"], T_cam_objs, pipe.reconstructor.init_poses):
        T_est = np.asarray(o["pose_cam"]["matrix"])
        init = surface_error(T_init, T_true, mesh) * 1000
        final = surface_error(T_est, T_true, mesh) * 1000
        T_w = np.asarray(o["pose_world"]["matrix"])
        zmin = geo.transform_points(T_w, np.asarray(mesh.vertices))[:, 2].min() * 1000
        print(f"{name:8s} | {o['icp_fitness']:.2f}        | {init:6.2f}                | {final:6.2f}                 | {zmin:6.2f}")
        ok &= final < 3.0 and abs(zmin) < 3.0

    # the table normal expressed in the world frame must be +z, and the GT objects must be upright there
    T_world_cam_est = np.asarray(scene["camera"]["T_world_cam"])
    n_world = T_world_cam_est[:3, :3] @ np.asarray(scene["support_plane"]["plane_cam"][:3])
    print("table normal in world frame:", np.round(n_world, 4))
    ok &= np.allclose(n_world, [0, 0, 1], atol=0.01)
    print("RESULT:", "PASS" if ok else "FAIL")
    json.dump({"pass": bool(ok)}, open(Path(args.out) / "synthetic_result.json", "w"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
