"""Point-cloud utilities: back-projection, support-plane fitting, ICP, rigid transforms.

Camera frame convention everywhere in this package is OpenCV: x right, y down, z forward.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from .camera import Intrinsics


def backproject(depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Depth (H, W) meters -> pointmap (H, W, 3) in OpenCV camera frame, NaN where depth is invalid."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    pts = np.stack([x, y, z], -1)
    pts[z <= 0] = np.nan
    return pts


def masked_points(pointmap: np.ndarray, mask: np.ndarray, erode_px: int = 2) -> np.ndarray:
    """Valid 3D points under a mask. Erodes the mask first to drop flying pixels at depth edges."""
    if erode_px > 0:
        import cv2

        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel) > 0
    pts = pointmap[mask]
    return pts[np.isfinite(pts).all(-1)]


def to_o3d(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def clean_points(points: np.ndarray, voxel: float = 0.003, nb: int = 20, std: float = 2.0) -> np.ndarray:
    pcd = to_o3d(points).voxel_down_sample(voxel)
    if len(pcd.points) > nb:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    return np.asarray(pcd.points)


def fit_support_plane(
    points: np.ndarray, distance_threshold: float = 0.008, iterations: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC the dominant plane (table / floor).

    Returns (plane [a, b, c, d] with normal pointing toward the camera, inlier mask over `points`).
    """
    pcd = to_o3d(points)
    plane, inliers = pcd.segment_plane(distance_threshold, ransac_n=3, num_iterations=iterations)
    plane = np.asarray(plane, dtype=np.float64)
    plane /= np.linalg.norm(plane[:3])
    if plane[3] < 0:  # camera origin must be on the positive side: n.0 + d > 0
        plane = -plane
    mask = np.zeros(len(points), bool)
    mask[inliers] = True
    return plane, mask


def world_from_plane(plane: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """T_cam_world: world z = plane normal (up), origin = `anchor` projected onto the plane,
    world x = camera x-axis projected onto the plane."""
    n = plane[:3]
    origin = anchor - (n @ anchor + plane[3]) * n
    x = np.array([1.0, 0.0, 0.0]) - n[0] * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, n], 1)
    T[:3, 3] = origin
    return T


def icp_refine(
    source: np.ndarray,
    target: np.ndarray,
    init: np.ndarray = np.eye(4),
    max_corr: tuple[float, ...] = (0.02, 0.01, 0.005),
) -> tuple[np.ndarray, float, float]:
    """Coarse-to-fine point-to-plane ICP. Returns (T applied to source, fitness, inlier rmse).

    `source` should be the visible part of the model (the observation is a single view),
    `target` is the observed point cloud.
    """
    src, tgt = to_o3d(source), to_o3d(target)
    for pcd in (src, tgt):
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    T = init.copy()
    result = None
    for dist in max_corr:
        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            dist,
            T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        T = result.transformation
    return np.asarray(T), float(result.fitness), float(result.inlier_rmse)


def visible_surface(vertices: np.ndarray, faces: np.ndarray, intr: Intrinsics, n_samples: int = 30000) -> np.ndarray:
    """Points on the mesh (in camera frame) that are visible from the camera, via a z-buffer
    over surface samples. Used as the ICP source so a full mesh is not pulled toward a partial view."""
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    pts = pts[pts[:, 2] > 1e-3]
    u = np.round(pts[:, 0] * intr.fx / pts[:, 2] + intr.cx).astype(int)
    v = np.round(pts[:, 1] * intr.fy / pts[:, 2] + intr.cy).astype(int)
    ok = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    pts, u, v = pts[ok], u[ok], v[ok]
    # downsampled z-buffer (4 px cells) so sparse samples still occlude each other
    cell = 4
    key = (v // cell) * (intr.width // cell + 1) + (u // cell)
    zbuf = np.full(key.max() + 1, np.inf)
    np.minimum.at(zbuf, key, pts[:, 2])
    return pts[pts[:, 2] <= zbuf[key] + 0.004]


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def rotation_to_quat_wxyz(R: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def rotation_to_rpy_deg(R: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    return [float(a) for a in Rotation.from_matrix(R).as_euler("xyz", degrees=True)]
