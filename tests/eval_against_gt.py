"""Score a reconstructed scene against the ground truth written by tests/render_isaac_frame.py.

All comparisons are in the OpenCV camera frame, so the pipeline's choice of world origin doesn't matter.

    python tests/eval_against_gt.py --frame data/frames/isaac_ycb --scene data/scenes/isaac_ycb
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def obb_extents(points: np.ndarray) -> np.ndarray:
    return np.sort(trimesh.PointCloud(points).bounding_box_oriented.primitive.extents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    frame, scene_dir = Path(args.frame), Path(args.scene)
    gt = json.loads((frame / "gt.json").read_text())
    scene = json.loads((scene_dir / "scene.json").read_text())

    gt_names = list(gt["objects"])
    gt_pts = {n: np.load(frame / "gt_points" / f"{n}.npy").astype(np.float64) for n in gt_names}

    est = []
    for o in scene["objects"]:
        mesh = trimesh.load(scene_dir / o["mesh"], process=False, force="mesh")
        mesh.apply_transform(np.asarray(o["pose_cam"]["matrix"]))
        est.append((o, mesh, trimesh.sample.sample_surface(mesh, 5000, seed=0)[0]))

    # GT points are mesh vertices (uneven density) - use the vertex set's bbox center as the reference center
    gt_center = {n: (p.min(0) + p.max(0)) / 2 for n, p in gt_pts.items()}
    cost = np.array([[np.linalg.norm(m.bounds.mean(0) - gt_center[n]) for n in gt_names] for _, m, _ in est])
    rows, cols = linear_sum_assignment(cost) if len(est) else ([], [])

    print(f"{'GT object':16s} {'est':8s} {'label':8s} {'center err':>11s} {'chamfer':>9s} "
          f"{'GT size (mm)':>20s} {'est size (mm)':>20s} {'ICP fit':>7s}")
    results = {}
    for r, c in zip(rows, cols):
        o, mesh, samples = est[r]
        n = gt_names[c]
        center_err = cost[r, c]
        est_tree = cKDTree(samples)
        # GT vertices are sparse on flat faces; densify by sampling the convex hull surface for the GT side
        gt_dense = trimesh.convex.convex_hull(gt_pts[n]).sample(5000)
        chamfer = 0.5 * (cKDTree(gt_dense).query(samples)[0].mean() + est_tree.query(gt_dense)[0].mean())
        g, e = obb_extents(gt_pts[n]) * 1000, obb_extents(mesh.vertices) * 1000
        fit = o.get("icp_fitness")
        print(f"{n:16s} {o['id']:8s} {o['label']:8s} {center_err * 1000:8.1f} mm {chamfer * 1000:6.1f} mm "
              f"{str(np.round(g).astype(int)):>20s} {str(np.round(e).astype(int)):>20s} "
              f"{fit if fit is None else round(fit, 2)!s:>7s}")
        results[n] = {"est_id": o["id"], "center_err_m": float(center_err), "chamfer_m": float(chamfer),
                      "gt_extents_mm": g.tolist(), "est_extents_mm": e.tolist()}
    missed = [gt_names[c] for c in range(len(gt_names)) if c not in set(cols)]
    extra = [est[r][0]["id"] for r in range(len(est)) if r not in set(rows)]
    print(f"detected {len(rows)}/{len(gt_names)}  missed={missed}  extra={extra}")
    (scene_dir / "eval_vs_gt.json").write_text(json.dumps({"objects": results, "missed": missed, "extra": extra}, indent=2))


if __name__ == "__main__":
    main()
