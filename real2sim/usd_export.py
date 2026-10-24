"""Standalone OpenUSD scene & rigid-body exporter using `usd-core` (`pxr`).

Generates `objects/obj_XXX/obj.usd` and `scene.usd` directly in the `real2sim` environment
without requiring Isaac Sim / Omniverse to be installed or launched. The resulting USD files
carry the exact `UsdPhysics` and `Physx*` schemas expected by Isaac Sim and Isaac Lab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _gf_matrix(T):
    from pxr import Gf

    return Gf.Matrix4d(np.asarray(T, dtype=np.float64).T.tolist())  # USD is row-vector


def _add_physics_material(stage, path: str, friction: float, restitution: float = 0.0):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(restitution)
    return mat


def write_object_usd(
    scene_dir: str | Path,
    obj: dict,
    density: float = 500.0,
    friction: float = 0.8,
) -> Path:
    """Write `objects/obj_XXX/obj.usd` with visual mesh, vertex colors, rigid body, and PhysX convex decomposition."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    scene_dir = Path(scene_dir).resolve()
    mesh = trimesh.load(scene_dir / obj["mesh"], process=False, force="mesh")
    usd_path = scene_dir / "objects" / obj["id"] / "obj.usd"
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    if usd_path.exists():
        usd_path.unlink()

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateDensityAttr(float(density))

    points = Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32))
    counts = Vt.IntArray([3] * len(mesh.faces))
    indices = Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).reshape(-1))

    visual = UsdGeom.Mesh.Define(stage, "/Object/visual")
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr(counts)
    visual.CreateFaceVertexIndicesAttr(indices)
    visual.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(mesh.vertices):
        pv = visual.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.asarray(colors[:, :3], dtype=np.float32) / 255.0))
        mtl = UsdShade.Material.Define(stage, "/Object/Looks/vertex_color")
        reader = UsdShade.Shader.Define(stage, "/Object/Looks/vertex_color/color")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        pbr = UsdShade.Shader.Define(stage, "/Object/Looks/vertex_color/pbr")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        mtl.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(mtl)

    collision = UsdGeom.Mesh.Define(stage, "/Object/collision")
    collision.CreatePointsAttr(points)
    collision.CreateFaceVertexCountsAttr(counts)
    collision.CreateFaceVertexIndicesAttr(indices)
    collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
    cprim = collision.GetPrim()
    UsdPhysics.CollisionAPI.Apply(cprim)
    UsdPhysics.MeshCollisionAPI.Apply(cprim).CreateApproximationAttr("convexDecomposition")

    # Apply PhysX convex-decomposition & collision schemas via standard USD API (works without Omniverse PhysxSchema)
    for schema in ("PhysxConvexDecompositionCollisionAPI", "PhysxCollisionAPI"):
        cprim.AddAppliedSchema(schema)
    cprim.CreateAttribute("physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int, False).Set(32)
    cprim.CreateAttribute("physxConvexDecompositionCollision:hullVertexLimit", Sdf.ValueTypeNames.Int, False).Set(64)
    cprim.CreateAttribute("physxConvexDecompositionCollision:voxelResolution", Sdf.ValueTypeNames.Int, False).Set(300000)
    cprim.CreateAttribute("physxConvexDecompositionCollision:shrinkWrap", Sdf.ValueTypeNames.Bool, False).Set(True)
    cprim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, False).Set(0.002)

    mat = _add_physics_material(stage, "/Object/Looks/physics", float(friction))
    UsdShade.MaterialBindingAPI.Apply(cprim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    stage.GetRootLayer().Save()
    return usd_path


def export_usd_scene(
    scene_dir: str | Path,
    density: float = 500.0,
    friction: float = 0.8,
    table_margin: float = 0.3,
) -> Path:
    """Author per-object `objects/obj_XXX/obj.usd` and top-level `scene.usd` from `scene.json`."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    scene_dir = Path(scene_dir).resolve()
    scene = json.loads((scene_dir / "scene.json").read_text())
    stage_path = scene_dir / "scene.usd"
    if stage_path.exists():
        stage_path.unlink()

    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    physics = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)

    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr(1000.0)

    # static table: top face at z = 0, covering every object footprint + margin
    xy = np.array([o["pose_world"]["position"][:2] for o in scene["objects"]] or [[0.0, 0.0]])
    r = max(max(o["extents"]) for o in scene["objects"]) if scene["objects"] else 0.2
    lo, hi = xy.min(0) - r - table_margin, xy.max(0) + r + table_margin
    thickness = 0.05
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(table).SetTranslate(Gf.Vec3d(*((lo + hi) / 2), -thickness / 2))
    UsdGeom.XformCommonAPI(table).SetScale(Gf.Vec3f(*(hi - lo), thickness))
    table.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.5, 0.45)])
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(
        _add_physics_material(stage, "/World/Looks/table_physics", float(friction)),
        UsdShade.Tokens.weakerThanDescendants, "physics",
    )

    # the real camera, so renders can be compared with the captured image
    intr = scene["camera"]["intrinsics"]
    cam = UsdGeom.Camera.Define(stage, "/World/RealSense")
    T_cv_to_usd = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes -> USD camera axes (-Z forward, +Y up)
    UsdGeom.Xformable(cam).AddTransformOp().Set(_gf_matrix(np.asarray(scene["camera"]["T_world_cam"]) @ T_cv_to_usd))
    pixel = 0.003  # mm per pixel; any value works as long as aperture and focal length agree
    cam.CreateHorizontalApertureAttr(intr["width"] * pixel)
    cam.CreateVerticalApertureAttr(intr["height"] * pixel)
    cam.CreateFocalLengthAttr(intr["fx"] * pixel)
    cam.CreateHorizontalApertureOffsetAttr((intr["cx"] - intr["width"] / 2) * pixel)
    cam.CreateVerticalApertureOffsetAttr(-(intr["cy"] - intr["height"] / 2) * pixel)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))

    UsdGeom.Scope.Define(stage, "/World/Objects")
    for obj in scene["objects"]:
        usd = write_object_usd(scene_dir, obj, density=density, friction=friction)
        prim = UsdGeom.Xform.Define(stage, f"/World/Objects/{obj['id']}")
        prim.GetPrim().GetReferences().AddReference(str(usd.relative_to(scene_dir)))
        p, q = obj["pose_world"]["position"], obj["pose_world"]["quat_wxyz"]
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*p))
        xf.AddOrientOp().Set(Gf.Quatf(q[0], q[1], q[2], q[3]))
        print(f"  {obj['id']:8s} {obj['label']:12s} pos={np.round(p, 3)} extents={np.round(obj['extents'], 3)}")
    stage.GetRootLayer().Save()
    return stage_path


def main():
    ap = argparse.ArgumentParser(description="Export obj.usd and scene.usd without launching Isaac Sim.")
    ap.add_argument("--scene", required=True, help="scene directory containing scene.json")
    ap.add_argument("--density", type=float, default=500.0, help="object density kg/m^3")
    ap.add_argument("--friction", type=float, default=0.8)
    ap.add_argument("--table_margin", type=float, default=0.3, help="table extends this far beyond the objects, m")
    args = ap.parse_args()
    stage_path = export_usd_scene(args.scene, density=args.density, friction=args.friction, table_margin=args.table_margin)
    print(f"[usd_export] wrote {stage_path}")


if __name__ == "__main__":
    main()
