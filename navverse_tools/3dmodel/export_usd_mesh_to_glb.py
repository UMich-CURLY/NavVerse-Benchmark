#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom


def triangulate(face_vertex_counts, face_vertex_indices):
    faces = []
    offset = 0
    for count in face_vertex_counts:
        indices = face_vertex_indices[offset : offset + count]
        offset += count
        if count < 3:
            continue
        for idx in range(1, count - 1):
            faces.append([indices[0], indices[idx], indices[idx + 1]])
    return np.asarray(faces, dtype=np.int64)


def mesh_from_prim(prim, xform_cache):
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = [int(x) for x in mesh.GetFaceVertexCountsAttr().Get()]
    indices = [int(x) for x in mesh.GetFaceVertexIndicesAttr().Get()]
    faces = triangulate(counts, indices)
    if len(points) == 0 or len(faces) == 0:
        return None

    transform = xform_cache.GetLocalToWorldTransform(prim)
    transformed = []
    for point in points:
        transformed_point = transform.Transform(Gf.Vec3d(*point))
        transformed.append([float(transformed_point[0]), float(transformed_point[1]), float(transformed_point[2])])
    return trimesh.Trimesh(vertices=np.asarray(transformed), faces=faces, process=False)


def load_usd_meshes(usd_path):
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {usd_path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = mesh_from_prim(prim, xform_cache)
        if mesh is not None:
            meshes.append(mesh)
    if not meshes:
        raise RuntimeError(f"no mesh prims found in {usd_path}")
    return meshes


def main():
    parser = argparse.ArgumentParser(description="Export visible USD meshes to a geometry-only GLB.")
    parser.add_argument("--input-usd", required=True)
    parser.add_argument("--output-glb", required=True)
    args = parser.parse_args()

    output_glb = Path(args.output_glb)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for index, mesh in enumerate(load_usd_meshes(args.input_usd)):
        scene.add_geometry(mesh, node_name=f"mesh_{index}", geom_name=f"mesh_{index}")
    scene.export(output_glb)
    print(f"Wrote {output_glb}")


if __name__ == "__main__":
    main()
