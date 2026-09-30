import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector


CUT_INTERIOR_MATERIAL_NAME = "NavVerseCutInterior"


def parse_vec2(value):
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected x,y")
    return parts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a VC raw building replacement with a single outdoor doorway."
    )
    parser.add_argument("--input-glb", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--center", help="Door center in VC navigation x,y coordinates")
    parser.add_argument("--bottom-z", type=float)
    parser.add_argument("--tangent", type=parse_vec2)
    parser.add_argument("--outward-normal", type=parse_vec2)
    parser.add_argument("--openings-json", help="Optional JSON list of openings for hollow_shell mode.")
    parser.add_argument("--door-width", type=float, default=4.5)
    parser.add_argument("--door-height", type=float, default=3.2)
    parser.add_argument("--door-depth", type=float, default=8.0)
    parser.add_argument("--inside-depth", type=float)
    parser.add_argument("--outside-depth", type=float)
    parser.add_argument(
        "--cut-mode",
        choices=("hollow_shell", "boolean_tunnel"),
        default="hollow_shell",
        help="hollow_shell builds an empty building shell with one facade opening; boolean_tunnel preserves the old through-cut behavior.",
    )
    parser.add_argument(
        "--align-shell-side-to-opening",
        action="store_true",
        help="Align the hollow-shell front wall to the first opening instead of the imported bbox extreme.",
    )
    parser.add_argument(
        "--input-z-up",
        action="store_true",
        help="Treat the input GLB vertex coordinates as Z-up instead of standard glTF Y-up.",
    )
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def openings_from_args(args):
    if args.openings_json:
        with open(args.openings_json, "r", encoding="utf-8") as f:
            openings = json.load(f)
        if not openings:
            raise ValueError(f"no openings in {args.openings_json}")
        return openings
    if args.center is None or args.bottom_z is None or args.tangent is None or args.outward_normal is None:
        raise ValueError("--center, --bottom-z, --tangent, and --outward-normal are required without --openings-json")
    center_x, center_y = [float(x) for x in args.center.split(",")]
    opening = {
        "center": [center_x, center_y],
        "bottom_z": float(args.bottom_z),
        "tangent": list(args.tangent),
        "outward_normal": list(args.outward_normal),
        "width": float(args.door_width),
        "height": float(args.door_height),
    }
    if args.inside_depth is not None and args.outside_depth is not None:
        opening["inside_depth"] = float(args.inside_depth)
        opening["outside_depth"] = float(args.outside_depth)
    return [opening]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def normalize_2d(vec):
    x, y = vec
    length = math.hypot(x, y)
    if length <= 1e-9:
        raise ValueError(f"zero-length vector: {vec}")
    return (x / length, y / length)


def make_cutter_for_opening(opening, depth):
    center_x, center_y = [float(x) for x in opening["center"]]
    tangent = normalize_2d(opening["tangent"])
    normal = normalize_2d(opening["outward_normal"])
    width = float(opening.get("width", 4.5))
    height = float(opening.get("height", 3.2))
    bottom_z = float(opening["bottom_z"])

    cutter_tangent = Vector((tangent[0], tangent[1], 0.0))
    cutter_vertical = Vector((0.0, 0.0, 1.0))
    cutter_normal = Vector((normal[0], normal[1], 0.0))
    inside_depth = opening.get("inside_depth")
    outside_depth = opening.get("outside_depth")
    if inside_depth is not None and outside_depth is not None:
        inside_depth = max(0.0, float(inside_depth))
        outside_depth = max(0.0, float(outside_depth))
        depth = max(inside_depth + outside_depth, 1e-3)
        normal_offset = (outside_depth - inside_depth) * 0.5
    else:
        normal_offset = 0.0

    cutter_center = Vector(
        (
            center_x,
            center_y,
            bottom_z + height * 0.5,
        )
    ) + cutter_normal * normal_offset

    matrix = Matrix(
        (
            (cutter_tangent.x, cutter_vertical.x, cutter_normal.x),
            (cutter_tangent.y, cutter_vertical.y, cutter_normal.y),
            (cutter_tangent.z, cutter_vertical.z, cutter_normal.z),
        )
    )
    if matrix.determinant() < 0:
        cutter_normal.negate()
        matrix = Matrix(
            (
                (cutter_tangent.x, cutter_vertical.x, cutter_normal.x),
                (cutter_tangent.y, cutter_vertical.y, cutter_normal.y),
                (cutter_tangent.z, cutter_vertical.z, cutter_normal.z),
            )
        )

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=cutter_center)
    cutter = bpy.context.object
    cutter.name = "door_opening_cutter"
    cutter.rotation_euler = matrix.to_euler()
    cutter.dimensions = (width, height, depth)
    bpy.context.view_layer.objects.active = cutter
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return cutter


def make_cutter(args):
    return make_cutter_for_opening(openings_from_args(args)[0], args.door_depth)


def cut_interior_material():
    material = bpy.data.materials.get(CUT_INTERIOR_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(CUT_INTERIOR_MATERIAL_NAME)
    material.diffuse_color = (0.72, 0.73, 0.74, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.72, 0.73, 0.74, 1.0)
        principled.inputs["Roughness"].default_value = 0.9
        if "Metallic" in principled.inputs:
            principled.inputs["Metallic"].default_value = 0.0
    return material


def material_slot_index(obj, material):
    for index, existing in enumerate(obj.data.materials):
        if existing == material:
            return index
    obj.data.materials.append(material)
    return len(obj.data.materials) - 1


def assign_cut_interior_faces(obj, cutter, material):
    material_index = material_slot_index(obj, material)
    cutter_inverse = cutter.matrix_world.inverted()
    cutter_corners = [Vector(corner) for corner in cutter.bound_box]
    minimum = Vector(tuple(min(corner[axis] for corner in cutter_corners) for axis in range(3)))
    maximum = Vector(tuple(max(corner[axis] for corner in cutter_corners) for axis in range(3)))
    tolerance = max(max(maximum[axis] - minimum[axis] for axis in range(3)) * 1e-5, 1e-4)
    assigned = 0
    for polygon in obj.data.polygons:
        center = cutter_inverse @ (obj.matrix_world @ polygon.center)
        if any(center[axis] < minimum[axis] - tolerance or center[axis] > maximum[axis] + tolerance for axis in range(3)):
            continue
        on_cutter_boundary = any(
            abs(center[axis] - minimum[axis]) <= tolerance
            or abs(center[axis] - maximum[axis]) <= tolerance
            for axis in range(3)
        )
        if not on_cutter_boundary:
            continue
        polygon.material_index = material_index
        assigned += 1
    obj.data.update()
    return assigned


def cutter_local_bounds(cutter):
    corners = [Vector(corner) for corner in cutter.bound_box]
    minimum = Vector(tuple(min(corner[axis] for corner in corners) for axis in range(3)))
    maximum = Vector(tuple(max(corner[axis] for corner in corners) for axis in range(3)))
    return minimum, maximum


def fallback_cut_open_mesh(obj, cutter, material):
    material_index = material_slot_index(obj, material)
    minimum, maximum = cutter_local_bounds(cutter)
    cutter_inverse = cutter.matrix_world.inverted()
    object_inverse = obj.matrix_world.inverted()
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)

    axes = (Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)))
    for axis_index, axis in enumerate(axes):
        for boundary in (minimum[axis_index], maximum[axis_index]):
            plane_local = Vector((0.0, 0.0, 0.0))
            plane_local[axis_index] = boundary
            plane_world = cutter.matrix_world @ plane_local
            normal_world = (cutter.matrix_world.to_3x3() @ axis).normalized()
            plane_object = object_inverse @ plane_world
            normal_object = (obj.matrix_world.to_3x3().transposed() @ normal_world).normalized()
            bmesh.ops.bisect_plane(
                mesh,
                geom=list(mesh.verts) + list(mesh.edges) + list(mesh.faces),
                plane_co=plane_object,
                plane_no=normal_object,
                dist=1e-5,
                clear_inner=False,
                clear_outer=False,
            )

    tolerance = 1e-4
    removed = []
    for face in mesh.faces:
        center_world = obj.matrix_world @ face.calc_center_median()
        center = cutter_inverse @ center_world
        if all(minimum[axis] - tolerance <= center[axis] <= maximum[axis] + tolerance for axis in range(3)):
            removed.append(face)
    if not removed:
        local_vertices = [cutter_inverse @ (obj.matrix_world @ vertex.co) for vertex in mesh.verts]
        inside_vertices = sum(
            all(minimum[axis] - tolerance <= point[axis] <= maximum[axis] + tolerance for axis in range(3))
            for point in local_vertices
        )
        world_vertices = [obj.matrix_world @ vertex.co for vertex in mesh.verts]
        world_min = tuple(min(point[axis] for point in world_vertices) for axis in range(3))
        world_max = tuple(max(point[axis] for point in world_vertices) for axis in range(3))
        cutter_world = [cutter.matrix_world @ Vector(corner) for corner in cutter.bound_box]
        cutter_min = tuple(min(point[axis] for point in cutter_world) for axis in range(3))
        cutter_max = tuple(max(point[axis] for point in cutter_world) for axis in range(3))
        print(
            f"Open-mesh fallback found no removable faces on {obj.name}; inside vertices: {inside_vertices}; "
            f"object bounds: {world_min}..{world_max}; cutter bounds: {cutter_min}..{cutter_max}"
        )
        mesh.free()
        return 0, 0
    bmesh.ops.delete(mesh, geom=removed, context="FACES")

    normal_min = max(float(minimum.z), -0.85)
    normal_max = min(float(maximum.z), 0.15)
    if normal_max <= normal_min + 1e-4:
        normal_min = float(minimum.z)
        normal_max = float(maximum.z)
    rectangles = (
        ((minimum.x, minimum.y, normal_min), (minimum.x, minimum.y, normal_max), (minimum.x, maximum.y, normal_max), (minimum.x, maximum.y, normal_min)),
        ((maximum.x, minimum.y, normal_max), (maximum.x, minimum.y, normal_min), (maximum.x, maximum.y, normal_min), (maximum.x, maximum.y, normal_max)),
        ((minimum.x, maximum.y, normal_min), (maximum.x, maximum.y, normal_min), (maximum.x, maximum.y, normal_max), (minimum.x, maximum.y, normal_max)),
        ((minimum.x, minimum.y, normal_max), (maximum.x, minimum.y, normal_max), (maximum.x, minimum.y, normal_min), (minimum.x, minimum.y, normal_min)),
    )
    uv_layer = mesh.loops.layers.uv.verify()
    assigned = 0
    for rectangle in rectangles:
        vertices = [mesh.verts.new(object_inverse @ (cutter.matrix_world @ Vector(point))) for point in rectangle]
        face = mesh.faces.new(vertices)
        face.material_index = material_index
        for loop, uv in zip(face.loops, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
            loop[uv_layer].uv = uv
        assigned += 1

    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    return assigned, len(removed)


def remove_interior_faces_outside_xy_bounds(obj, material, bounds):
    material_index = material_slot_index(obj, material)
    minimum, maximum = bounds
    tolerance = 1e-4
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    outside = []
    for face in mesh.faces:
        if face.material_index != material_index:
            continue
        if any(
            (world := obj.matrix_world @ vertex.co).x < minimum.x - tolerance
            or world.x > maximum.x + tolerance
            or world.y < minimum.y - tolerance
            or world.y > maximum.y + tolerance
            for vertex in face.verts
        ):
            outside.append(face)
    if outside:
        bmesh.ops.delete(mesh, geom=outside, context="FACES")
        mesh.to_mesh(obj.data)
        obj.data.update()
    mesh.free()
    return len(outside)


def scene_mesh_objects(exclude=None):
    exclude = exclude or set()
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj not in exclude]


def restore_z_up_import(objects):
    correction = Matrix.Rotation(math.radians(-90.0), 4, "X")
    for obj in objects:
        obj.matrix_world = correction @ obj.matrix_world


def imported_bbox(objects):
    points = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        raise RuntimeError("no mesh object bounds imported from input GLB")
    mn = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    mx = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return mn, mx


def local_to_world(tangent, outward, t_value, n_value, z_value):
    return Vector(
        (
            tangent.x * t_value + outward.x * n_value,
            tangent.y * t_value + outward.y * n_value,
            z_value,
        )
    )


def add_quad(vertices, faces, uvs, tangent, outward, corners, uv_kind):
    face = []
    face_uv = []
    t_values = [corner[0] for corner in corners]
    n_values = [corner[1] for corner in corners]
    z_values = [corner[2] for corner in corners]
    t_min, t_max = min(t_values), max(t_values)
    n_min, n_max = min(n_values), max(n_values)
    z_min, z_max = min(z_values), max(z_values)
    for t_value, n_value, z_value in corners:
        vertices.append(local_to_world(tangent, outward, t_value, n_value, z_value))
        face.append(len(vertices) - 1)
        if uv_kind == "front":
            face_uv.append((0.0 if t_max == t_min else (t_value - t_min) / (t_max - t_min), 0.0 if z_max == z_min else (z_value - z_min) / (z_max - z_min)))
        elif uv_kind == "side":
            face_uv.append((0.0 if n_max == n_min else (n_value - n_min) / (n_max - n_min), 0.0 if z_max == z_min else (z_value - z_min) / (z_max - z_min)))
        else:
            face_uv.append((0.0 if t_max == t_min else (t_value - t_min) / (t_max - t_min), 0.0 if n_max == n_min else (n_value - n_min) / (n_max - n_min)))
    faces.append(face)
    uvs.append(face_uv)


def make_hollow_shell(args, imported_objects):
    raw_openings = openings_from_args(args)
    tangent_2d = normalize_2d(raw_openings[0]["tangent"])
    outward_2d = normalize_2d(raw_openings[0]["outward_normal"])
    tangent = Vector((tangent_2d[0], tangent_2d[1], 0.0))
    outward = Vector((outward_2d[0], outward_2d[1], 0.0))

    bbox_min, bbox_max = imported_bbox(imported_objects)
    bbox_corners = [
        Vector((x, y, 0.0))
        for x in (bbox_min.x, bbox_max.x)
        for y in (bbox_min.y, bbox_max.y)
    ]
    t_min = min(c.dot(tangent) for c in bbox_corners)
    t_max = max(c.dot(tangent) for c in bbox_corners)
    n_min = min(c.dot(outward) for c in bbox_corners)
    n_max = max(c.dot(outward) for c in bbox_corners)
    z_min = bbox_min.z
    z_max = bbox_max.z

    side_dirs = {
        "front": outward,
        "back": -outward,
        "right": tangent,
        "left": -tangent,
    }
    wall_openings = {side: [] for side in side_dirs}
    for item in raw_openings:
        item_outward = Vector((*normalize_2d(item["outward_normal"]), 0.0))
        side, side_dot = max(side_dirs.items(), key=lambda value: item_outward.dot(value[1]))
        if item_outward.dot(side_dot) < 0.65:
            continue
        center = Vector((float(item["center"][0]), float(item["center"][1]), 0.0))
        width = float(item.get("width", args.door_width))
        height = float(item.get("height", args.door_height))
        bottom_z = float(item.get("bottom_z", args.bottom_z if args.bottom_z is not None else 0.0))
        horizontal = center.dot(tangent if side in {"front", "back"} else outward)
        z_min = min(z_min, bottom_z)
        wall_openings[side].append(
            {
                "center": center,
                "h": horizontal,
                "width": width,
                "bottom_z": bottom_z,
                "height": height,
            }
        )
    if not any(wall_openings.values()):
        raise RuntimeError("all openings were filtered out by facade orientation")

    front_n = n_max
    if args.align_shell_side_to_opening:
        opening_center = wall_openings["front"][0]["center"] if wall_openings["front"] else None
        if opening_center is None or any(wall_openings[side] for side in ("back", "left", "right")):
            raise RuntimeError("aligned hollow shell requires front-facing openings only")
        front_n = opening_center.dot(outward)
        if front_n <= n_min + 1e-3:
            raise RuntimeError("aligned hollow-shell front wall would collapse the building depth")
        n_max = front_n
    back_n = n_min

    vertices = []
    faces = []
    uvs = []

    def quad(corners, uv_kind="front"):
        add_quad(vertices, faces, uvs, tangent, outward, corners, uv_kind)

    def wall_with_openings(side, h_min, h_max, openings, make_corners, uv_kind="front"):
        rects = []
        for opening in openings:
            rects.append(
                (
                    max(h_min, opening["h"] - opening["width"] * 0.5),
                    min(h_max, opening["h"] + opening["width"] * 0.5),
                    max(z_min, opening["bottom_z"]),
                    min(z_max, opening["bottom_z"] + opening["height"]),
                )
            )
        h_cuts = sorted({h_min, h_max, *[value for rect in rects for value in rect[:2]]})
        z_cuts = sorted({z_min, z_max, *[value for rect in rects for value in rect[2:]]})
        for hi in range(len(h_cuts) - 1):
            h0, h1 = h_cuts[hi], h_cuts[hi + 1]
            if h1 <= h0:
                continue
            for zi in range(len(z_cuts) - 1):
                z0, z1 = z_cuts[zi], z_cuts[zi + 1]
                if z1 <= z0:
                    continue
                hc = (h0 + h1) * 0.5
                zc = (z0 + z1) * 0.5
                if any(rect[0] <= hc <= rect[1] and rect[2] <= zc <= rect[3] for rect in rects):
                    continue
                quad(make_corners(h0, h1, z0, z1), uv_kind)

    wall_with_openings(
        "front",
        t_min,
        t_max,
        wall_openings["front"],
        lambda h0, h1, z0, z1: [(h0, front_n, z0), (h1, front_n, z0), (h1, front_n, z1), (h0, front_n, z1)],
    )
    wall_with_openings(
        "back",
        t_min,
        t_max,
        wall_openings["back"],
        lambda h0, h1, z0, z1: [(h1, back_n, z0), (h0, back_n, z0), (h0, back_n, z1), (h1, back_n, z1)],
    )
    wall_with_openings(
        "left",
        n_min,
        n_max,
        wall_openings["left"],
        lambda h0, h1, z0, z1: [(t_min, h0, z0), (t_min, h1, z0), (t_min, h1, z1), (t_min, h0, z1)],
        "side",
    )
    wall_with_openings(
        "right",
        n_min,
        n_max,
        wall_openings["right"],
        lambda h0, h1, z0, z1: [(t_max, h1, z0), (t_max, h0, z0), (t_max, h0, z1), (t_max, h1, z1)],
        "side",
    )
    quad([(t_min, front_n, z_max), (t_max, front_n, z_max), (t_max, back_n, z_max), (t_min, back_n, z_max)], "top")

    mesh = bpy.data.meshes.new("hollow_building_shell_mesh")
    mesh.from_pydata([tuple(v) for v in vertices], [], faces)
    mesh.update()
    shell = bpy.data.objects.new("hollow_building_shell", mesh)
    bpy.context.collection.objects.link(shell)

    material = None
    for obj in imported_objects:
        if obj.data.materials:
            material = obj.data.materials[0]
            break
    if material is None:
        material = bpy.data.materials.new("hollow_building_material")
        material.diffuse_color = (0.35, 0.35, 0.35, 1.0)
    shell.data.materials.append(material)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly, face_uv in zip(mesh.polygons, uvs):
        for loop_index, uv in zip(poly.loop_indices, face_uv):
            uv_layer.data[loop_index].uv = uv

    for obj in imported_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    return [shell]


def cut_meshes(cutter, interior_material, source_bounds):
    mesh_objects = scene_mesh_objects(exclude={cutter})
    if not mesh_objects:
        raise RuntimeError("no mesh objects imported from input GLB")

    assigned_faces = 0
    for obj in mesh_objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        modifier = obj.modifiers.new("door_opening", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        try:
            modifier.solver = "EXACT"
        except TypeError:
            pass
        if hasattr(modifier, "use_hole_tolerant"):
            modifier.use_hole_tolerant = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        object_assigned = assign_cut_interior_faces(obj, cutter, interior_material)
        if object_assigned == 0:
            object_assigned, removed_faces = fallback_cut_open_mesh(obj, cutter, interior_material)
            if removed_faces:
                print(f"Open-mesh fallback removed {removed_faces} faces and added {object_assigned} reveal faces from {obj.name}")
        pruned_faces = remove_interior_faces_outside_xy_bounds(obj, interior_material, source_bounds)
        if pruned_faces:
            print(f"Removed {pruned_faces} cut-interior faces outside the source XY bounds from {obj.name}")
            object_assigned = max(0, object_assigned - pruned_faces)
        assigned_faces += object_assigned
        obj.select_set(False)

    bpy.data.objects.remove(cutter, do_unlink=True)
    return mesh_objects, assigned_faces


def export_glb(mesh_objects, output_glb):
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    Path(output_glb).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_glb),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=False,
        export_materials="EXPORT",
    )


def main():
    args = parse_args()
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=args.input_glb)
    imported_objects = scene_mesh_objects()
    if args.input_z_up:
        restore_z_up_import(imported_objects)
    source_bounds = imported_bbox(imported_objects)
    if args.cut_mode == "boolean_tunnel":
        output_objects = imported_objects
        interior_material = cut_interior_material()
        assigned_faces = 0
        for opening in openings_from_args(args):
            cutter = make_cutter_for_opening(opening, args.door_depth)
            output_objects, opening_faces = cut_meshes(cutter, interior_material, source_bounds)
            assigned_faces += opening_faces
        print(f"Assigned {assigned_faces} boolean-cut faces to {CUT_INTERIOR_MATERIAL_NAME}")
    else:
        output_objects = make_hollow_shell(args, imported_objects)
    export_glb(output_objects, args.output_glb)
    print(f"Exported cut building GLB: {args.output_glb}")


if __name__ == "__main__":
    main()
