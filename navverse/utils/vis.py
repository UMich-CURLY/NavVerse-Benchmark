import omni
from pxr import Sdf, UsdGeom, Gf, UsdShade, Vt, Usd
import numpy as np
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from scipy.spatial.transform import Rotation as R
import isaacsim.core.utils.prims as prim_utils

def visualize_points(nodes, prim_path="/World/Points", color=(0.0, 1.0, 1.0), width=2.0):
    '''Create and draw a Points on the stage following the nodes'''
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Points.Define(stage, prim_path)
    prim.CreatePointsAttr(nodes)
    prim.CreateWidthsAttr(np.array([width], dtype=float))
    prim.CreateDisplayColorAttr([color])

def visualize_arrow(nodes, robot_height, prim_path="/World/Arrow", color=(1, 0, 0), scale=(1.0, 0.5, 0.5)):
    '''Create and draw an Arrow on the stage following the nodes'''
    # create a xform, remove old xform if it exists
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim_list = prim_utils.find_matching_prim_paths(prim_path+"/arrow*")
    for prim_path in prim_list:
        prim_utils.delete_prim(prim_path)
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path+"/arrow",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=scale,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        }
    )
    markers = VisualizationMarkers(cfg)
    nodes = np.array(nodes)
    marker_locations = np.concatenate([nodes[:, :2], np.ones((len(nodes), 1))*robot_height], axis=1)
    marker_orientations = R.from_euler('z', nodes[:, 2]).as_quat()
    marker_orientations = np.concatenate([marker_orientations[:, 3:4], marker_orientations[:, :3]], axis=1)
    markers.visualize(marker_locations, marker_orientations)

def visualize_curve(path, prim_path="/World/Path", color=(1, 0, 0), width=1.0 ):
    '''Create and draw a BasisCurve on the stage following the nodes'''

    if len(path) <= 1:
        print('[WARNING]: No valid path to visualize')
        return

    nodes = [Gf.Vec3f(float(pt[0]), float(pt[1]), float(pt[2])) for pt in path]

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.BasisCurves.Define(stage, prim_path)
    prim.CreatePointsAttr(nodes)

    # Set the number of curve verts to be the same as the number of points we have
    curve_verts = prim.CreateCurveVertexCountsAttr()
    curve_verts.Set([len(nodes)])

    # Set the curve type to linear so that each node is connected to the next
    type_attr = prim.CreateTypeAttr()
    type_attr.Set('linear')
    type_attr = prim.GetTypeAttr().Get()
    # Set the width of the curve

    width_attr = prim.CreateWidthsAttr(np.array([width], dtype=float))

    width =  Vt.FloatArray.FromNumpy(np.asarray([width for x in range(len(nodes))]))

    width_attr.Set(width)

    # color_primvar = prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    UsdGeom.Primvar(prim.GetDisplayColorAttr()).SetInterpolation("constant")
    prim.GetDisplayColorAttr().Set([color])

def visualize_mesh(prim_path, points, indices, colors=None, opacity=None, use_prevsrf=True):
    '''
    Create a mesh in USD
    '''

    time = Usd.TimeCode.Default()
    stage = omni.usd.get_context().get_stage()

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(points, time)
    mesh.GetFaceVertexIndicesAttr().Set(indices, time)
    mesh.GetFaceVertexCountsAttr().Set([3] * len(indices), time)

    if use_prevsrf:

        mtl_path = Sdf.Path(f"/World/Looks/PreviewSurface_{prim_path.split('/')[-1]}")

        mtl = UsdShade.Material.Define(stage, mtl_path)
        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(colors) 
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
        mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
        # Bind the mesh
        UsdShade.MaterialBindingAPI(mesh).Bind(mtl)

    # Opacity seems to be broken
    else:
        UsdGeom.Primvar(mesh.GetDisplayColorAttr()).SetInterpolation("constant")
        UsdGeom.Primvar(mesh.GetDisplayOpacityAttr()).SetInterpolation("constant")
        if colors:
            mesh.GetDisplayColorAttr().Set(colors, time)
        if opacity:
            mesh.GetDisplayOpacityAttr().Set(opacity, time)

    return prim_path