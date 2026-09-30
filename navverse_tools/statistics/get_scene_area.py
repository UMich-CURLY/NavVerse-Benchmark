# python get_scene_area.py
import logging
import argparse

# -----------------------------
# HELPER FUNCTION
# -----------------------------
from pxr import Usd, UsdGeom

log_file = "output_prims.log"

# minimal logger setup
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def get_prim_bbox(usd_file, prim_path):
    stage = Usd.Stage.Open(usd_file)
    prim = stage.GetPrimAtPath(prim_path)

    # Iterate over all prims recursively
    # for prim in stage.Traverse():
    #     logger.info(f"{prim.GetPath()} {prim.GetTypeName()}")

    if prim is None:
        raise ValueError(f"Prim {prim_path} not found")

    if not prim.IsA(UsdGeom.Imageable):
        raise ValueError(f"Prim {prim_path} is not geometry")

    imageable = UsdGeom.Imageable(prim)
    # Compute bounding box in world coordinates
    bbox = imageable.ComputeWorldBound(
        time=Usd.TimeCode.Default(),
        purpose1=UsdGeom.Tokens.default_,
        purpose2=UsdGeom.Tokens.render,
        purpose3=UsdGeom.Tokens.proxy,
    )

    bbox_range = bbox.GetRange()
    min_pt = bbox_range.GetMin()  # Gf.Vec3d
    max_pt = bbox_range.GetMax()  # Gf.Vec3d
    return tuple(min_pt), tuple(max_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the bounding box of a USD prim")
    parser.add_argument("usd_file")
    parser.add_argument("prim_path")
    args = parser.parse_args()
    min_bound, max_bound = get_prim_bbox(args.usd_file, args.prim_path)
    print("Min:", min_bound)
    print("Max:", max_bound)
