# Example pair (decimated vs undecimated) used for face-count comparison:
#   decimated:   navverse_data/grscenes_commercial_decimated/scenes/MWHLEPQKTIFZIAABAAAAAAA8_usd/start_result_navigation.usd
#   undecimated: navverse_data/grscenes_commercial/scenes/MV4AFHQKTKJZ2AABAAAAADQ8_usd/start_result_navigation.usd
# Usage (run from repo root): python navverse_tools/model_decimation/tests/count_faces_in_usd.py --scene_folder navverse_data/grscenes_commercial/scenes --scene_name start_result_navigation.usd

## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu

import os
import sys
import argparse
import csv
# Ensure the package root (model_decimation) is on sys.path so `utils` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf
from utils.mesh_utils import count_total_faces_in_mesh_prim

# Helper: try to write an Excel file if pandas is available, otherwise write CSV
def _write_results_table(results, output_path):
    """results: list of (scene_file, total_faces). Writes to XLSX if possible, else CSV."""
    if not results:
        print(f"[INFO] No results to write to {output_path}")
        return

    try:
        if output_path.lower().endswith('.xlsx'):
            try:
                import pandas as _pd
                df = _pd.DataFrame(results, columns=['scene_file', 'total_faces'])
                df.to_excel(output_path, index=False)
                print(f"[INFO] Wrote Excel file: {output_path}")
                return
            except Exception as e:
                print(f"[WARNING] Failed to write .xlsx with pandas: {e} -- falling back to CSV")

        # Fallback to CSV
        csv_path = output_path
        if output_path.lower().endswith('.xlsx'):
            csv_path = output_path[:-5] + '.csv'

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scene_file', 'total_faces'])
            for row in results:
                writer.writerow(row)

        print(f"[INFO] Wrote CSV file: {csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write results to {output_path}: {e}")

def count_total_faces_in_usd(usd_path):
    """Counts the total number of mesh faces in a USD file."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[ERROR] Failed to open USD stage: {usd_path}")
        return 0

    total_faces = 0
    for prim in stage.Traverse():
        total_faces += count_total_faces_in_mesh_prim(prim)

    print(f"[INFO] Total faces in USD file '{usd_path}': {total_faces}")

    return total_faces

def count_total_faces_in_folder(scene_folder, scene_name):
    """Walk folder, count faces for each matching scene_name, return list of (scene_file, total_faces)."""
    results = []
    for root, dirs, files in os.walk(scene_folder):
        # Only process folders containing the specified scene file
        if scene_name in files:
            scene_file = os.path.join(root, scene_name)
            total = count_total_faces_in_usd(scene_file)
            results.append((scene_file, total))
    return results


# --- SCRIPT USAGE EXAMPLE ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Count total mesh faces in USD files")
    parser.add_argument("--scene_folder", 
                        type=str, 
                        required=True,
                        help="Path to the folder containing the scene USD files to process")
    parser.add_argument("--scene_name", 
                        type=str, 
                        required=True,
                        default="start_result_navigation.usd",
                        help="Name of the scene file to process")

    args = parser.parse_args()
    
    # Use parsed arguments
    scene_folder = args.scene_folder
    scene_name = args.scene_name

    print(f"[INFO] Scene folder: {scene_folder}")
    print(f"[INFO] Scene name: {scene_name}")
    print(f"[INFO] Starting mesh face counting process...")

    # Validate scene folder exists
    if not os.path.exists(scene_folder):
        print(f"[ERROR] Scene folder does not exist: {scene_folder}")
        sys.exit(1)

    # Count total faces in the input USD files (may be multiple instances)
    results = count_total_faces_in_folder(scene_folder, scene_name)

    # Optionally write results to an output file
    output_file = getattr(args, 'output_file', None)
    if output_file:
        _write_results_table(results, output_file)
    
