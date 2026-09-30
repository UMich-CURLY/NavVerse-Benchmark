## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu
## Usage (run from repo root): python remove_prim.py --input_folder navverse_data/grscenes_home_raw/scenes/ --prim_name GroundPlane --amount 1

from pxr import Usd
import os
import argparse

def remove_prims_by_name(usd_path, name, amount=1):
    # Open the stage
    stage = Usd.Stage.Open(usd_path)

    # Find all prims with the specified name
    prims_to_remove = [prim for prim in stage.Traverse() if prim.GetName() == name]

    # Report how many were found
    print(f"Found {len(prims_to_remove)} prim(s) named '{name}'.")

    # Remove the specified amount of prims
    for i in range(min(amount, len(prims_to_remove))):
        stage.RemovePrim(prims_to_remove[i].GetPath())

    # Save the changes
    stage.GetRootLayer().Save()

    print(f"Prim removal complete for {usd_path}")

def remove_prims_in_folder(input_folder, name, amount=1):

    for root, dirs, files in os.walk(input_folder):
        if 'start_result_navigation.usd' in files:
            usd_path = os.path.join(root, 'start_result_navigation.usd')
            print(f"Processing USD file: {usd_path}")
            remove_prims_by_name(usd_path, name, amount)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Remove prim with specified prim name in USD files")
    parser.add_argument("--input_folder", 
                        type=str, 
                        required=True,
                        help="Path to the input folder containing USD files to process")
    parser.add_argument("--prim_name", 
                        type=str, 
                        required=True,
                        help="Name of the prim to remove")
    parser.add_argument("--amount", 
                        type=int, 
                        default=1,
                        help="Number of prims to remove (default: 1)")

    args = parser.parse_args()

    remove_prims_in_folder(args.input_folder, args.prim_name, args.amount)
