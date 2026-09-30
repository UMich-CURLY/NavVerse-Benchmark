import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from pxr import Usd

def remove_specific_prims(usd_path):
    """
    Removes PhysicsMaterial, physicsScene, and GroundPlane from a USD file.
    
    Args:
        usd_path (str): Path to the USD file.
        
    Returns:
        str: Status message indicating success or failure.
    """
    try:
        # Open the stage
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            return f"[ERROR] Failed to open {usd_path}"
        
        # Prims to look for (case-insensitive match for robustness)
        target_names = {"physicsmaterial", "physicsscene", "groundplane"}
        
        prims_to_remove = []
        # Traverse all prims in the stage
        for prim in stage.Traverse():
            if prim.GetName().lower() in target_names:
                prims_to_remove.append(prim.GetPath())
        
        if not prims_to_remove:
            return f"[INFO] No matching prims found in {usd_path}"
            
        # Sort by path length descending to ensure we remove children before parents
        # (Though these specific prims are usually root-level or flat)
        prims_to_remove.sort(key=lambda x: len(str(x)), reverse=True)
        
        removed_count = 0
        for path in prims_to_remove:
            stage.RemovePrim(path)
            removed_count += 1
            
        # Save the changes to the root layer
        stage.GetRootLayer().Save()
        return f"[SUCCESS] Removed {removed_count} prims from {usd_path}"
        
    except Exception as e:
        return f"[ERROR] Processing {usd_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Remove PhysicsMaterial, physicsScene, and GroundPlane from USD files in parallel.")
    parser.add_argument("--input_folder", 
                        type=str, 
                        required=True, 
                        help="Path to the input folder containing USD files to process")
    parser.add_argument("--num_workers", 
                        type=int, 
                        default=8, 
                        help="Number of parallel worker processes (default: 8)")
    
    args = parser.parse_args()

    # Find all USD files recursively
    target_files = []
    print(f"Scanning {args.input_folder} for USD files...")
    
    for root, dirs, files in os.walk(args.input_folder):
        # Isaac Sim scene convention: start_result_navigation.usd is usually the main file
        if "start_result_navigation.usd" in files:
            target_files.append(os.path.join(root, "start_result_navigation.usd"))
            continue
            
        for file in files:
            # Match .usd files, but avoid common processed suffixes to prevent double processing
            if file.endswith(".usd") and not any(suffix in file for suffix in ["_renamed.usd", "_decimated.usd"]):
                target_files.append(os.path.join(root, file))
    
    target_files.sort()
    total_files = len(target_files)
    print(f"Found {total_files} files to process.")

    if total_files == 0:
        print("No USD files found in the specified folder.")
        return

    # Use multiprocessing pool to process files in parallel
    print(f"Starting processing with {args.num_workers} workers...")
    with Pool(args.num_workers) as pool:
        # tqdm for progress tracking
        results = list(tqdm(pool.imap(remove_specific_prims, target_files), 
                           total=total_files, 
                           desc="Processing USDs"))

    # Summary of results
    success_count = sum(1 for res in results if "[SUCCESS]" in res)
    info_count = sum(1 for res in results if "[INFO]" in res)
    error_count = sum(1 for res in results if "[ERROR]" in res)

    print("\n" + "="*50)
    print("Processing Summary:")
    print(f"  Total files:    {total_files}")
    print(f"  Successfully modified: {success_count}")
    print(f"  No changes needed:     {info_count}")
    print(f"  Errors encountered:    {error_count}")
    print("="*50)

    if error_count > 0:
        print("\nError details:")
        for res in results:
            if "[ERROR]" in res:
                print(f"  {res}")

if __name__ == "__main__":
    main()
