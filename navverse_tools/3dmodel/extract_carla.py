import os
import subprocess
import argparse
from tqdm import tqdm
import time

def is_lfs_pointer(filepath):
    try:
        if os.path.getsize(filepath) > 1024:
            return False
        with open(filepath, 'rb') as f:
            header = f.read(100)
            return b'version https://git-lfs.github.com/spec/v1' in header
    except Exception:
        return False

def pull_lfs_files(files, carla_content_path):
    if not files:
        return
    
    batch_size = 50
    env = os.environ.copy()
    software_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/software"))
    env["PATH"] = f"{software_dir}:{env['PATH']}"

    for i in tqdm(range(0, len(files), batch_size), desc="Pulling LFS files"):
        batch = files[i:i+batch_size]
        rel_paths = [os.path.relpath(f, carla_content_path) for f in batch]
        
        cmd = ["git", "lfs", "pull", "--include", ",".join(rel_paths)]
        
        try:
            subprocess.run(cmd, cwd=carla_content_path, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error pulling batch {i}: {e}")

def export_models(files, carla_content_path, output_path, umodel_path):
    if not files:
        return

    env = os.environ.copy()
    lib_path = os.path.abspath(os.path.join(os.path.dirname(umodel_path), "../lib"))
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
    
    batch_size = 20
    
    success_count = 0
    fail_count = 0
    
    for i in tqdm(range(0, len(files), batch_size), desc="Exporting models"):
        batch = files[i:i+batch_size]
        rel_paths = [os.path.relpath(f, carla_content_path) for f in batch]
        
        cmd = [
            umodel_path,
            f"-path={carla_content_path}",
            "-game=ue4.26",
            "-export",
            f"-out={output_path}",
            "-lods",
            "-uc",
            "-gltf"
        ] + rel_paths
        
        try:
            result = subprocess.run(cmd, env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = result.stdout.decode(errors='replace')
            
            if result.returncode == 0:
                success_count += len(batch)
            else:
                fail_count += len(batch)
                
        except Exception as e:
            print(f"Exception during export: {e}")
            fail_count += len(batch)

    print(f"Export finished. Success (approx): {success_count}, Failed (approx): {fail_count}")

def main():
    parser = argparse.ArgumentParser(description="Extract models from Carla content")
    parser.add_argument("--content_path", required=True, help="Path to carla-content repo")
    parser.add_argument("--output_path", required=True, help="Path to save extracted models")
    parser.add_argument("--target_subdir", default="Static", help="Subdirectory to scan within content path")
    args = parser.parse_args()

    umodel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/software/UEViewer/umodel"))
    
    target_dir = os.path.join(args.content_path, args.target_subdir)
    
    if not os.path.exists(target_dir):
        print(f"Target directory {target_dir} does not exist.")
        return

    print(f"Scanning {target_dir} for uassets...")
    uasset_files = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".uasset") and os.path.basename(file).startswith("SM_"): 
                uasset_files.append(os.path.join(root, file))
                
    print(f"Found {len(uasset_files)} SM_*.uasset files.")
    
    lfs_files = []
    for f in tqdm(uasset_files, desc="Checking LFS status"):
        if is_lfs_pointer(f):
            lfs_files.append(f)
            
    if lfs_files:
        print(f"Found {len(lfs_files)} LFS pointer files to pull.")
        pull_lfs_files(lfs_files, args.content_path)
    else:
        print("No LFS pointers found.")
    
    print("Exporting models...")
    export_models(uasset_files, args.content_path, args.output_path, umodel_path)
    
if __name__ == "__main__":
    main()

    
if __name__ == "__main__":
    main()
