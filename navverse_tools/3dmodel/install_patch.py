"""Script to patch the installed objaverse package with the local modified version."""

import shutil
import os
import inspect
import sys

try:
    import objaverse.xl.thingiverse
except ImportError:
    print("Error: 'objaverse' package not found. Please ensure it is installed in the current environment.")
    sys.exit(1)

def install_patch():
    # Path to the installed file
    installed_file_path = inspect.getfile(objaverse.xl.thingiverse)
    
    # Path to the local modified file (in the same directory as this script)
    local_file_path = os.path.join(os.path.dirname(__file__), 'thingiverse.py')
    
    if not os.path.exists(local_file_path):
        print(f"Error: Local modified file not found at {local_file_path}")
        sys.exit(1)
        
    print(f"Found installed package at: {installed_file_path}")
    print(f"Applying patch from: {local_file_path}")
    
    # Create backup of the installed file
    backup_path = installed_file_path + ".bak"
    if not os.path.exists(backup_path):
        print(f"Creating backup at: {backup_path}")
        shutil.copy2(installed_file_path, backup_path)
    else:
        print(f"Backup already exists at: {backup_path}")
    
    # Overwrite the installed file with the local file
    try:
        shutil.copy2(local_file_path, installed_file_path)
        print("Successfully patched thingiverse.py in site-packages.")
    except Exception as e:
        print(f"Error applying patch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_patch()
