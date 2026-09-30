import os
import shutil
import subprocess
import json
import re

def get_file_size_str(file_path):
    if not os.path.exists(file_path):
        return "N/A"
    size_bytes = os.path.getsize(file_path)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def main():
    base_dir = "data"
    template_path = os.path.join(base_dir, "template.blend")
    
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return

    # Find all subdirectories in data/
    data_folders = [
        f for f in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, f)) 
        and not f.startswith('.')
        and f != "__pycache__"
    ]
    
    # Sort for consistent order
    data_folders.sort()
    
    print(f"Found {len(data_folders)} folders in {base_dir}")

    successful_scenes = []

    for folder_name in data_folders:
        folder_path = os.path.join(base_dir, folder_name)
        target_blend = os.path.join(folder_path, "template.blend")
        
        print(f"Processing {folder_name}...")
        
        try:
            shutil.copy2(template_path, target_blend)
        except Exception as e:
            print(f"Failed to copy template to {folder_path}: {e}")
            continue

        output_usd = os.path.abspath(os.path.join(folder_path, "model.usdc"))
        output_glb = os.path.abspath(os.path.join(folder_path, "model.glb"))
        
        # Path to the external export script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        export_script = os.path.join(script_dir, "export_script.py")
        
        cmd = [
            "blender",
            target_blend,
            "--background",
            "--python", export_script,
            "--",
            output_usd,
            output_glb
        ]

        faces_count = "Unknown"
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Blender error for {folder_name}:\n{result.stderr}")
            else:
                # print(f"Blender output for {folder_name}:\n{result.stdout}")
                
                # Parse total faces
                match = re.search(r"TOTAL_FACES:\s*(\d+)", result.stdout)
                if match:
                    faces_count = match.group(1)
                
                print(f"Successfully exported USD and GLB for {folder_name} (Faces: {faces_count})")
                
                # Get file size
                file_size = get_file_size_str(output_glb)
                
                successful_scenes.append({
                    "name": folder_name,
                    "faces": faces_count,
                    "size": file_size
                })
                
        except FileNotFoundError:
            print("Error: 'blender' command not found.")
            break
        except Exception as e:
            print(f"Failed to run Blender for {folder_name}: {e}")

    # Generate scenes.json
    json_path = os.path.join(base_dir, "scenes.json")
    with open(json_path, 'w') as f:
        json.dump(successful_scenes, f, indent=2)
    print(f"Generated {json_path}")

    # Generate viewer.html
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scene Viewer</title>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <style>
        body { margin: 0; font-family: sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
        
        #header {
            background: #333; color: white; padding: 10px 20px; display: flex; gap: 20px; align-items: center;
        }
        #header h1 { margin: 0; font-size: 18px; flex: 1; }
        .nav-btn {
            background: #555; color: white; border: none; padding: 8px 16px; cursor: pointer; border-radius: 4px;
        }
        .nav-btn.active { background: #007bff; }
        
        /* Main Layout */
        #main-content { display: flex; flex: 1; overflow: hidden; position: relative; }
        
        /* 3D Viewer Mode */
        #sidebar { 
            width: 300px; border-right: 1px solid #ccc; overflow-y: auto; background: #f8f9fa; display: flex; flex-direction: column;
        }
        #scene-list { flex: 1; }
        .scene-item { padding: 12px 15px; cursor: pointer; border-bottom: 1px solid #e9ecef; }
        .scene-item:hover { background: #e2e6ea; }
        .scene-item.active { background: #007bff; color: white; }
        .scene-meta { font-size: 11px; color: #666; margin-top: 4px; }
        
        #viewer-container { flex: 1; background: #f0f0f0; position: relative; }
        model-viewer { width: 100%; height: 100%; }

        /* Gallery Mode */
        #gallery-view { 
            display: none; flex: 1; overflow-y: auto; padding: 20px; background: #eee;
        }
        .gallery-grid {
            display: flex; flex-wrap: wrap; gap: 20px;
        }
        .card {
            background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            width: calc(33.333% - 20px); box-sizing: border-box;
            display: flex; flex-direction: column;
        }
        .card h3 { margin: 0 0 5px 0; font-size: 16px; }
        .card-meta { font-size: 12px; color: #666; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 8px; }
        .card-images { display: flex; gap: 10px; }
        .img-container { flex: 1; position: relative; }
        .img-container img { width: 100%; border-radius: 4px; display: block; }
        .img-label { font-size: 12px; color: #666; margin-bottom: 4px; }
        
        .delete-btn {
            position: absolute; top: 5px; right: 5px; 
            background: rgba(255,0,0,0.8); color: white; border: none; 
            width: 24px; height: 24px; border-radius: 50%; cursor: pointer;
            display: flex; align-items: center; justify-content: center; font-size: 14px;
        }
        .delete-btn:hover { background: red; }

        @media (max-width: 1000px) {
            .card { width: calc(50% - 20px); }
        }

        /* Print Styles */
        @media print {
            body { height: auto; overflow: visible; }
            #header, #sidebar, #viewer-container { display: none !important; }
            #main-content { display: block; overflow: visible; }
            #gallery-view { display: block !important; background: white; padding: 0; overflow: visible; }
            .gallery-grid { gap: 10px; }
            .card { 
                width: calc(50% - 10px); 
                box-shadow: none; border: 1px solid #ccc; page-break-inside: avoid; 
                margin-bottom: 20px;
            }
            .delete-btn { display: none; }
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>Storefront Browser</h1>
        <button class="nav-btn active" onclick="switchMode('viewer')">3D Viewer</button>
        <button class="nav-btn" onclick="switchMode('gallery')">Gallery Cards</button>
    </div>

    <div id="main-content">
        <!-- 3D Viewer Structure -->
        <div id="sidebar">
            <div id="scene-list"></div>
        </div>
        <div id="viewer-container">
            <model-viewer id="viewer" camera-controls auto-rotate shadow-intensity="1" exposure="1"></model-viewer>
        </div>

        <!-- Gallery Structure -->
        <div id="gallery-view">
            <div class="gallery-grid" id="gallery-grid"></div>
        </div>
    </div>

    <script>
        let allScenes = [];
        
        async function init() {
            try {
                const response = await fetch('scenes.json');
                allScenes = await response.json();
                renderSidebar();
                renderGallery();
                
                // Handle hash routing on init
                handleHash();
                
                // Listen for hash changes
                window.addEventListener('hashchange', handleHash);
                
                // If no hash, default to first scene
                if (!window.location.hash && allScenes.length > 0) {
                     loadScene3D(allScenes[0]);
                }
            } catch (err) {
                console.error("Error loading scenes", err);
            }
        }

        function handleHash() {
            const hash = window.location.hash.substring(1); // remove #
            if (!hash) return;

            if (hash === 'gallery') {
                switchMode('gallery');
            } else if (hash === 'viewer') {
                 switchMode('viewer');
                 if (allScenes.length > 0) loadScene3D(allScenes[0]);
            } else if (hash.startsWith('scene=')) {
                const sceneName = hash.split('=')[1];
                const scene = allScenes.find(s => s.name === sceneName);
                if (scene) {
                    switchMode('viewer');
                    loadScene3D(scene);
                }
            }
        }

        function switchMode(mode) {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            // Correct button selection based on onclick attribute text
            const btn = Array.from(document.querySelectorAll('.nav-btn')).find(b => b.getAttribute('onclick').includes(mode));
            if (btn) btn.classList.add('active');

            if (mode === 'viewer') {
                document.getElementById('sidebar').style.display = 'flex';
                document.getElementById('viewer-container').style.display = 'flex';
                document.getElementById('gallery-view').style.display = 'none';
                // Only set hash if not already setting a specific scene
                if (!window.location.hash.startsWith('#scene=')) {
                   window.location.hash = 'viewer'; 
                }
            } else {
                document.getElementById('sidebar').style.display = 'none';
                document.getElementById('viewer-container').style.display = 'none';
                document.getElementById('gallery-view').style.display = 'block';
                window.location.hash = 'gallery';
            }
        }

        // 3D Viewer Functions
        function renderSidebar() {
            const list = document.getElementById('scene-list');
            list.innerHTML = '';
            allScenes.forEach(scene => {
                const div = document.createElement('div');
                div.className = 'scene-item';
                div.setAttribute('data-scene', scene.name); // for easy finding
                div.innerHTML = `
                    <div>${scene.name}</div>
                    <div class="scene-meta">${scene.faces} faces | ${scene.size}</div>
                `;
                div.onclick = () => {
                   // Set hash to trigger route handler
                   window.location.hash = `scene=${scene.name}`;
                };
                list.appendChild(div);
            });
        }

        function loadScene3D(scene) {
            // Update UI active state
            document.querySelectorAll('.scene-item').forEach(el => el.classList.remove('active'));
            const activeItem = document.querySelector(`.scene-item[data-scene="${scene.name}"]`);
            if (activeItem) {
                activeItem.classList.add('active');
                activeItem.scrollIntoView({ block: 'nearest' });
            }

            const viewer = document.getElementById('viewer');
            // Check if src is different to avoid reloading if already loaded (optional optimization)
            const newSrc = `${scene.name}/model.glb`;
            if (viewer.src && viewer.src.endsWith(newSrc)) return;

            viewer.src = newSrc;
            viewer.alt = `Model for ${scene.name}`;
        }

        // Gallery Functions
        function renderGallery() {
            const grid = document.getElementById('gallery-grid');
            grid.innerHTML = '';
            allScenes.forEach(scene => {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <h3>${scene.name}</h3>
                    <div class="card-meta">
                        Faces: ${scene.faces} &bull; Size: ${scene.size}
                    </div>
                    <div class="card-images">
                        <div class="img-container">
                            <div class="img-label">RGB</div>
                            <img src="${scene.name}/rgb.png" onerror="this.style.display='none'">
                            <button class="delete-btn" title="Delete Image" onclick="deleteFile('${scene.name}/rgb.png', this)">×</button>
                        </div>
                        <div class="img-container">
                            <div class="img-label">Depth</div>
                            <img src="${scene.name}/depth.png" onerror="this.style.display='none'">
                            <button class="delete-btn" title="Delete Image" onclick="deleteFile('${scene.name}/depth.png', this)">×</button>
                        </div>
                    </div>
                `;
                grid.appendChild(card);
            });
        }

        async function deleteFile(path, btnElement) {
            if (!confirm(`Are you sure you want to delete ${path}?`)) return;

            try {
                const res = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: path })
                });
                const data = await res.json();
                
                if (data.success) {
                    const container = btnElement.parentElement;
                    container.style.opacity = '0.3';
                    btnElement.remove(); 
                } else {
                    alert('Error deleting file: ' + (data.error || 'Unknown error'));
                }
            } catch (err) {
                alert('Error connecting to server. Make sure you are running via the python server.');
                console.error(err);
            }
        }

        init();
    </script>
</body>
</html>
"""
    
    html_path = os.path.join(base_dir, "viewer.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"Generated {html_path}")

if __name__ == "__main__":
    main()
