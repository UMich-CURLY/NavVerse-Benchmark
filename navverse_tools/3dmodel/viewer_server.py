import os
import json
import shutil
from flask import Flask, jsonify, request, send_from_directory, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(BASE_DIR, 'data', 'render')
RESULT_DIR = os.path.join(BASE_DIR, 'result')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Model Viewer</title>
    <style>
        :root {
            --primary-color: #3b82f6;
            --primary-hover: #2563eb;
            --success-color: #10b981;
            --bg-color: #f8fafc;
            --sidebar-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            display: block; /* Changed from flex */
            height: auto; /* Allow body to scroll */
            overflow-y: auto; /* Main scroll */
            background-color: var(--bg-color);
            color: var(--text-color);
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }

        #sidebar {
            width: 280px;
            height: 100vh;
            position: fixed; /* Fixed sidebar */
            top: 0;
            left: 0;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            box-shadow: 4px 0 24px rgba(0,0,0,0.02);
            z-index: 20; /* Lower than expanded card */
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            background: var(--sidebar-bg);
            z-index: 20;
        }
        
        .sidebar-header h3 {
            margin: 0;
            font-size: 1.2rem;
            color: var(--text-color);
        }

        #keyword-list {
            padding: 10px;
        }

        .keyword-item {
            padding: 12px 16px;
            margin-bottom: 4px;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.95rem;
            color: #475569;
        }

        .keyword-item:hover {
            background-color: #f1f5f9;
            color: var(--primary-color);
        }

        .keyword-item.active {
            background-color: #eff6ff;
            color: var(--primary-color);
            font-weight: 600;
        }
        
        .keyword-badge {
            font-size: 0.75rem;
            background: #e2e8f0;
            padding: 2px 8px;
            border-radius: 12px;
            color: #64748b;
        }
        
        .has-selection-indicator {
            width: 8px;
            height: 8px;
            background-color: var(--primary-color);
            border-radius: 50%;
            margin-left: 8px;
            display: none;
        }
        
        .keyword-item.has-selection .has-selection-indicator {
            display: block;
        }

        #main-content {
            margin-left: 280px; /* Offset for fixed sidebar */
            display: flex;
            flex-direction: column;
            overflow: visible; /* No clipping */
            position: relative;
            min-height: 100vh;
        }

        #controls {
            padding: 20px 32px;
            background-color: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 16px;
            align-items: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
            z-index: 40; /* Above cards and sidebar */
            position: sticky;
            top: 0;
        }

        #card-grid-container {
            flex-grow: 1;
            padding: 32px;
            overflow: visible; /* No clipping */
        }


        #card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 24px;
            padding-bottom: 40px;
        }

        .card-wrapper {
            position: relative;
            height: 320px; /* Reserve space in grid */
            z-index: 1;
        }
        
        .card-wrapper:hover {
            z-index: 100; /* Bring active card to front */
        }

        .card {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s ease-out;
            position: absolute; /* Float above wrapper */
            top: 0;
            left: 0;
            width: 100%;
            height: 100%; /* Start at same size as wrapper */
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            display: flex;
            flex-direction: column;
            z-index: 1;
        }

        .card:hover {
            /* Expand dimensions */
            width: 150%; 
            left: -25%; /* Center horizontally */
            height: auto; /* Allow height to grow */
            min-height: 100%;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            border-color: #cbd5e1;
            transform: none; /* Disable scale transform if present */
            z-index: 100; /* Ensure it's above sidebar */
        }

        .card.selected {
            border: 2px solid var(--primary-color);
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15);
        }

        .card.selected:hover {
             transform: none;
        }
        
        .card.selected::after {
            content: "✓";
            position: absolute;
            top: 12px;
            right: 12px;
            background: var(--primary-color);
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            z-index: 5;
        }

        .card-image-container {
            width: 100%;
            height: 240px;
            background-color: #f8fafc;
            position: relative;
            overflow: hidden;
            border-bottom: 1px solid var(--border-color);
            transition: height 0.2s ease-out;
            flex-shrink: 0;
        }
        
        .card:hover .card-image-container {
            height: 480px; /* 2x original height */
        }

        .card img.main-image {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            padding: 12px;
            box-sizing: border-box;
            transition: opacity 0.2s;
        }
        
        /* Show hover grid on hover */
        .card:hover img.main-image {
            opacity: 0;
        }

        .hover-grid {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            opacity: 0;
            pointer-events: none; /* Let clicks pass through */
            transition: opacity 0.2s ease-in-out;
            background: white;
        }

        .card:hover .hover-grid {
            opacity: 1;
            pointer-events: auto;
        }
        
        .hover-grid img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border: 1px solid #f1f5f9;
            background: white;
            box-sizing: border-box;
            padding: 4px;
            display: block; /* Remove inline spacing */
        }

        .card-info {
            padding: 16px;
        }

        .card-title input {
            width: 100%;
            border: 1px solid transparent;
            background: transparent;
            font-size: 0.95rem;
            font-weight: 500;
            color: var(--text-color);
            text-align: center;
            padding: 2px 4px;
            border-radius: 4px;
            transition: all 0.2s;
        }

        .card-title input:hover {
            border-color: #cbd5e1;
            background: #f8fafc;
        }

        .card-title input:focus {
            outline: none;
            border-color: var(--primary-color);
            background: white;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }

        .card-status {
            font-size: 0.8rem;
            color: var(--primary-color);
            text-align: center;
            margin-top: 4px;
            height: 1.2em;
            font-weight: 600;
        }

        button {
            padding: 10px 24px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
        }

        button:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
            box-shadow: 0 6px 8px -1px rgba(59, 130, 246, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }

        button#finish-btn {
            background-color: var(--success-color);
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
        }

        button#finish-btn:hover {
            background-color: #059669;
            box-shadow: 0 6px 8px -1px rgba(16, 185, 129, 0.4);
        }
        
        /* Toast Notification */
        #toast-container {
            position: fixed;
            bottom: 32px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .toast {
            background: #1e293b;
            color: white;
            padding: 12px 24px;
            border-radius: 50px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
            display: flex;
            align-items: center;
            gap: 12px;
            opacity: 0;
            transform: translateY(20px);
            animation: slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            font-weight: 500;
            min-width: 300px;
            justify-content: center;
        }
        
        .toast.error {
            background: #ef4444;
        }
        
        .toast.success {
            background: #10b981;
        }
        
        @keyframes slideUp {
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes fadeOut {
            to { opacity: 0; transform: translateY(-10px); }
        }
        
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #94a3b8;
            gap: 16px;
        }
        
        .empty-icon {
            font-size: 4rem;
            opacity: 0.5;
        }
    </style>
</head>
<body>

<div id="sidebar">
    <div class="sidebar-header">
        <h3>Model Library</h3>
    </div>
    <div id="keyword-list"></div>
</div>

<div id="main-content">
    <div id="controls">
        <h2 id="current-keyword">Select a keyword</h2>
        <div style="flex-grow: 1;"></div>
        <button onclick="saveSelection()">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"></path></svg>
            Save Selection
        </button>
        <button id="finish-btn" onclick="finishProcessing()">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
            Finish & Export
        </button>
    </div>
    <div id="card-grid-container">
        <div id="card-grid">
            <div class="empty-state">
                <div class="empty-icon">👈</div>
                <div>Select a keyword from the sidebar to start</div>
            </div>
        </div>
    </div>
</div>

<div id="toast-container"></div>

<script>
    let currentData = {};
    let currentKeyword = null;

    // Fetch structure on load
    fetch('/api/structure')
        .then(response => response.json())
        .then(data => {
            currentData = data;
            renderSidebar();
        });

    function showToast(message, type = 'success') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = type === 'success' ? '✓' : '✕';
        
        toast.innerHTML = `<span>${icon}</span> ${message}`;
        container.appendChild(toast);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'fadeOut 0.3s forwards';
            setTimeout(() => {
                container.removeChild(toast);
            }, 300);
        }, 3000);
    }

    function renderSidebar() {
        const list = document.getElementById('keyword-list');
        list.innerHTML = '';
        Object.keys(currentData).sort().forEach(keyword => {
            const count = Object.keys(currentData[keyword]).length;
            const hasSelection = Object.values(currentData[keyword]).some(obj => obj.selected);
            
            const div = document.createElement('div');
            div.className = 'keyword-item';
            if (hasSelection) div.classList.add('has-selection');
            if (currentKeyword === keyword) div.classList.add('active');
            
            div.innerHTML = `
                <div style="display: flex; align-items: center;">
                    ${keyword}
                    <div class="has-selection-indicator" title="Has selected items"></div>
                </div>
                <span class="keyword-badge">${count}</span>
            `;
            
            div.onclick = () => loadKeyword(keyword);
            list.appendChild(div);
        });
    }

    function loadKeyword(keyword) {
        currentKeyword = keyword;
        document.getElementById('current-keyword').textContent = keyword;
        
        renderSidebar(); // Re-render to update active state

        const grid = document.getElementById('card-grid');
        grid.innerHTML = '';

        const objects = currentData[keyword];
        
        // Sort: regular files first, no_texture last
        const sortedObjectNames = Object.keys(objects).sort((a, b) => {
            const aNoTex = a.includes('no_texture');
            const bNoTex = b.includes('no_texture');
            if (aNoTex && !bNoTex) return 1;
            if (!aNoTex && bNoTex) return -1;
            return a.localeCompare(b);
        });

        if (sortedObjectNames.length === 0) {
            grid.innerHTML = '<div class="empty-state">No objects found in this category</div>';
            return;
        }

        sortedObjectNames.forEach(objName => {
            const objData = objects[objName];
            
            // Create wrapper for layout stability
            const wrapper = document.createElement('div');
            wrapper.className = 'card-wrapper';
            
            const card = document.createElement('div');
            card.className = 'card';
            if (objData.selected) card.classList.add('selected');
            card.dataset.objName = objName;
            card.onclick = (e) => {
                if (e.target.tagName !== 'INPUT') {
                    toggleSelect(card);
                }
            };

            // Find images for hover grid (render_0 to render_3)
            const renderFiles = [];
            for(let i=0; i<4; i++) {
                const f = `render_${i}.png`;
                if(objData.files.includes(f)) {
                    renderFiles.push(`/images/${keyword}/${objName}/${f}`);
                } else {
                    renderFiles.push('https://via.placeholder.com/150?text=No+Img');
                }
            }
            
            // Main image
            let imgFile = objData.files.find(f => f.endsWith('.png'));
            if (objData.files.includes('render_0.png')) imgFile = 'render_0.png';
            
            const mainImgPath = imgFile 
                ? `/images/${keyword}/${objName}/${imgFile}` 
                : 'https://via.placeholder.com/150?text=No+Image';

            const displayName = objData.custom_name || objName;

            card.innerHTML = `
                <div class="card-image-container">
                    <img src="${mainImgPath}" alt="${objName}" class="main-image">
                    <div class="hover-grid">
                        <img src="${renderFiles[0]}" alt="0">
                        <img src="${renderFiles[1]}" alt="1">
                        <img src="${renderFiles[2]}" alt="2">
                        <img src="${renderFiles[3]}" alt="3">
                    </div>
                </div>
                <div class="card-info">
                    <div class="card-title">
                        <input type="text" value="${displayName}" onclick="event.stopPropagation()" onchange="updateName(this, '${objName}')">
                    </div>
                    <div class="card-status">${objData.selected ? 'Selected' : ''}</div>
                </div>
            `;
            wrapper.appendChild(card);
            grid.appendChild(wrapper);
        });
    }

    function updateName(input, objName) {
        if (currentKeyword && currentData[currentKeyword][objName]) {
            currentData[currentKeyword][objName].custom_name = input.value;
        }
    }

    function toggleSelect(card) {
        card.classList.toggle('selected');
        const statusDiv = card.querySelector('.card-status');
        statusDiv.textContent = card.classList.contains('selected') ? 'Selected' : '';
    }

    function saveSelection() {
        if (!currentKeyword) {
            showToast('Please select a keyword first', 'error');
            return;
        }
        
        const grid = document.getElementById('card-grid');
        const updates = {};
        let hasChanges = false;
        
        grid.querySelectorAll('.card').forEach(card => {
            const objName = card.dataset.objName;
            const isSelected = card.classList.contains('selected');
            const nameInput = card.querySelector('input');
            const customName = nameInput ? nameInput.value : null;

            updates[objName] = {
                selected: isSelected,
                custom_name: customName
            };
            
            // Update local state
            if (currentData[currentKeyword][objName]) {
                const obj = currentData[currentKeyword][objName];
                if (obj.selected !== isSelected || obj.custom_name !== customName) {
                    obj.selected = isSelected;
                    obj.custom_name = customName;
                    hasChanges = true;
                }
            }
        });

        fetch('/api/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                keyword: currentKeyword,
                updates: updates
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                showToast('Selection saved successfully!');
                renderSidebar(); // Update sidebar indicators
            } else {
                showToast('Error saving selection', 'error');
            }
        })
        .catch(err => {
            showToast('Network error occurred', 'error');
        });
    }

    function finishProcessing() {
        if (!confirm('This will copy all selected objects to the result folder and merge JSONs. Continue?')) return;
        
        const btn = document.getElementById('finish-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = 'Processing...';
        btn.disabled = true;

        fetch('/api/finish', {
            method: 'POST'
        })
        .then(res => res.json())
        .then(data => {
            btn.innerHTML = originalText;
            btn.disabled = false;
            if (data.status === 'success') {
                showToast(`Finished! Copied ${data.count} objects to result folder`);
            } else {
                showToast('Error processing: ' + data.message, 'error');
            }
        })
        .catch(err => {
            btn.innerHTML = originalText;
            btn.disabled = false;
            showToast('Network error occurred', 'error');
        });
    }
</script>

</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/structure')
def get_structure():
    """
    Scans RENDER_DIR and returns:
    {
        "keyword1": {
            "object1": { "files": ["render_0.png", ...], "selected": true/false },
            ...
        },
        ...
    }
    """
    structure = {}
    if not os.path.exists(RENDER_DIR):
        return jsonify(structure)

    for keyword in os.listdir(RENDER_DIR):
        kw_path = os.path.join(RENDER_DIR, keyword)
        if not os.path.isdir(kw_path):
            continue
        
        structure[keyword] = {}
        for obj_name in os.listdir(kw_path):
            obj_path = os.path.join(kw_path, obj_name)
            if not os.path.isdir(obj_path):
                continue
            
            files = os.listdir(obj_path)
            
            # Check info.json for selection status
            selected = False
            custom_name = None
            info_path = os.path.join(obj_path, 'info.json')
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        selected = info.get('selected', False)
                        custom_name = info.get('custom_name')
                except:
                    pass
            
            structure[keyword][obj_name] = {
                "files": files,
                "selected": selected,
                "custom_name": custom_name
            }
            
    return jsonify(structure)

@app.route('/api/save', methods=['POST'])
def save_selection():
    data = request.json
    keyword = data.get('keyword')
    updates = data.get('updates', {})
    
    if not keyword:
        return jsonify({"status": "error", "message": "No keyword provided"})
    
    kw_path = os.path.join(RENDER_DIR, keyword)
    if not os.path.exists(kw_path):
        return jsonify({"status": "error", "message": "Keyword not found"})
        
    for obj_name, data in updates.items():
        obj_path = os.path.join(kw_path, obj_name)
        info_path = os.path.join(obj_path, 'info.json')
        
        # Determine what we are saving
        # If data is a dict, it has 'selected' and 'custom_name'
        # If it's a bool (legacy support), it's just selected
        if isinstance(data, dict):
            is_selected = data.get('selected', False)
            custom_name = data.get('custom_name')
        else:
            is_selected = data
            custom_name = None

        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                info['selected'] = is_selected
                if custom_name:
                    info['custom_name'] = custom_name
                
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=4)
            except Exception as e:
                print(f"Error updating {info_path}: {e}")
        else:
            # Create info.json if it doesn't exist? 
            # Usually it should exist from render script.
            pass

    return jsonify({"status": "success"})

@app.route('/api/finish', methods=['POST'])
def finish_processing():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    merged_data = []
    copy_count = 0
    
    # Iterate over all files
    for root, dirs, files in os.walk(RENDER_DIR):
        if 'info.json' in files:
            info_path = os.path.join(root, 'info.json')
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                if info.get('selected'):
                    # Copy object file
                    local_path = info.get('local_path')
                    if local_path and os.path.exists(local_path):
                        # Determine filename
                        custom_name = info.get('custom_name')
                        original_filename = os.path.basename(local_path)
                        filename = original_filename
                        
                        # Note: User requested NOT to rename the file, but just keep custom_name in metadata.
                        # So we keep filename as original_filename.
                        
                        # To avoid collision, check destination
                        dest_path = os.path.join(RESULT_DIR, filename)
                        
                        # Check for collision
                        if os.path.exists(dest_path):
                            # Rename strategy if collision happens
                            base, ext = os.path.splitext(filename)
                            dest_path = os.path.join(RESULT_DIR, f"{base}_{info.get('keyword', 'obj')}{ext}")
                            
                        shutil.copy2(local_path, dest_path)
                        
                        # Update path in merged json to point to new location relative to result folder
                        info['exported_filename'] = os.path.basename(dest_path)
                        merged_data.append(info)
                        copy_count += 1
                    else:
                        print(f"Warning: Object file not found for selected item: {local_path}")
            except Exception as e:
                print(f"Error processing {info_path}: {e}")

    # Save merged json
    with open(os.path.join(RESULT_DIR, 'all_selected_objects.json'), 'w') as f:
        json.dump(merged_data, f, indent=4)
        
    return jsonify({
        "status": "success", 
        "count": copy_count, 
        "result_dir": RESULT_DIR
    })

@app.route('/images/<keyword>/<obj_name>/<filename>')
def serve_image(keyword, obj_name, filename):
    return send_from_directory(os.path.join(RENDER_DIR, keyword, obj_name), filename)

if __name__ == '__main__':
    print(f"Starting server at http://localhost:5001")
    print(f"Scanning directory: {RENDER_DIR}")
    app.run(host='0.0.0.0', port=5001, debug=True)
