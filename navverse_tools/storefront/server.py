import os
from flask import Flask, request, jsonify, send_from_directory

# Configure Flask to serve the 'data' folder as static assets
# static_url_path='' makes the files in 'data' accessible at the root URL path
app = Flask(__name__, static_folder='data', static_url_path='')

@app.route('/')
def index():
    # Serve the viewer.html by default
    return send_from_directory('data', 'viewer.html')

@app.route('/api/delete', methods=['POST'])
def delete_file():
    try:
        data = request.json
        file_path = data.get('path')
        
        if not file_path:
            return jsonify({'success': False, 'error': 'No path provided'}), 400
        
        # Security check: ensure path is within data directory
        # This is a basic check.
        if '..' in file_path or file_path.startswith('/'):
            return jsonify({'success': False, 'error': 'Invalid path'}), 400

        full_path = os.path.join('data', file_path)
        
        if os.path.exists(full_path):
            os.remove(full_path)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
