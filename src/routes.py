from flask import render_template, jsonify, send_from_directory
import os

def init_routes(app):
    """Initialize routes for the Flask app."""
    from .app import allowed_file, detect_file
    
    @app.route('/')
    def home():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/api/detect', methods=['POST'])
    def detect():
        """Handle file upload and detection."""
        return detect_file()
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files."""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return send_from_directory(os.path.join(root_dir, 'static'), filename)
    
    @app.route('/api/health')
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'ok',
            'version': app.config.get('VERSION', '0.1.0')
        })
