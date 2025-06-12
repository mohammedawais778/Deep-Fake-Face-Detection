"""
Deep Fake Face Detection Package
"""
from flask import Flask
import os

__version__ = '0.1.0'

def create_app():
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize Flask app with template and static folders
    app = Flask(__name__, 
                template_folder=os.path.join(base_dir, 'templates'),
                static_folder=os.path.join(base_dir, 'static'))
    
    # Configure app
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    app.config['VERSION'] = __version__
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize routes
    from . import routes
    routes.init_routes(app)
    
    return app

# Create app instance
app = create_app()

# This allows running the app directly with: python -m src
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
