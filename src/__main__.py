from . import app
import logging

def create_app():
    """Create and configure the Flask application."""
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deepfake_detector.log'),
            logging.StreamHandler()
        ]
    )
    
    return app

if __name__ == '__main__':
    # Create and configure the app
    app = create_app()
    
    # Start the application
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        raise
