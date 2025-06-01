import os
import psutil
import logging
import signal
import sys
from datetime import datetime
import threading
from typing import Dict, List, Optional, Tuple
import time
from functools import wraps
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory, render_template, abort
import shutil
import cv2
from PIL import Image

# Import required modules
import torch
from convert_to_torch import DeepfakeDetector
TORCH_AVAILABLE = True

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d',
    handlers=[
        logging.FileHandler('deepfake_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):  # 1GB threshold
        self.threshold_mb = threshold_mb
        self.lock = threading.Lock()
        self.monitoring = False
        
    def get_memory_usage(self) -> Dict[str, float]:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # RSS in MB
            'vms_mb': memory_info.vms / (1024 * 1024),  # VMS in MB
            'percent': process.memory_percent()
        }
        
    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            threading.Thread(target=self._monitor_memory, daemon=True).start()
            
    def _monitor_memory(self):
        while self.monitoring:
            usage = self.get_memory_usage()
            if usage['rss_mb'] > self.threshold_mb:
                logger.warning(f"High memory usage: {usage['rss_mb']:.2f}MB RSS")
                if hasattr(model_manager, 'model'):
                    model_manager.unload_model()  # Free up memory
            time.sleep(60)  # Check every minute
            
memory_monitor = MemoryMonitor()

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self.lock = threading.Lock()

    def is_allowed(self, key):
        now = time.time()
        with self.lock:
            # Clean old entries
            self.requests = {k: v for k, v in self.requests.items() 
                           if now - v[-1] < self.window_seconds}
            
            if key not in self.requests:
                self.requests[key] = []
            
            self.requests[key] = [t for t in self.requests[key] 
                                if now - t < self.window_seconds]
            
            if len(self.requests[key]) >= self.max_requests:
                return False
            
            self.requests[key].append(now)
            return True

rate_limiter = RateLimiter()

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not rate_limiter.is_allowed(request.remote_addr):
            abort(429, description="Too many requests")
        return f(*args, **kwargs)
    return decorated_function

# Model management with caching and versioning
class ModelManager:
    def __init__(self):
        self.model = None
        self.model_hash = None
        self.last_load_time = None
        self.lock = threading.Lock()
        self.last_used = time.time()
        self.load_timeout = 30  # seconds
        logger.info("ModelManager initialized")

    def get_model_hash(self, model_path):
        """Get model file hash for version checking"""
        import hashlib
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        return None

    def unload_model(self):
        """Unload model to free up memory"""
        with self.lock:
            if self.model is not None:
                logger.info("Unloading model to free memory")
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def load_model(self, force=False):
        """Thread-safe model loading with caching and timeout"""
        logger.info("Entering load_model function")
        start_time = time.time()
        
        def check_timeout():
            if time.time() - start_time > self.load_timeout:
                raise TimeoutError("Model loading timed out")
        
        with self.lock:
            try:
                model_path = os.path.join("model", "deepfake_detector.pth")
                logger.info(f"Checking model at path: {model_path}")
                check_timeout()
                
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found at {model_path}")
                    return None
                    
                current_hash = self.get_model_hash(model_path)
                logger.info(f"Current model hash: {current_hash}")
                check_timeout()

                # Check if model needs to be reloaded
                if not force and self.model is not None and self.model_hash == current_hash:
                    self.last_used = time.time()
                    logger.info("Using cached model")
                    return self.model

                # Unload existing model if any
                self.unload_model()
                check_timeout()

                try:
                    # Step 1: Create model instance
                    logger.info("Step 1: Creating new model instance...")
                    # Try to create model in offline mode first
                    os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'model', 'torch_home')
                    os.environ['TORCH_OFFLINE'] = '1'
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable Metal fallback on macOS
                    
                    model = DeepfakeDetector(use_cuda=False)  # Force CPU mode for stability
                    logger.info("Model instance created successfully")
                    check_timeout()
                    
                    # Step 2: Load state dict
                    logger.info("Step 2: Loading state dict...")
                    try:
                        state_dict = torch.load(model_path, map_location='cpu')
                    except Exception as e:
                        logger.error(f"Error loading state dict: {str(e)}")
                        raise
                    logger.info("State dict loaded successfully")
                    check_timeout()
                    
                    # Step 3: Apply state dict
                    logger.info("Step 3: Applying state dict to model...")
                    try:
                        model.load_state_dict(state_dict)
                    except Exception as e:
                        logger.error(f"Error applying state dict: {str(e)}")
                        raise
                    logger.info("State dict applied successfully")
                    check_timeout()
                    
                    # Step 4: Set eval mode
                    logger.info("Step 4: Setting model to eval mode...")
                    model.eval()
                    logger.info("Model set to eval mode")
                    check_timeout()
                    
                    # Step 5: Test with dummy input
                    logger.info("Step 5: Testing model with dummy input...")
                    dummy_input = torch.randn(1, 3, 224, 224)
                    with torch.no_grad():
                        _ = model(dummy_input)
                    logger.info("Model test successful")
                    check_timeout()

                    # Step 6: Update model state
                    self.model = model
                    self.model_hash = current_hash
                    self.last_load_time = time.time()
                    self.last_used = time.time()
                    
                    memory_usage = memory_monitor.get_memory_usage()
                    logger.info(f"Model loaded successfully (Version: {current_hash[:8]})")
                    logger.info(f"Current memory usage: {memory_usage['rss_mb']:.2f}MB RSS")
                    logger.info(f"Total load time: {time.time() - start_time:.2f}s")
                    
                    return model
                    
                except TimeoutError:
                    logger.error("Model loading timed out")
                    if hasattr(self, 'model') and self.model is not None:
                        self.unload_model()
                    return None
                except Exception as e:
                    logger.error(f"Error during model loading: {str(e)}", exc_info=True)
                    if hasattr(self, 'model') and self.model is not None:
                        self.unload_model()
                    return None
                    
            except Exception as e:
                logger.error(f"Error in load_model: {str(e)}", exc_info=True)
                return None

    def get_model(self):
        """Get the current model, loading it if necessary"""
        if self.model is None:
            return self.load_model(force=True)
        return self.model

model_manager = ModelManager()

# Try to import TensorFlow components
try:
    from predict import predict_image as predict_tf, predict_video
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available")
    TF_AVAILABLE = False

if not (TORCH_AVAILABLE or TF_AVAILABLE):
    raise ImportError("Neither PyTorch nor TensorFlow models could be loaded")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Configure flask app
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Load the frontend HTML from templates folder
@app.route('/')
def home():
    try:
        # Try to render the template from the templates folder
        return render_template('html_frontend.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Error loading application interface',
            'details': str(e)
        }), 500

@app.route('/api/health')
def health():
    # Simple health check for frontend
    try:
        # Try getting the model
        model = model_manager.get_model()
        if model is None:
            logger.error("Could not load model in health check")
            return jsonify({'model_loaded': False})

        # Try a dummy prediction to check model
        test_img = None
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            if f.lower().endswith(('jpg', 'jpeg', 'png')):
                test_img = os.path.join(app.config['UPLOAD_FOLDER'], f)
                break

        if test_img and TORCH_AVAILABLE:
            logger.info(f"Running test prediction on {test_img}")
            result = model.predict(test_img)
            model_loaded = result['success']
        else:
            model_loaded = model is not None  # Consider model loaded if we got it from manager

        return jsonify({'model_loaded': model_loaded})
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'model_loaded': False})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/detect', methods=['POST'])
@rate_limit
def detect_file():
    try:
        logger.info(f"Received file upload request from {request.remote_addr}")
        
        if 'file' not in request.files:
            logger.error("No file found in request")
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename received")
            return jsonify({'success': False, 'error': 'No file selected'})
            
        if not allowed_file(file.filename):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            })
        
        # Secure the filename and create upload path
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        try:
            file.save(filepath)
            logger.info(f"File saved successfully: {filepath}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'error': 'Error saving uploaded file'})
        
        # Get the model and make prediction
        try:
            model = model_manager.get_model()
            if model is None:
                raise RuntimeError("Could not load model")
            
            result = model.predict(filepath)
            
            # Clean up the uploaded file
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file: {str(e)}")
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Error processing image',
                'details': str(e)
            })
            
    except Exception as e:
        logger.error(f"Unhandled error in detect_file: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/batch-status/<batch_id>')
def batch_status(batch_id):
    """Check status of a batch processing job"""
    result = batch_processor.get_batch_result(batch_id)
    if result is None:
        return jsonify({
            'success': False,
            'error': 'Batch not found'
        })
    return jsonify({
        'success': True,
        'completed': result['completed'],
        'results': result.get('results')
    })

# Constants
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
MAX_BATCH_SIZE = 10  # Maximum number of files in a batch
CLEANUP_INTERVAL = 3600  # 1 hour

def cleanup_old_files():
    """Clean up old files from the uploads directory"""
    while True:
        try:
            now = time.time()
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                # Skip if file is less than 1 hour old
                if now - os.path.getmtime(filepath) < CLEANUP_INTERVAL:
                    continue
                try:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filename}")
                except Exception as e:
                    logger.error(f"Error removing file {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in cleanup thread: {str(e)}")
        time.sleep(CLEANUP_INTERVAL)  # Check every hour

def cleanup_old_files():
    """Clean up old files from the uploads directory"""
    while True:
        try:
            now = time.time()
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                # Skip if file is less than 1 hour old
                if now - os.path.getmtime(filepath) < CLEANUP_INTERVAL:
                    continue
                try:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filename}")
                except Exception as e:
                    logger.error(f"Error removing file {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in cleanup thread: {str(e)}")
        time.sleep(CLEANUP_INTERVAL)  # Check every hour

class BatchProcessor:
    def __init__(self):
        self.batch_results = {}
        self.lock = threading.Lock()
        
    def process_files(self, files: List[Tuple[str, str]]) -> str:
        """
        Process a batch of files asynchronously
        Args:
            files: List of (filepath, filetype) tuples
        Returns:
            batch_id: Unique identifier for this batch
        """
        batch_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + str(hash(str(files)))[:8]
        
        def process_batch():
            results = []
            for filepath, filetype in files:
                try:
                    if filetype.startswith('image'):
                        if TORCH_AVAILABLE and model_manager.get_model() is not None:
                            result = predict_torch(model_manager.get_model(), filepath)
                        elif TF_AVAILABLE:
                            result = predict_tf(filepath)
                        else:
                            result = {'success': False, 'error': 'No models available'}
                    elif filetype.startswith('video') and TF_AVAILABLE:
                        result = predict_video(filepath)
                    else:
                        result = {'success': False, 'error': 'Unsupported file type'}
                    
                    if result['success']:
                        # Add file metadata
                        file_size = os.path.getsize(filepath)
                        image_dims = self._get_file_dimensions(filepath, filetype)
                        result.update({
                            'file_path': os.path.basename(filepath),
                            'file_size': file_size,
                            'image_dimensions': image_dims
                        })
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {str(e)}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'file_path': os.path.basename(filepath)
                    })
            
            with self.lock:
                self.batch_results[batch_id] = {
                    'completed': True,
                    'results': results,
                    'timestamp': time.time()
                }
        
        with self.lock:
            self.batch_results[batch_id] = {
                'completed': False,
                'timestamp': time.time()
            }
        
        threading.Thread(target=process_batch).start()
        return batch_id
    
    def get_batch_result(self, batch_id: str) -> Optional[Dict]:
        """Get results for a batch if available"""
        with self.lock:
            return self.batch_results.get(batch_id)
    
    def cleanup_old_results(self):
        """Clean up results older than 1 hour"""
        with self.lock:
            now = time.time()
            self.batch_results = {
                k: v for k, v in self.batch_results.items()
                if now - v['timestamp'] < CLEANUP_INTERVAL
            }
    
    def _get_file_dimensions(self, filepath: str, filetype: str) -> str:
        try:
            if filetype.startswith('image'):
                with Image.open(filepath) as img:
                    return f"{img.width}x{img.height}"
            elif filetype.startswith('video'):
                cap = cv2.VideoCapture(filepath)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    return f"{w}x{h}"
        except Exception as e:
            logger.error(f"Error getting dimensions for {filepath}: {str(e)}")
        return "?x?"

batch_processor = BatchProcessor()

def cleanup_old_files():
    """Clean up old files from the uploads directory"""
    while True:
        try:
            now = time.time()
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                # Skip if file is less than 1 hour old
                if now - os.path.getmtime(filepath) < CLEANUP_INTERVAL:
                    continue
                try:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filename}")
                except Exception as e:
                    logger.error(f"Error removing file {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in cleanup thread: {str(e)}")
        time.sleep(CLEANUP_INTERVAL)  # Check every hour

# Add request logging middleware
@app.before_request
def log_request_info():
    logger.info('Headers: %s', dict(request.headers))
    logger.info('Body: %s', request.get_data())

def init_upload_dir():
    """Initialize and clean the upload directory"""
    logger.info("Initializing upload directory...")
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
            logger.info(f"Created upload directory at {UPLOAD_FOLDER}")
        else:
            # Clean existing files
            for filename in os.listdir(UPLOAD_FOLDER):
                try:
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove file {filename}: {str(e)}")
            logger.info("Cleaned existing upload directory")
    except Exception as e:
        logger.error(f"Error initializing upload directory: {str(e)}", exc_info=True)
        raise

# Initialize required directories and model on startup
def init_app():
    """Initialize the application"""
    try:
        logger.info("Starting application initialization...")
        
        # Step 1: Check directories
        logger.info("Step 1: Checking directories...")
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        logger.info(f"Upload directory confirmed: {UPLOAD_FOLDER}")
        
        if not os.path.exists('templates'):
            logger.error("Templates directory not found")
            raise RuntimeError("Templates directory not found")
        logger.info("Templates directory found")
        
        # Step 2: Check model file
        logger.info("Step 2: Checking model file...")
        model_path = os.path.join("model", "deepfake_detector.pth")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        logger.info(f"Found model file at: {model_path}")
        
        # Step 3: Verify PyTorch is working
        logger.info("Step 3: Verifying PyTorch...")
        try:
            dummy_tensor = torch.randn(1, 3, 224, 224)
            logger.info("PyTorch tensor creation successful")
        except Exception as e:
            logger.error(f"PyTorch verification failed: {str(e)}")
            return False
        
        # Step 4: Create model instance without loading weights
        logger.info("Step 4: Creating model instance...")
        try:
            model = DeepfakeDetector(use_cuda=False)
            logger.info("Model instance created successfully")
        except Exception as e:
            logger.error(f"Error creating model instance: {str(e)}")
            return False
        
        # Step 5: Load model weights
        logger.info("Step 5: Loading model weights...")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            return False
        
        # Step 6: Test model with dummy input
        logger.info("Step 6: Testing model...")
        try:
            with torch.no_grad():
                output = model(dummy_tensor)
            logger.info("Model test successful")
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            return False
            
        logger.info("Application initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}", exc_info=True)
        return False

def start_background_tasks():
    """Start background monitoring and cleanup tasks"""
    logger.info("Starting background tasks...")
    
    # Start memory monitoring
    memory_monitor.start_monitoring()
    logger.info("Memory monitoring started")
    
    # Start file cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    logger.info("File cleanup thread started")
    
    # Start batch result cleanup
    def cleanup_batch_results():
        while True:
            batch_processor.cleanup_old_results()
            time.sleep(CLEANUP_INTERVAL)
    
    batch_cleanup_thread = threading.Thread(target=cleanup_batch_results, daemon=True)
    batch_cleanup_thread.start()
    logger.info("Batch result cleanup thread started")

# Initialize the app before running
if not init_app():
    logger.error("Application initialization failed!")

def shutdown_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received shutdown signal {signum}")
    try:
        # Cleanup tasks
        logger.info("Running cleanup tasks...")
        
        # Unload model to free memory
        if hasattr(model_manager, 'model') and model_manager.model is not None:
            logger.info("Unloading model...")
            model_manager.unload_model()
        
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                try:
                    os.remove(os.path.join(UPLOAD_FOLDER, filename))
                except Exception as e:
                    logger.error(f"Error removing file {filename}: {str(e)}")
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
        
        logger.info("Cleanup completed, shutting down...")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    import signal
    import sys
    
    logger.info("Starting application in main thread...")
    
    # Initialize model first
    logger.info("Initializing model...")
    try:
        # Force offline mode
        os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'model', 'torch_home')
        os.environ['TORCH_OFFLINE'] = '1'
        
        # Initialize model
        model = DeepfakeDetector(use_cuda=False)
        model_path = os.path.join("model", "deepfake_detector.pth")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Test the model
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = model(dummy_input)
            
        logger.info("Model initialization successful")
        
        # Store in model manager
        model_manager.model = model
        model_manager.model_hash = model_manager.get_model_hash(model_path)
        model_manager.last_load_time = time.time()
        model_manager.last_used = time.time()
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        sys.exit(1)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Create required directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize background tasks
    start_background_tasks()
    
    # Get initial memory usage
    initial_memory = memory_monitor.get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory['rss_mb']:.2f}MB RSS")
    
    # Start the Flask app
    app.run(debug=False, port=8000)
