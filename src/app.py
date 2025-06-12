import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image as PILImage
from datetime import datetime
from functools import wraps
from flask import Flask, jsonify, request, current_app, send_from_directory, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('../static', path)
import cv2
import numpy as np

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create file handler which logs even debug messages
    log_file = os.path.join(log_dir, 'app.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Disable overly verbose loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging()
logger.info("Logging initialized")

# Image transformation for model input
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

class APIError(Exception):
    """Base exception for API errors"""
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['status'] = 'error'
        rv['code'] = self.status_code
        return rv

def handle_api_error(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except APIError as e:
            response = jsonify(e.to_dict())
            response.status_code = e.status_code
            return response
        except Exception as e:
            logger.exception("Unexpected error occurred")
            response = jsonify({
                'status': 'error',
                'code': 500,
                'message': 'An unexpected error occurred',
                'details': str(e)
            })
            response.status_code = 500
            return response
    return decorated_function

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
UPLOAD_FOLDER = 'uploads'

# Import PyTorch for model loading
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image as PILImage
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep learning features will be disabled.")

# ResNet-based model for deepfake detection
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()
        # Simple model with adaptive pooling to handle different input sizes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Fixed size output
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)  # Adjusted for 7x7 output
        self.sigmoid = nn.Sigmoid()
        
        # Debug info
        self.debug_info = {}
    
    def forward(self, x):
        self.debug_info = {}
        self.debug_info['input_shape'] = str(x.shape)
        self.debug_info['input_dtype'] = str(x.dtype)
        self.debug_info['input_range'] = f"[{x.min().item():.4f}, {x.max().item():.4f}]"
        
        try:
            # Log input details
            logger.info(f"Model input - Shape: {x.shape}, Type: {x.dtype}, Device: {x.device}")
            logger.info(f"Input range: {self.debug_info['input_range']}")
            
            # First conv layer
            x = self.conv1(x)
            self.debug_info['after_conv1'] = str(x.shape)
            logger.info(f"After conv1: {x.shape}")
            
            # Activation
            x = torch.relu(x)
            
            # Pooling
            x = self.pool(x)
            self.debug_info['after_pool'] = str(x.shape)
            logger.info(f"After pool: {x.shape}")
            
            # Adaptive pooling to fixed size
            x = self.adaptive_pool(x)
            self.debug_info['after_adaptive_pool'] = str(x.shape)
            logger.info(f"After adaptive pool: {x.shape}")
            
            # Flatten
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            self.debug_info['after_flatten'] = str(x.shape)
            logger.info(f"After flatten: {x.shape}")
            
            # Check if dimensions match
            if x.shape[1] != 16 * 7 * 7:
                error_msg = f"Flattened dimension {x.shape[1]} does not match expected {16 * 7 * 7}"
                logger.error(error_msg)
                logger.error(f"Debug info: {self.debug_info}")
                raise ValueError(error_msg)
            
            # Fully connected layer
            x = self.fc1(x)
            self.debug_info['after_fc1'] = str(x.shape)
            logger.info(f"After fc1: {x.shape}")
            
            # Sigmoid activation
            x = self.sigmoid(x)
            self.debug_info['after_sigmoid'] = str(x.shape)
            logger.info(f"After sigmoid: {x.shape}")
            
            # Verify output
            if torch.isnan(x).any() or torch.isinf(x).any():
                error_msg = f"Output contains invalid values (NaN or Inf)"
                logger.error(error_msg)
                logger.error(f"Output: {x}")
                raise ValueError(error_msg)
            
            return x
            
        except Exception as e:
            logger.error(f"Error in model forward pass: {str(e)}")
            logger.error(f"Debug info: {self.debug_info}")
            if 'x' in locals():
                logger.error(f"Current tensor shape: {x.shape if hasattr(x, 'shape') else 'No shape'}")
                logger.error(f"Current tensor device: {x.device if hasattr(x, 'device') else 'No device'}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Add the detect_file route
@app.route('/api/detect', methods=['POST'])
@handle_api_error
def detect_file():
    """
    Handle file upload and detection.
    
    Returns:
        JSON response with detection results or error message
    """
    logger.info("-" * 80)
    logger.info("Received new detection request")
    
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            error_msg = 'No file part in the request'
            logger.error(error_msg)
            raise APIError(error_msg, status_code=400)
            
        file = request.files['file']
        logger.info(f"Processing file: {file.filename}")
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error_msg = 'No selected file'
            logger.error(error_msg)
            raise APIError(error_msg, status_code=400)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to temporary location: {filepath}")
            
            try:
                # Ensure upload directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save the file temporarily
                file.save(filepath)
                logger.info(f"File saved successfully: {os.path.exists(filepath)}, size: {os.path.getsize(filepath)} bytes")
                
                try:
                    # Process the file based on its type
                    logger.info(f"Starting to process file: {filepath}")
                    
                    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        logger.info("Processing as image file")
                        result = process_image(filepath, filename, os.path.getsize(filepath))
                    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                        logger.info("Processing as video file")
                        result = process_video(filepath, filename, os.path.getsize(filepath))
                    else:
                        error_msg = 'Unsupported file type'
                        logger.error(error_msg)
                        raise APIError(error_msg, status_code=400)
                    
                    logger.info("File processed successfully")
                    return jsonify({
                        'status': 'success',
                        'result': result
                    })
                    
                except Exception as process_error:
                    logger.error(f"Error processing file: {str(process_error)}", exc_info=True)
                    if hasattr(process_error, 'args') and len(process_error.args) > 0:
                        logger.error(f"Error details: {process_error.args}")
                    raise RuntimeError(f"Error processing file: {str(process_error)}")
                    
                finally:
                    # Clean up the temporary file
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            logger.info(f"Temporary file removed: {filepath}")
                    except Exception as cleanup_error:
                        logger.warning(f"Could not remove temporary file {filepath}: {str(cleanup_error)}")
                
            except Exception as save_error:
                logger.error(f"Error saving file: {str(save_error)}", exc_info=True)
                raise RuntimeError(f"Could not save uploaded file: {str(save_error)}")
                
        else:
            error_msg = f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            logger.error(error_msg)
            raise APIError(error_msg, status_code=400)
            
    except APIError as api_error:
        # Re-raise API errors as they are already properly formatted
        logger.error(f"API Error: {str(api_error)}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in detect_file: {str(e)}", exc_info=True)
        error_type = type(e).__name__
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(f"Error type: {error_type}, Message: {error_msg}")
        raise RuntimeError(f"An error occurred while processing your request: {str(e)}")
    
    finally:
        logger.info("-" * 80 + "\n")

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepfakeDetector, self).__init__()
        self.model = SimpleCNN(num_classes)
    
    def forward(self, x):
        return self.model(x)

# Image transformation for model input
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load PyTorch model
def load_pytorch_model():
    """Initialize and return a PyTorch model for deepfake detection."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available. Please install torch and torchvision.")
    
    # Initialize the model
    model = DeepfakeDetector()
    model.eval()
    
    return model

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    'image': os.path.join(BASE_DIR, 'models', 'model', 'image_model.h5'),
    'video': os.path.join(BASE_DIR, 'models', 'model', 'lstm_model.h5')
}

def validate_file(file):
    """Validate the uploaded file."""
    if not file or file.filename == '':
        raise APIError('No file selected', status_code=400)
    
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
        raise APIError(
            f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
            status_code=400
        )
    
    # Check file size by reading the content length
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > MAX_CONTENT_LENGTH:
        raise APIError(
            f'File too large. Maximum size is {MAX_CONTENT_LENGTH / (1024*1024)}MB',
            status_code=400
        )
    
    return True

@handle_api_error
def detect_file():
    """
    Handle file upload and detection.
    
    Returns:
        JSON response with detection results or error message
    """
    logger.info("=== Starting file upload processing ===")
    
    # Check if file was included in the request
    if 'file' not in request.files:
        error_msg = 'No file part in the request'
        logger.error(error_msg)
        raise APIError(error_msg, status_code=400)
    
    file = request.files['file']
    logger.info(f"Received file: {file.filename}")
    logger.info(f"File content type: {file.content_type}")
    logger.info(f"File content length: {file.content_length if file.content_length else 'Not provided'}")
    
    try:
        # Validate the uploaded file
        logger.info("Validating file...")
        validate_file(file)
        
        # Generate a secure filename and create upload directory
        filename = secure_filename(file.filename)
        logger.info(f"Secured filename: {filename}")
        
        # Create a temporary file to store the uploaded file
        import tempfile
        _, temp_path = tempfile.mkstemp()
        file.save(temp_path)
        file_size = os.path.getsize(temp_path)
        
        try:
            # Process the file based on its type
            if file.content_type.startswith('image/'):
                result = process_image(temp_path, filename, file_size)
            elif file.content_type.startswith('video/'):
                result = process_video(temp_path, filename, file_size)
            else:
                raise APIError('Unsupported file type', status_code=400)
                
            return jsonify({
                'status': 'success',
                'code': 200,
                'data': result
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            raise APIError(f"Error processing file: {str(e)}", status_code=500)
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
                logger.debug(f"Temporary file {temp_path} deleted")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")
                
    except Exception as e:
        logger.error(f"Unexpected error in detect_file: {str(e)}", exc_info=True)
        raise APIError(f"An unexpected error occurred: {str(e)}", status_code=500)
            
    except APIError:
        raise  # Re-raise APIError
        raise APIError("An unexpected error occurred", status_code=500)
        
    finally:
        # Clean up the uploaded file if it exists
        if 'filepath' in locals() and filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {str(e)}")

def process_video(filepath, filename, file_size):
    """Process a video file for deepfake detection using frame-by-frame analysis."""
    try:
        logger.info(f"Processing video: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        # Check if OpenCV is available
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise ImportError("OpenCV is required for video processing. Please install it with: pip install opencv-python-headless")
        
        # Initialize video capture
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {filepath}")
        
        try:
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Video properties - Frames: {frame_count}, FPS: {fps}, Duration: {duration:.2f}s")
            
            # Initialize model for frame processing
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DeepfakeDetector().to(device)
            model.eval()
            
            # Process frames (sample every second to save computation)
            frame_skip = int(fps) if fps > 0 else 1
            predictions = []
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames to process ~1 frame per second
                if processed_frames % frame_skip != 0:
                    processed_frames += 1
                    continue
                
                try:
                    # Convert BGR to RGB and resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = PILImage.fromarray(frame_rgb)
                    
                    # Preprocess and predict
                    img_tensor = transform(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(img_tensor)
                        pred = output.item()
                        predictions.append(pred)
                    
                    processed_frames += 1
                    
                except Exception as frame_error:
                    logger.warning(f"Error processing frame {processed_frames}: {str(frame_error)}")
                    continue
            
            # Calculate average prediction
            if not predictions:
                raise ValueError("No frames were successfully processed")
                
            avg_prediction = sum(predictions) / len(predictions)
            confidence = avg_prediction * 100
            
            result = {
                'filename': filename,
                'is_fake': avg_prediction > 0.5,
                'confidence': confidence,
                'file_type': 'video',
                'file_size': file_size,
                'model_used': 'ResNet18 with custom head',
                'duration': duration,
                'frames_processed': len(predictions)
            }
            
            logger.info(f"Processed video with average confidence: {confidence:.2f}%")
            return result
            
        finally:
            cap.release()
            
    except Exception as e:
        logger.error(f"Error processing video {filename}: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error processing video: {str(e)}")

def process_image(filepath, filename, file_size):
    """Process an image file for deepfake detection using PyTorch."""
    try:
        logger.info(f"Processing image: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please install it with: pip install torch torchvision")
        
        # Load and preprocess the image
        logger.info("Loading and preprocessing image...")
        try:
            # Open image
            logger.info(f"Opening image from: {filepath}")
            image = PILImage.open(filepath)
            logger.info(f"Original image size: {image.size}, mode: {image.mode}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Debug: Save the loaded image
            debug_path = os.path.join(os.path.dirname(filepath), 'debug_input.jpg')
            image.save(debug_path)
            logger.info(f"Saved debug input image to: {debug_path}")
            
            # Apply transformations step by step for better debugging
            logger.info("Applying transformations...")
            
            # Resize
            resize = transforms.Resize((256, 256))
            image_resized = resize(image)
            logger.info(f"After resize: {image_resized.size}")
            
            # Center crop
            crop = transforms.CenterCrop(224)
            image_cropped = crop(image_resized)
            logger.info(f"After crop: {image_cropped.size}")
            
            # Convert to tensor
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(image_cropped)
            logger.info(f"After to_tensor: shape={image_tensor.shape}, dtype={image_tensor.dtype}, range=[{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
            
            # Normalize
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
            image_tensor = normalize(image_tensor)
            logger.info(f"After normalize: shape={image_tensor.shape}, dtype={image_tensor.dtype}, range=[{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            logger.info(f"Final tensor shape with batch: {image_tensor.shape}")
            
        except Exception as img_error:
            logger.error(f"Error loading image: {str(img_error)}", exc_info=True)
            if 'image' in locals():
                logger.error(f"Image info - size: {getattr(image, 'size', 'N/A')}, mode: {getattr(image, 'mode', 'N/A')}")
            raise ValueError(f"Could not process image: {str(img_error)}")
        
        # Initialize model and make prediction
        logger.info("Initializing model and making prediction...")
        try:
            # Set up device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Debug: Check PyTorch version and CUDA availability
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            
            # Initialize the model
            logger.info("Initializing model...")
            try:
                model = DeepfakeDetector()
                logger.info("Model initialized successfully")
                
                # Move model to device
                model = model.to(device)
                model.eval()
                logger.info("Model moved to device and set to eval mode")
                
            except Exception as model_init_error:
                logger.error(f"Error initializing model: {str(model_init_error)}", exc_info=True)
                raise RuntimeError(f"Failed to initialize model: {str(model_init_error)}")
            
            # Debug: Print model architecture and parameters
            logger.info(f"Model architecture:\n{model}")
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total trainable parameters: {total_params:,}")
            
            # Verify tensor properties before model input
            logger.info("Verifying input tensor...")
            logger.info(f"Input tensor shape: {image_tensor.shape}")
            logger.info(f"Input tensor dtype: {image_tensor.dtype}")
            logger.info(f"Input tensor range: {image_tensor.min().item():.4f} to {image_tensor.max().item():.4f}")
            logger.info(f"Input tensor mean: {image_tensor.mean().item():.4f}, std: {image_tensor.std().item():.4f}")
            
            # Ensure tensor is float32
            if image_tensor.dtype != torch.float32:
                logger.warning(f"Converting tensor from {image_tensor.dtype} to float32")
                image_tensor = image_tensor.float()
            
            # Move image tensor to the same device as model
            logger.info(f"Moving tensor to device: {device}")
            image_tensor = image_tensor.to(device)
            logger.info("Tensor moved to device successfully")
            
            try:
                logger.info("-"*40)
                logger.info("Starting model inference...")
                
                # Debug: Print model architecture and parameters
                logger.info("Model architecture:")
                for name, param in model.named_parameters():
                    logger.info(f"  {name}: {param.shape} (requires_grad={param.requires_grad})")
                
                # Debug: Print input tensor details
                logger.info("Input tensor details before model:")
                logger.info(f"  Shape: {image_tensor.shape}")
                logger.info(f"  Dtype: {image_tensor.dtype}")
                logger.info(f"  Device: {image_tensor.device}")
                logger.info(f"  Min: {image_tensor.min().item():.6f}, Max: {image_tensor.max().item():.6f}")
                logger.info(f"  Mean: {image_tensor.mean().item():.6f}, Std: {image_tensor.std().item():6f}")
                
                # Debug: Save input tensor as image
                try:
                    debug_dir = os.path.join(os.path.dirname(filepath), 'debug')
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_tensor = image_tensor.squeeze(0).cpu()
                    debug_tensor = debug_tensor.permute(1, 2, 0).numpy()
                    debug_tensor = (debug_tensor * 255).astype('uint8')
                    debug_img_path = os.path.join(debug_dir, 'debug_model_input.jpg')
                    cv2.imwrite(debug_img_path, cv2.cvtColor(debug_tensor, cv2.COLOR_RGB2BGR))
                    logger.info(f"Saved debug model input to: {debug_img_path}")
                except Exception as img_save_err:
                    logger.warning(f"Could not save debug model input: {str(img_save_err)}")
                
                # Run model inference
                with torch.no_grad():
                    try:
                        logger.info("Running model forward pass...")
                        output = model(image_tensor)
                        logger.info(f"Model forward pass completed successfully")
                        logger.info(f"Output shape: {output.shape}")
                        logger.info(f"Output values: {output}")
                        
                        # Check if output contains valid values
                        if torch.isnan(output).any():
                            logger.error("Model output contains NaN values")
                            raise ValueError("Model output contains NaN values")
                            
                        if torch.isinf(output).any():
                            logger.error("Model output contains Inf values")
                            raise ValueError("Model output contains Inf values")
                            
                        prediction = output.item()
                        logger.info(f"Raw prediction: {prediction:.6f}")
                        
                        # Convert prediction to confidence score (0-100%)
                        confidence = prediction * 100
                        logger.info(f"Prediction completed. Confidence: {confidence:.2f}%")
                        
                    except Exception as forward_error:
                        logger.error(f"Error in model forward pass: {str(forward_error)}", exc_info=True)
                        logger.error(f"Input tensor shape: {image_tensor.shape}")
                        logger.error(f"Input tensor device: {image_tensor.device}")
                        logger.error(f"Model device: {next(model.parameters()).device}")
                        raise
                
            except Exception as inference_error:
                logger.error("-"*40)
                logger.error("MODEL INFERENCE FAILED")
                logger.error("-"*40)
                logger.error(f"Error type: {type(inference_error).__name__}")
                logger.error(f"Error message: {str(inference_error)}")
                logger.error("\nDetailed traceback:")
                import traceback
                logger.error(traceback.format_exc())
                logger.error("-"*40)
                
                # Additional debug info
                logger.error("Model state:")
                logger.error(f"  Model type: {type(model).__name__}")
                logger.error(f"  Model on CUDA: {next(model.parameters()).is_cuda}")
                logger.error(f"  Input tensor shape: {image_tensor.shape}")
                logger.error(f"  Input tensor device: {image_tensor.device}")
                logger.error(f"  Input tensor dtype: {image_tensor.dtype}")
                logger.error(f"  Input tensor range: [{image_tensor.min().item():.6f}, {image_tensor.max().item():.6f}]")
                
                # Try to run a simple tensor operation to check for CUDA errors
                try:
                    test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
                    test_result = test_tensor * 2
                    logger.error(f"CUDA test passed: {test_result}")
                except Exception as cuda_test_err:
                    logger.error(f"CUDA test failed: {str(cuda_test_err)}")
                
                raise RuntimeError(f"Model inference failed: {str(inference_error)}")
            
            # Prepare result
            result = {
                'filename': filename,
                'is_fake': prediction > 0.5,  # Threshold at 0.5
                'confidence': confidence,
                'file_type': 'image',
                'file_size': file_size,
                'model_used': 'ResNet18 with custom head'
            }
            
            logger.info("Image processing completed successfully")
            return result
            
        except Exception as model_error:
            logger.error(f"Error during model prediction: {str(model_error)}", exc_info=True)
            raise RuntimeError(f"Error during model prediction: {str(model_error)}")
            
    except Exception as e:
        logger.error(f"Error processing image {filename}: {str(e)}", exc_info=True)
        raise
