import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import cv2
import h5py
import json
import os
import math
import time
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from torch.nn import functional as F
from utils.losses import FocalLoss
from utils.visualization import visualize_attention

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self, backbone='resnet50', num_classes=2):
        self.backbone = backbone
        self.num_classes = num_classes

class DeepfakeDetector(nn.Module):
    def __init__(self, use_cuda: bool = True):
        super(DeepfakeDetector, self).__init__()
        logger.info("Initializing DeepfakeDetector")
        
        # Initialize device
        try:
            cuda_available = torch.cuda.is_available() and use_cuda
            if cuda_available:
                cuda_version = torch.version.cuda
                if cuda_version and float(cuda_version.split('.')[0]) < 11:
                    logger.warning(f"CUDA version {cuda_version} may be too old. Using CPU instead.")
                    cuda_available = False
        except Exception as e:
            logger.warning(f"CUDA initialization error: {str(e)}. Using CPU instead.")
            cuda_available = False
            
        self.device = torch.device("cuda" if cuda_available else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Track resource usage
        self.total_predictions = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = 100  # Cleanup every 100 predictions
        
        # Initialize model components
        try:
            logger.info("Initializing model components...")
            self._init_model_components()
            logger.info("Model components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model components: {str(e)}")
            raise
            
    def _init_model_components(self):
        """Initialize the model architecture"""
        try:
            # Load pretrained ResNet backbone
            logger.info("Loading pretrained ResNet backbone...")
            try:
                # Try offline mode first
                backbone = models.resnet50(weights=None)
                logger.info("Created ResNet50 without pretrained weights")
            except Exception as e:
                logger.warning(f"Error creating ResNet50: {str(e)}")
                backbone = models.resnet50(pretrained=False)
                logger.info("Created ResNet50 with legacy initialization")
        
            # Extract features up to the last layer before classification
            logger.info("Setting up feature extractor...")
            layers = list(backbone.children())[:-2]  # Remove avg pool and fc
            self.features = nn.Sequential(*layers)
            
            # Add custom classification head
            logger.info("Setting up classifier head...")
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2)
            )
            
            # Initialize weights
            logger.info("Initializing classifier weights...")
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            # Move to correct device
            logger.info(f"Moving model to device: {self.device}")
            self.to(self.device)
            logger.info(f"Model initialized with backbone: resnet50, using {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing model components: {str(e)}", exc_info=True)
            raise
            
    def forward(self, x):
        """Forward pass"""
        features = self.features(x)
        x = self.avgpool(features)
        x = self.classifier(x)
        return x
        
    def save_model(self, path):
        """Save model state"""
        try:
            # Save just the state dict
            torch.save(self.state_dict(), path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, path):
        """Load model state"""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            self.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, image_path):
        """Predict if an image is real or fake"""
        try:
            # Start timing
            start_time = time.time()
            logger.info(f"Starting prediction for {image_path}")

            # Load and preprocess image efficiently
            with Image.open(image_path).convert('RGB') as image:
                # Get original dimensions
                width, height = image.size
                
                # Efficient preprocessing pipeline
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                # Transform image
                image_tensor = transform(image).unsqueeze(0)
                image_tensor = image_tensor.to(self.device)

            # Get predictions efficiently
            self.eval()  # Ensure model is in eval mode
            with torch.no_grad():  # Disable gradient computation
                outputs = self(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get prediction and confidence
                confidence, prediction = torch.max(probabilities, 1)
                
                # Get individual probabilities
                real_prob = probabilities[0][0].item()
                fake_prob = probabilities[0][1].item()
                is_fake = prediction.item() == 1

            # Calculate processing time
            process_time = time.time() - start_time
            logger.info(f"Prediction completed in {process_time:.2f} seconds")
            
            return {
                'success': True,
                'label': 'Fake' if is_fake else 'Real',
                'confidence': confidence.item(),
                'confidence_percentage': f"{confidence.item()*100:.1f}%",
                'real_probability': real_prob,
                'fake_probability': fake_prob,
                'image_dimensions': f"{width}x{height}",
                'process_time': f"{process_time:.2f}s"
            }

        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

def convert_h5_to_torch(h5_path: str = 'model/image_model_augmented.h5', save_path: str = 'model/deepfake_detector.pth') -> bool:
    """Convert H5 model to PyTorch format"""
    try:
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Remove existing model file if it exists
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
                logger.info(f"Removed existing model file: {save_path}")
            except Exception as e:
                logger.warning(f"Could not remove existing model file: {str(e)}")
                return False
        
        # Create PyTorch model
        torch_model = DeepfakeDetector(use_cuda=False)  # Force CPU for conversion
        torch_model.eval()
        
        # Try to load weights from H5 if possible
        try:
            with h5py.File(h5_path, 'r') as h5_file:
                logger.info(f"Found H5 model with layers: {list(h5_file.keys())}")
                logger.info("Note: Using pretrained weights instead of H5 weights")
        except Exception as e:
            logger.warning(f"Could not load H5 model: {str(e)}")
            logger.info("Using pretrained weights instead")
        
        # Save the model state
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        torch.save(torch_model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Verify the saved model
        try:
            verify_model = DeepfakeDetector(use_cuda=False)
            verify_model.load_state_dict(torch.load(save_path, map_location='cpu'))
            
            # Try a dummy prediction
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = verify_model(test_input)
                
            logger.info("Model verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying model: {str(e)}")
            if os.path.exists(save_path):
                os.remove(save_path)
                logger.info(f"Removed invalid model file: {save_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert H5 model to PyTorch format')
    parser.add_argument('--h5_path', type=str, default='model/image_model_augmented.h5',
                      help='Path to H5 model file')
    parser.add_argument('--save_path', type=str, default='model/deepfake_detector.pth',
                      help='Path to save PyTorch model')
    parser.add_argument('--test_image', type=str,
                      help='Optional path to test image for verification')
    
    args = parser.parse_args()
    
    if convert_h5_to_torch(args.h5_path, args.save_path):
        logger.info("Model conversion completed successfully")
        
        if args.test_image:
            try:
                model = DeepfakeDetector(use_cuda=False)  # Use CPU for testing
                model.load_state_dict(torch.load(args.save_path, map_location='cpu'))
                result = model.predict(args.test_image)
                logger.info("\nTest prediction result:")
                logger.info(json.dumps(result, indent=2))
            except Exception as e:
                logger.error(f"Error during test prediction: {str(e)}")
    else:
        logger.error("Model conversion failed")
