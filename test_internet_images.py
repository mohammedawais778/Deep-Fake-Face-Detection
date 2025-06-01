import os
import requests
from urllib.parse import urlparse
from convert_to_torch import DeepfakeDetector
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_image(url: str, save_dir: str = "test_images") -> str:
    """Download an image from URL and save it locally"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get filename from URL or create one
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "test_image.jpg"
    
    save_path = os.path.join(save_dir, filename)
    
    # Download image
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded image to {save_path}")
        return save_path
    else:
        raise Exception(f"Failed to download image from {url}")

def test_image(model_path: str, image_path: str) -> dict:
    """Test an image with the deepfake detector model"""
    try:
        logger.info("Initializing model...")
        model = DeepfakeDetector(use_cuda=False)  # Use CPU for testing
        
        logger.info(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        logger.info("Making prediction...")
        result = model.predict(image_path)
        logger.info("Prediction completed successfully")
        
        return result
    except Exception as e:
        logger.error(f"Error in test_image: {str(e)}")
        raise

def main():
    model_path = 'model/deepfake_detector.pth'
      # Test local images
    test_images = [
        "uploads/fake_183_253_4.jpg",  # Known fake
        "uploads/real_183_5.jpg",      # Known real
        "uploads/ai_2.jpeg",           # AI generated
        "uploads/Mark_Zuckerberg_at_the_37th_G8_Summit_in_Deauville_018_v1.jpg"  # Real photo
    ]
    
    for image_path in test_images:
        try:
            logger.info(f"\nTesting image: {image_path}")
            result = test_image(model_path, image_path)
            logger.info(f"Prediction result: {result}")
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    main()
