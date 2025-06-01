import os
from convert_to_torch import DeepfakeDetector
import torch
import logging
import shutil
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_images():
    """Create some test images with different patterns"""
    os.makedirs('test_images', exist_ok=True)
    
    # Create a pure white image
    white_img = Image.new('RGB', (224, 224), color='white')
    white_img.save('test_images/white.jpg')
    
    # Create a pure black image
    black_img = Image.new('RGB', (224, 224), color='black')
    black_img.save('test_images/black.jpg')
    
    # Create a random noise image
    random_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    random_img.save('test_images/random.jpg')
    
    # Create a gradient image
    gradient = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        gradient[i, :, :] = i
    gradient_img = Image.fromarray(gradient)
    gradient_img.save('test_images/gradient.jpg')

def test_model_predictions():
    """Test model predictions on different types of images"""
    try:
        # Create test images
        create_test_images()
        
        # Initialize model
        model = DeepfakeDetector(use_cuda=False)
        model_path = os.path.join("model", "deepfake_detector.pth")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return
        
        # Load model weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        # Test each image
        results = {}
        for img_file in os.listdir('test_images'):
            if img_file.endswith('.jpg'):
                img_path = os.path.join('test_images', img_file)
                prediction = model.predict(img_path)
                results[img_file] = prediction
                logger.info(f"\nResults for {img_file}:")
                logger.info(f"Prediction: {prediction['label']}")
                logger.info(f"Confidence: {prediction['confidence_percentage']}")
                logger.info(f"Real probability: {prediction['real_probability']:.4f}")
                logger.info(f"Fake probability: {prediction['fake_probability']:.4f}")
        
        # Analyze results
        fake_count = sum(1 for pred in results.values() if pred['label'] == 'Fake')
        total_count = len(results)
        
        logger.info(f"\nSummary:")
        logger.info(f"Total images tested: {total_count}")
        logger.info(f"Images predicted as fake: {fake_count}")
        logger.info(f"Percentage predicted as fake: {(fake_count/total_count)*100:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
    finally:
        # Cleanup test images
        if os.path.exists('test_images'):
            shutil.rmtree('test_images')

if __name__ == '__main__':
    test_model_predictions()
