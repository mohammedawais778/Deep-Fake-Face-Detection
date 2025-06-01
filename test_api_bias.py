import os
import requests
from PIL import Image
import numpy as np
import logging

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
    """Test model predictions through the Flask API"""
    try:
        # Create test images
        create_test_images()
        
        # Test each image
        results = {}
        for img_file in os.listdir('test_images'):
            if img_file.endswith('.jpg'):
                img_path = os.path.join('test_images', img_file)
                
                # Send request to Flask app
                with open(img_path, 'rb') as f:
                    files = {'file': (img_file, f, 'image/jpeg')}
                    response = requests.post('http://localhost:5000/api/detect', files=files)
                
                if response.status_code == 200:
                    prediction = response.json()
                    results[img_file] = prediction
                    logger.info(f"\nResults for {img_file}:")
                    logger.info(f"Prediction: {prediction.get('label', 'N/A')}")
                    logger.info(f"Confidence: {prediction.get('confidence_percentage', 'N/A')}")
                    logger.info(f"Real probability: {prediction.get('real_probability', 'N/A')}")
                    logger.info(f"Fake probability: {prediction.get('fake_probability', 'N/A')}")
                else:
                    logger.error(f"Error processing {img_file}: {response.text}")
        
        # Analyze results
        fake_count = sum(1 for pred in results.values() if pred.get('label') == 'Fake')
        total_count = len(results)
        
        logger.info(f"\nSummary:")
        logger.info(f"Total images tested: {total_count}")
        logger.info(f"Images predicted as fake: {fake_count}")
        logger.info(f"Percentage predicted as fake: {(fake_count/total_count)*100:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
    finally:
        # Don't cleanup test images yet so we can inspect them
        pass

if __name__ == '__main__':
    test_model_predictions()
