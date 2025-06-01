import os
from convert_to_torch import DeepfakeDetector
import logging
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_model(model_path: str, test_dir: str = "uploads", viz_dir: str = "visualizations"):
    """Test the improved model with visualization"""
    # Create visualization directory
    os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize model
    model = DeepfakeDetector(use_cuda=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Test images
    test_files = [
        'fake_183_253_4.jpg',  # Known fake
        'real_183_5.jpg',      # Known real
        'ai_2.jpeg',           # AI generated
        'Mark_Zuckerberg_at_the_37th_G8_Summit_in_Deauville_018_v1.jpg'  # Real photo
    ]
    
    results = []
    
    for filename in test_files:
        image_path = os.path.join(test_dir, filename)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
            
        # Get prediction with visualization
        viz_path = os.path.join(viz_dir, f"viz_{filename}")
        prediction, visualization = model.predict_with_visualization(image_path, viz_path)
        
        # Regular prediction
        detail_pred = model.predict(image_path)
        
        logger.info(f"\nResults for {filename}:")
        logger.info(f"Label: {detail_pred['label']}")
        logger.info(f"Confidence: {detail_pred['confidence']:.3f}")
        logger.info(f"Stability: {detail_pred['prediction_stability']:.3f}")
        logger.info(f"Attention Score: {detail_pred['attention_score']:.3f}")
        logger.info(f"Visualization saved to: {viz_path}")
        
        results.append(detail_pred)
    
    return results

if __name__ == "__main__":
    model_path = 'model/deepfake_detector.pth'
    results = test_improved_model(model_path)
