import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple

def generate_gradcam(model: torch.nn.Module, 
                    image_tensor: torch.Tensor,
                    target_layer: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM visualization for model decision"""
    
    # Register hooks
    gradients = []
    activations = []
    
    def save_gradient(grad):
        gradients.append(grad)
        
    def save_activation(module, input, output):
        activations.append(output)
    
    # Register forward and backward hooks
    handle_forward = target_layer.register_forward_hook(save_activation)
    handle_backward = target_layer.register_full_backward_hook(lambda m, i, o: save_gradient(o[0]))
    
    # Forward pass
    model.zero_grad()
    output = model(image_tensor)
    
    # Backward pass
    output.backward()
    
    # Get gradients and activations
    gradients = gradients[0]
    activations = activations[0]
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Calculate attention weights
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # Generate heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Convert to numpy
    heatmap = heatmap.detach().cpu().numpy()
    
    return heatmap

def apply_heatmap(image: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    """Apply heatmap overlay to the original image"""
    
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert to RGB if necessary
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Blend original image with heatmap
    superimposed = cv2.addWeighted(image_np, 0.7, heatmap, 0.3, 0)
    
    return superimposed

def visualize_attention(model: torch.nn.Module,
                       image_path: str,
                       save_path: str = None) -> Tuple[dict, np.ndarray]:
    """Generate attention visualization for model decision"""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = model.eval_transforms(image).unsqueeze(0)
    image_tensor = image_tensor.to(model.device)
    
    # Get last conv layer
    target_layer = None
    for module in model.features.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        raise ValueError("Could not find convolutional layer for visualization")
    
    # Generate heatmap
    heatmap = generate_gradcam(model, image_tensor, target_layer)
    
    # Apply heatmap to image
    visualization = apply_heatmap(image, heatmap)
    
    if save_path:
        # Save visualization
        cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    # Make prediction
    with torch.no_grad():
        prediction = model.predict(image_path)
    
    return prediction, visualization
