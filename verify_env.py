import sys
import os

def check_environment():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nPyTorch not installed")
    
    try:
        from PIL import Image
        import PIL
        print(f"\nPillow version: {PIL.__version__}")
    except ImportError:
        print("\nPillow not installed")
    
    try:
        import cv2
        print(f"\nOpenCV version: {cv2.__version__}")
    except ImportError:
        print("\nOpenCV not installed")
    
    try:
        import h5py
        print(f"\nh5py version: {h5py.__version__}")
    except ImportError:
        print("\nh5py not installed")

if __name__ == "__main__":
    check_environment()
