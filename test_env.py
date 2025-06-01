import torch
import torchvision
import cv2
import PIL
import h5py
import sys
import os

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
print("OpenCV version:", cv2.__version__)
print("PIL version:", PIL.__version__)
print("Current working directory:", os.getcwd())
