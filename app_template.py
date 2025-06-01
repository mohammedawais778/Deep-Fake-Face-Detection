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

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):  # 1GB threshold
        self.threshold_mb = threshold_mb
        self.lock = threading.Lock()
        self.monitoring = False
