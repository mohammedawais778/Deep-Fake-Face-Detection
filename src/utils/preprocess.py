
import cv2
import os
import numpy as np
from mtcnn import MTCNN

def extract_frames(video_path, max_frames=30):
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while count < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    
    cap.release()
    return frames

def detect_faces(frame):
    """Detect faces in a single frame using MTCNN."""
    detector = MTCNN()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    return faces

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image for model input."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Detect face
    faces = detect_faces(img)
    if not faces:
        raise ValueError("No faces detected in the image")
    
    # Get the first face
    x, y, w, h = faces[0]['box']
    face = img[y:y+h, x:x+w]
    
    # Resize to target size
    face = cv2.resize(face, target_size)
    
    # Convert to RGB and normalize
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255.0
    
    return face

def preprocess_video(video_path, target_size=(224, 224), max_frames=30):
    """Preprocess video for model input."""
    # Extract frames
    frames = extract_frames(video_path, max_frames)
    if not frames:
        raise ValueError("No frames could be extracted from the video")
    
    processed_frames = []
    
    for frame in frames:
        try:
            # Detect faces in frame
            faces = detect_faces(frame)
            if not faces:
                continue
                
            # Get the first face
            x, y, w, h = faces[0]['box']
            face = frame[y:y+h, x:x+w]
            
            # Resize and normalize
            face = cv2.resize(face, target_size)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype('float32') / 255.0
            
            processed_frames.append(face)
            
            # Stop if we have enough frames
            if len(processed_frames) >= max_frames:
                break
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    
    if not processed_frames:
        raise ValueError("No valid faces found in the video")
    
    return np.array(processed_frames)
