import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from utils.preprocess import extract_frames, detect_faces
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import h5py
import json

class CustomDTypePolicy(tf.keras.mixed_precision.Policy):
    def __init__(self, name):
        super().__init__(name)

def load_model_with_compatibility(model_path):
    try:
        # First try loading with custom objects
        custom_objects = {
            'DTypePolicy': tf.keras.mixed_precision.Policy,
            'CustomDTypePolicy': CustomDTypePolicy
        }
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            # If that fails, try loading the model architecture and weights separately
            with h5py.File(model_path, 'r') as f:
                model_config = f.attrs.get('model_config')
                if model_config is not None:
                    if isinstance(model_config, bytes):
                        model_config = model_config.decode('utf-8')
                    model_config = json.loads(model_config)
                    
                    # Remove problematic arguments
                    if 'layers' in model_config:
                        for layer in model_config['layers']:
                            if 'config' in layer:
                                layer['config'].pop('batch_shape', None)
                                layer['config'].pop('synchronized', None)
                    
                    # Create model from config
                    model = tf.keras.models.model_from_config(model_config)
                    
                    # Load weights
                    model.load_weights(model_path)
                    return model
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

# Load models
try:
    model_cnn = load_model_with_compatibility('model/image_model_augmented.h5')
    model_lstm = load_model_with_compatibility('model/lstm_model_retrained.h5')
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Reuse ResNet for both video prediction and feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def predict_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model_cnn.predict(x)[0][0]
        label = "Real" if pred < 0.5 else "Deepfake"
        return {
            'success': True,
            'label': label,
            'confidence': float(pred) if label == 'Deepfake' else 1 - float(pred),
            'raw_confidence': float(pred),
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'label': 'Unknown',
            'confidence': 0.0,
            'raw_confidence': 0.0,
            'error': str(e)
        }

def predict_video(video_path):
    try:
        frames = extract_frames(video_path, None, max_frames=30)
        faces = detect_faces(frames)
        if len(faces) < 10:
            return {
                'success': False,
                'label': 'Unknown',
                'confidence': 0.0,
                'raw_confidence': 0.0,
                'error': 'Not enough detectable faces in video.'
            }
        faces = faces[:10]

        processed_faces = []
        for face in faces:
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(face.astype('float32'))
            processed_faces.append(face)

        processed_faces = np.array(processed_faces)
        features = resnet_model.predict(processed_faces, verbose=0)  
        features = np.expand_dims(features, axis=0)  

        pred = model_lstm.predict(features)[0][0]
        label = "Real" if pred < 0.5 else "Deepfake"
        return {
            'success': True,
            'label': label,
            'confidence': float(pred) if label == 'Deepfake' else 1 - float(pred),
            'raw_confidence': float(pred),
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'label': 'Unknown',
            'confidence': 0.0,
            'raw_confidence': 0.0,
            'error': str(e)
        }

def draw_face_box(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def predict_image_with_cnn(image_path):
    image = load_img(image_path, target_size=(224, 224))
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0) / 255.0
    pred = model_cnn.predict(array)[0][0]
    label = "🔴 Deepfake" if pred >= 0.5 else "🟢 Real"
    return f"Prediction: {label} ({pred*100:.2f}% confidence)"
