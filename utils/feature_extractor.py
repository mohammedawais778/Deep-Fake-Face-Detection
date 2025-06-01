

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(faces):
    features = []
    for face in faces:
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        feature = resnet_model.predict(face)[0]
        features.append(feature)
    return np.array(features)
