import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def convert_model():
    try:
        print("Loading original model...")
        model = load_model('model/image_model_augmented.h5')
        
        print("Saving model in TensorFlow format...")
        model.save('model/image_model_augmented_tf', save_format='tf')
        
        print("Model converted successfully!")
        print("New model saved at: model/image_model_augmented_tf")
    except Exception as e:
        print(f"Error converting model: {str(e)}")

if __name__ == "__main__":
    convert_model() 