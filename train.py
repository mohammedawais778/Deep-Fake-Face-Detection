import os
import subprocess

# Ensure required directories exist
os.makedirs("data/faces", exist_ok=True)
os.makedirs("data/features", exist_ok=True)
os.makedirs("model", exist_ok=True)

steps = [
    ("Training image-based deepfake classifier...","train_image_model.py"),
    ("Extracting faces from videos...", "prepare_faces.py"),
    ("Extracting features from faces...", "prepare_features.py"),
    ("Splitting data and training LSTM model...", "split_and_train_lstm.py")
]

for message, script in steps:
    print("\n===", message)
    ret = subprocess.call(["python", script])
    if ret != 0:
        print(f"❌ Failed: {script}")
        break
else:
    print("\n✅ Training pipeline completed successfully.")
    print("Saved model: model/lstm_model_augmented.h5")
