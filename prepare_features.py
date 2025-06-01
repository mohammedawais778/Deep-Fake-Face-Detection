import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
input_path = 'data/faces_grouped' 

X, y = [], []

for folder in sorted(os.listdir(input_path)):
    folder_path = os.path.join(input_path, folder)
    if not os.path.isdir(folder_path):
        continue

    label = 0 if 'fake' in folder.lower() else 1
    files = sorted(os.listdir(folder_path))[:10]
    if len(files) < 10:
        print(f"Skipping {folder}: not enough frames")
        continue

    sequence = []
    for fname in files:
        img_path = os.path.join(folder_path, fname)
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = preprocess_input(x)
        sequence.append(x)

    X.append(sequence)
    y.append(label)

if len(X) == 0:
    print("❌ No valid sequences found. Check your data/faces_grouped folder structure and frame counts.")
    exit()

X = np.array(X)  # (N, 10, 224, 224, 3)
y = np.array(y)

features = []
for i, seq in enumerate(X):
    print(f"Extracting features for video {i+1}/{len(X)}")
    preds = model.predict(seq, verbose=0)  # shape (10, 2048)
    features.append(preds)

features = np.array(features)  # (N, 10, 2048)
np.save('data/X.npy', features)
np.save('data/y.npy', y)
print("✅ Feature extraction complete. Saved to data/X.npy and data/y.npy")
