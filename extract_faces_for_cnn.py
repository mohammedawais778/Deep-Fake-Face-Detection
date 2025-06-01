# -*- coding: utf-8 -*-

import os
import cv2
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
import shutil

def extract_faces(video_dir, label, output_root, max_frames=10):
    detector = MTCNN()
    files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    os.makedirs(output_root, exist_ok=True)
    images = []

    for f in files:
        cap = cv2.VideoCapture(os.path.join(video_dir, f))
        count = 0
        while count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = detector.detect_faces(frame)
            if results:
                x, y, w, h = results[0]['box']
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                filename = f"{label}_{f.split('.')[0]}_{count}.jpg"
                filepath = os.path.join(output_root, filename)
                cv2.imwrite(filepath, face)
                images.append(filepath)
                count += 1
        cap.release()
    return images

# Paths
real_vid_dir = "data/original_sequences/youtube/raw/videos"
fake_vid_dir = "data/manipulated_sequences/Face2Face/raw/videos"
output_all = "data/images/all"

# Extract
real_imgs = extract_faces(real_vid_dir, "real", os.path.join(output_all, "real"))
fake_imgs = extract_faces(fake_vid_dir, "fake", os.path.join(output_all, "fake"))

# Split to train/val
def split_and_copy(source_folder, label):
    images = [os.path.join(source_folder, label, f) for f in os.listdir(os.path.join(source_folder, label))]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for dest, files in [("train", train_imgs), ("val", val_imgs)]:
        dest_folder = os.path.join("data/images", dest, label)
        os.makedirs(dest_folder, exist_ok=True)
        for f in files:
            shutil.copy(f, dest_folder)

split_and_copy(output_all, "real")
split_and_copy(output_all, "fake")
