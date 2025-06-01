import os
import cv2
from mtcnn import MTCNN
from utils.preprocess import extract_frames
import numpy as np

real_path = 'data/original_sequences/youtube/raw/videos'
fake_path = 'data/manipulated_sequences/Face2Face/raw/videos'
face_output_dir = 'data/faces_grouped'
os.makedirs(face_output_dir, exist_ok=True)

detector = MTCNN()

video_id = 0
for label, path in enumerate([real_path, fake_path]):
    label_str = 'real' if label == 0 else 'fake'
    for video_file in os.listdir(path):
        if not video_file.endswith('.mp4'):
            continue
        video_path = os.path.join(path, video_file)
        frames = extract_frames(video_path, None, max_frames=30)
        faces = []
        for frame in frames:
            result = detector.detect_faces(frame)
            if result:
                x, y, w, h = result[0]['box']
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                faces.append(face)
        if len(faces) >= 10:
            folder_name = f"{label_str}_{video_id}"
            folder_path = os.path.join(face_output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            for i in range(10):
                filename = f"{i}.jpg"
                filepath = os.path.join(folder_path, filename)
                cv2.imwrite(filepath, faces[i])
            video_id += 1

print("✅ Face extraction complete. Saved to data/faces_grouped/")
