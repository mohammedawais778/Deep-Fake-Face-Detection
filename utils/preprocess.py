
import cv2
import os
from mtcnn import MTCNN

def extract_frames(video_path, output_folder, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frames = []

    while count < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1

    cap.release()
    return frames

def detect_faces(frames):
    detector = MTCNN()
    face_images = []

    for frame in frames:
        result = detector.detect_faces(frame)
        if result:
            x, y, w, h = result[0]['box']
            face = frame[y:y+h, x:x+w]
            face_images.append(cv2.resize(face, (224, 224)))

    return face_images
