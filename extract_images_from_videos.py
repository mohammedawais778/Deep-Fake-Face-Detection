import os
import cv2
from mtcnn import MTCNN

def extract_faces_from_video(video_path, output_dir, label_prefix, max_frames=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    detector = MTCNN()
    count = 0
    frame_idx = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_faces(frame)
        if results:
            x, y, w, h = results[0]['box']
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            out_path = os.path.join(output_dir, f"{label_prefix}_{frame_idx}.jpg")
            cv2.imwrite(out_path, face)
            count += 1

        frame_idx += 1

    cap.release()

# Example usage:
real_videos_dir = "data/original_sequences/youtube/raw/videos"
fake_videos_dir = "data/manipulated_sequences/Face2Face/raw/videos"
output_real = "data/images/train/real"
output_fake = "data/images/train/fake"

for filename in os.listdir(real_videos_dir):
    if filename.endswith(".mp4"):
        extract_faces_from_video(os.path.join(real_videos_dir, filename), output_real, "real")

for filename in os.listdir(fake_videos_dir):
    if filename.endswith(".mp4"):
        extract_faces_from_video(os.path.join(fake_videos_dir, filename), output_fake, "fake")
