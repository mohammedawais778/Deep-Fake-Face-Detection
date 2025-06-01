#  Deepfake Detection Web Application (Image + Video)

A full-stack deepfake detection system that allows users to upload images or videos and get real-time predictions on whether the media is real or manipulated using deep learning techniques (CNN for images, LSTM for videos).


## Features

-  **Detect deepfakes in both images and videos**
-  **LSTM-based video analysis** with ResNet embeddings
-  **CNN-based image classification** using fine-tuned ResNet50
-  Upload images/videos via a sleek HTML interface
-  Backend built with Flask for API and model inference
-  Outputs prediction labels and confidence percentages
-  Tested on FaceForensics++ dataset


## Architecture


         +--------------------------+
         |     User Frontend        |
         |  (HTML/CSS + JavaScript) |
         +-----------+--------------+
                     |
                     v
        +------------+--------------+
        |       Flask Backend       |
        | - File upload             |
        | - Inference routing       |
        | - REST API endpoints      |
        +------------+--------------+
                     |
     +---------------+------------------+
     |   Model Inference (predict.py)   |
     |  - CNN (Images)                  |
     |  - LSTM + ResNet (Videos)        |
     +---------------+------------------+
                     |
     +---------------+------------------+
     |     Trained Models (*.h5)        |
     +----------------------------------+



##  Model Performance

## Image Classifier (CNN - ResNet50)
- **Accuracy:** 50%
- **Precision:** 50%
- **Recall:** 50%
- **F1 Score:** 50%

## Video Classifier (LSTM)
- **Accuracy:** 33.33%
- **Precision:** 33%
- **Recall:** 50%
- **F1 Score:** 25%

---

##  Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector


2. Install Dependencies
pip install -r requirements.txt


3. Download Dataset
Download a subset from FaceForensics++
Place videos in:

data/original_sequences/youtube/raw/videos

data/manipulated_sequences/Face2Face/raw/videos


4. Preprocess and Train
bash
Copy
Edit
python prepare_faces.py
python prepare_features.py
python split_and_train_lstm.py
python train_image_model.py


5. Run the App
bash
Copy
Edit
python app.py
Open http://localhost:8000 in your browser.


🔍 Project Structure
graphql
Copy
Edit
deepfake_detector/
│
├── data/                   # Dataset: faces, features, images
├── model/                  # Saved CNN and LSTM models
├── utils/                  # Face extractor, preprocess tools
├── templates/              # HTML frontend (or embedded HTML)
├── app.py                  # Main Flask application
├── predict.py              # Model inference logic
├── train_image_model.py    # CNN training script
├── split_and_train_lstm.py # LSTM training script
├── prepare_faces.py        # Face extraction
├── prepare_features.py     # ResNet feature extraction
└── README.md



## Future Improvements
Deploy to cloud (e.g., Heroku, AWS, or Render)
Real-time webcam stream analysis
Detect multiple types of manipulations (FaceSwap, NeuralTextures, etc.)
Add user authentication and scan history


##Acknowledgements

FaceForensics++
MTCNN
ResNet50 - Keras
TensorFlow, Keras, OpenCV, Flask


## Author
Mohammed Awais
Data Science,AI,ML

