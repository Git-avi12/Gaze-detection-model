Gaze Detection and Object Monitoring System
This project implements a real-time gaze detection and object monitoring system using classical computer vision and machine learning techniques. 
It was developed as part of my internship at Hysteresis Pvt Ltd for integration into a proctoring portal, enabling automated monitoring of user attention and device usage.

Features:
Blink Detection using Eye Aspect Ratio (EAR)
Gaze Direction Estimation (Left, Right, Up, Down, Center)
Stare Detection – identifies prolonged fixation at the same spot
Head Tilt Recognition based on facial landmarks
Object Detection for identifying devices (phone, laptop, etc.) using TensorFlow Object Detection API

Tech Stack:
Python
OpenCV – image processing and real-time video analysis
dlib – facial landmark detection (68-point predictor)
NumPy & SciPy – numerical operations and EAR calculation
TensorFlow – pre-trained SSD MobileNet for object detection

Project Structure
gaze_detection.py – Main script for gaze, blink, and head tilt detection
shape_predictor_68_face_landmarks.dat – Pre-trained dlib model for facial landmarks
frozen_inference_graph.pb – Pre-trained TensorFlow SSD MobileNet model

How to Run:

Clone this repository:

git clone https://github.com/<your-username>/gaze-detection.git
cd gaze-detection

Install dependencies:

pip install opencv-python dlib numpy scipy tensorflow

Download the required models:

dlib 68 face landmarks model

SSD MobileNet COCO model

Run the script:

python gaze_detection.py

Use Case

Designed for online proctoring applications, where it can help monitor user attentiveness (blinks, gaze, stares, head tilt) and detect use of restricted devices (e.g., phone, laptop).
